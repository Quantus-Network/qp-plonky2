//! Generates `formal/Plonky2Bridge/Generated/Wrapper{N}.lean` — the composition of a recorded
//! `n = N` private-batch wrapper trace into `RPrivateBatch` (PLAN.md Step 8e).
//!
//! `build_private_batch_constraints` makes its gadget calls in a fixed order determined by
//! `n`: the per-leaf dummy checks, the first-real header scan, the per-leaf consistency
//! checks, the masked exit/amount candidates, the input and output totals, the fee
//! comparator, the exit-slot grouping loop, the pairwise nullifier-uniqueness constraints,
//! the dummy-nullifier hashes with the nullifier selects, and the odd-even switch network.
//! [`Shape::read`] walks the recorded calls with a cursor and checks each one against the
//! role it must play (which named targets it reads, which constants, how the outputs chain),
//! so a change in the wrapper's structure is a generator error rather than a proof that
//! quietly talks about different wires. [`render`] then emits the decode definitions (`row`,
//! `out`, `cands`, `slots`, `rounds`, …), the concrete-`N` unfolding of `scanRefL`, and a
//! `sound` proof that discharges every hypothesis of `Plonky2Bridge.private_batch_val_rows`
//! by rewriting with the exporter-generated per-call decode lemmas `privateBatchWrapper{N}_f{k}`.

use core::fmt::Write as _;

use plonky2::field::types::PrimeField64;
use plonky2::iop::target::Target;

use crate::circuit::lean_target;
use crate::gadget::{Call, Fact};
use crate::trace::{load, traces_dir, LoadedTrace};

/// The leaf public-input layout (`wormhole/aggregator/src/private_batch/circuit/constants.rs`).
mod layout {
    pub const ASSET_ID: usize = 0;
    pub const OUTPUT_AMOUNT_1: usize = 1;
    pub const OUTPUT_AMOUNT_2: usize = 2;
    pub const VOLUME_FEE_BPS: usize = 3;
    pub const NULLIFIER: usize = 4;
    pub const EXIT_ACCOUNT_1: usize = 8;
    pub const EXIT_ACCOUNT_2: usize = 12;
    pub const BLOCK_HASH: usize = 16;
    pub const BLOCK_NUMBER: usize = 20;
    pub const INPUT_AMOUNT: usize = 21;
    pub const LEAF_PI_LEN: usize = 22;
    pub const PRE_IMAGE_LEN: usize = 4;
    pub const FEE_DENOMINATOR: u64 = 10_000;
    pub const FEE_COMPLEMENT_BITS: usize = 14;
    pub const FEE_DIFF_BITS: usize = 52;
    pub const AMOUNT_BITS: usize = 32;
}

/// `bytes_digest_eq(x, y)` as recorded: four `is_equal` then the `and` tree (indices).
#[derive(Debug, Clone, Copy)]
struct DigestEq {
    eq: [usize; 4],
    and: [usize; 3],
}

/// One leaf's calls, by role (indices into the trace's calls).
#[derive(Debug, Clone)]
struct Leaf {
    pis: Vec<Target>,
    dummy: DigestEq,
    /// `not(is_dummy)`.
    is_real: usize,
    /// `not(found_real)`, `and(is_real, not_found)`: absent for leaf 0, where the builder
    /// folds them against the constant `false`.
    not_found: Option<usize>,
    take: Option<usize>,
    /// `or(found_real, is_real)`, when a later leaf reads it.
    found: Option<usize>,
    /// The six reference `select`s: block hash limbs, block number, fee.
    sel_block: [usize; 4],
    sel_number: usize,
    sel_fee: usize,
    /// Consistency: `bytes_digest_eq`, `or`, `connect` for the block hash; `connect` for the
    /// asset id (absent for leaf 0, the reference itself); `is_equal`, `or`, `connect` for
    /// the fee.
    block: DigestEq,
    block_or: usize,
    block_pin: usize,
    asset_pin: Option<usize>,
    fee_eq: usize,
    fee_or: usize,
    fee_pin: usize,
    /// Masked candidates: exit limbs and amount of each of the two slots.
    mask_exit: [[usize; 4]; 2],
    mask_amount: [usize; 2],
    /// Masked input amount and its accumulation (absent for leaf 0, folded).
    mask_input: usize,
    add_input: Option<usize>,
    /// Dummy nullifier: inner hash, outer hash, the four nullifier selects.
    hash_inner: usize,
    hash_outer: usize,
    null_sel: [usize; 4],
}

/// One `(i, j)` nullifier-uniqueness constraint.
#[derive(Debug, Clone, Copy)]
struct Pair {
    i: usize,
    j: usize,
    not_i: usize,
    not_j: usize,
    both: usize,
    eq: DigestEq,
    col: usize,
    pin: usize,
}

/// One exit slot of the grouping loop.
#[derive(Debug, Clone)]
struct Slot {
    /// Per earlier slot: `bytes_digest_eq` and the `or` into the duplicate flag (absent for
    /// the first, folded against `false`).
    dedup: Vec<(DigestEq, Option<usize>)>,
    /// Per candidate: `bytes_digest_eq`, `select(match, amount, 0)` and the `add` into the
    /// accumulator (absent for the first, folded against `zero`).
    matches: Vec<(DigestEq, usize, Option<usize>)>,
    sum_sel: usize,
    exit_sel: [usize; 4],
}

/// One switch of the odd-even network.
#[derive(Debug, Clone, Copy)]
struct Switch {
    assert: usize,
    /// Per limb: the `select`s producing routed position `i` and `i + 1`.
    sel: [[usize; 2]; 4],
}

/// The recorded wrapper, indexed by role.
#[derive(Debug, Clone)]
pub struct Shape {
    pub n: usize,
    leaves: Vec<Leaf>,
    /// The output-total `add`s over the `2n` masked amounts (the first is folded).
    add_output: Vec<Option<usize>>,
    /// `sub`, `range_check`, `mul`, `mul`, `sub`, `range_check` of the fee comparator.
    fee: [usize; 6],
    slots: Vec<Slot>,
    pairs: Vec<Pair>,
    /// The switches, grouped by round.
    rounds: Vec<Vec<Switch>>,
    switches: Vec<Target>,
    one: Target,
    zero: Target,
    total: Target,
    ten_thousand: Target,
    fee_ref: Target,
    number_ref: Target,
    block_ref: [Target; 4],
    total_in: Target,
    total_out: Target,
    /// `(target, value)` in the order of the export's `constants`.
    constants: Vec<(Target, u64)>,
    calls: Vec<Call>,
}

fn out_of(f: &Fact) -> Option<Target> {
    match *f {
        Fact::Select { out, .. }
        | Fact::Not { out, .. }
        | Fact::And { out, .. }
        | Fact::Or { out, .. }
        | Fact::Add { out, .. }
        | Fact::Sub { out, .. }
        | Fact::Mul { out, .. } => Some(out),
        _ => None,
    }
}

/// Walks the recorded calls in order, checking each against its expected role.
struct Cursor<'a> {
    calls: &'a [Call],
    k: usize,
}

impl Cursor<'_> {
    fn next(&mut self, what: &str) -> Result<(usize, &Fact), String> {
        let k = self.k;
        let call = self
            .calls
            .get(k)
            .ok_or_else(|| format!("call {k}: expected {what}, but the trace ends"))?;
        self.k += 1;
        Ok((k, &call.fact))
    }

    fn select(
        &mut self,
        b: Target,
        x: Target,
        y: Target,
        what: &str,
    ) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Select {
                b: b1,
                x: x1,
                y: y1,
                out,
            } if b1 == b && x1 == x && y1 == y => Ok((k, out)),
            ref f => Err(format!("call {k}: expected select for {what}, got {f:?}")),
        }
    }

    fn not(&mut self, b: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Not { b: b1, out } if b1 == b => Ok((k, out)),
            ref f => Err(format!("call {k}: expected not for {what}, got {f:?}")),
        }
    }

    fn and(&mut self, b1: Target, b2: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::And { b1: x, b2: y, out } if x == b1 && y == b2 => Ok((k, out)),
            ref f => Err(format!("call {k}: expected and for {what}, got {f:?}")),
        }
    }

    fn or(&mut self, b1: Target, b2: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Or { b1: x, b2: y, out } if x == b1 && y == b2 => Ok((k, out)),
            ref f => Err(format!("call {k}: expected or for {what}, got {f:?}")),
        }
    }

    fn is_equal(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::IsEqual {
                x: x1,
                y: y1,
                equal,
                ..
            } if x1 == x && y1 == y => Ok((k, equal)),
            ref f => Err(format!("call {k}: expected is_equal for {what}, got {f:?}")),
        }
    }

    fn connect(&mut self, x: Target, y: Target, what: &str) -> Result<usize, String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Connect { x: x1, y: y1 } if x1 == x && y1 == y => Ok(k),
            ref f => Err(format!("call {k}: expected connect for {what}, got {f:?}")),
        }
    }

    fn add(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Add { x: x1, y: y1, out } if x1 == x && y1 == y => Ok((k, out)),
            ref f => Err(format!("call {k}: expected add for {what}, got {f:?}")),
        }
    }

    fn sub(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Sub { x: x1, y: y1, out } if x1 == x && y1 == y => Ok((k, out)),
            ref f => Err(format!("call {k}: expected sub for {what}, got {f:?}")),
        }
    }

    fn mul(&mut self, x: Target, y: Target, what: &str) -> Result<(usize, Target), String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::Mul { x: x1, y: y1, out } if x1 == x && y1 == y => Ok((k, out)),
            ref f => Err(format!("call {k}: expected mul for {what}, got {f:?}")),
        }
    }

    fn range_check(&mut self, x: Target, bits: usize, what: &str) -> Result<usize, String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::RangeCheck { x: x1, bits: b } if x1 == x && b == bits => Ok(k),
            ref f => Err(format!(
                "call {k}: expected range_check {bits} for {what}, got {f:?}"
            )),
        }
    }

    fn poseidon2(&mut self, inputs: [Target; 4], what: &str) -> Result<(usize, usize), String> {
        let (k, f) = self.next(what)?;
        match f {
            Fact::Poseidon2 { rows, inputs: i } if *i == inputs => match rows[..] {
                [row] => Ok((k, row)),
                _ => Err(format!(
                    "call {k}: expected a one-block poseidon2 for {what}, got rows {rows:?}"
                )),
            },
            f => Err(format!(
                "call {k}: expected poseidon2 for {what}, got {f:?}"
            )),
        }
    }

    fn assert_bool(&mut self, b: Target, what: &str) -> Result<usize, String> {
        let (k, f) = self.next(what)?;
        match *f {
            Fact::AssertBool { b: b1 } if b1 == b => Ok(k),
            ref f => Err(format!(
                "call {k}: expected assert_bool for {what}, got {f:?}"
            )),
        }
    }

    /// `bytes_digest_eq(x, y)`: four `is_equal` then `and(e0, e1)`, `and(e2, e3)`, `and(·, ·)`.
    fn digest_eq(
        &mut self,
        x: [Target; 4],
        y: [Target; 4],
        what: &str,
    ) -> Result<(DigestEq, Target), String> {
        let mut eq = [0usize; 4];
        let mut flags = [x[0]; 4];
        for l in 0..4 {
            let (k, e) = self.is_equal(x[l], y[l], &format!("{what} limb {l}"))?;
            eq[l] = k;
            flags[l] = e;
        }
        let (ka, fa) = self.and(flags[0], flags[1], &format!("{what} and(e0, e1)"))?;
        let (kb, fb) = self.and(flags[2], flags[3], &format!("{what} and(e2, e3)"))?;
        let (kc, m) = self.and(fa, fb, &format!("{what} and-root"))?;
        Ok((
            DigestEq {
                eq,
                and: [ka, kb, kc],
            },
            m,
        ))
    }
}

fn limbs(pis: &[Target], base: usize) -> [Target; 4] {
    [pis[base], pis[base + 1], pis[base + 2], pis[base + 3]]
}

/// The switch layout of `permute_digests4` over `n` positions: round `r` pairs the
/// positions `(i, i + 1)` for `i ≡ r (mod 2)`.
fn switch_positions(n: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|r| (r % 2..n).step_by(2).filter(|i| i + 1 < n).collect())
        .collect()
}

impl Shape {
    pub fn read(t: &LoadedTrace) -> Result<Shape, String> {
        let named = |name: &str| -> Result<&Vec<Target>, String> {
            t.ex.named
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, ts)| ts)
                .ok_or_else(|| format!("trace has no named targets {name:?}"))
        };
        let n =
            t.ex.named
                .iter()
                .filter(|(name, _)| name.starts_with("leaf_pis_"))
                .count();
        if n < 2 {
            return Err(format!("expected at least two leaf_pis_i, found {n}"));
        }
        let pis: Vec<Vec<Target>> = (0..n)
            .map(|i| named(&format!("leaf_pis_{i}")).cloned())
            .collect::<Result<_, _>>()?;
        if pis.iter().any(|p| p.len() != layout::LEAF_PI_LEN) {
            return Err("leaf_pis_i is not 22 targets".into());
        }
        let pres: Vec<Vec<Target>> = (0..n)
            .map(|i| named(&format!("dummy_pre_image_{i}")).cloned())
            .collect::<Result<_, _>>()?;
        if pres.iter().any(|p| p.len() != layout::PRE_IMAGE_LEN) {
            return Err("dummy_pre_image_i is not 4 targets".into());
        }
        let switches = named("switches")?.clone();
        let positions = switch_positions(n);
        let switch_count: usize = positions.iter().map(Vec::len).sum();
        if switches.len() != switch_count {
            return Err(format!(
                "expected {switch_count} switches for n={n}, found {}",
                switches.len()
            ));
        }

        let constants: Vec<(Target, u64)> =
            t.ex.constants
                .iter()
                .map(|(x, v)| (*x, v.to_canonical_u64()))
                .collect();
        let const_target = |v: u64| -> Result<Target, String> {
            let hits: Vec<Target> = constants
                .iter()
                .filter(|(_, c)| *c == v)
                .map(|(x, _)| *x)
                .collect();
            match hits.as_slice() {
                [x] => Ok(*x),
                _ => Err(format!(
                    "expected exactly one constant target = {v}, found {}",
                    hits.len()
                )),
            }
        };
        let one = const_target(1)?;
        let zero = const_target(0)?;
        let total_value = (2 * n) as u64;
        let total = const_target(total_value)?;
        let ten_thousand = const_target(layout::FEE_DENOMINATOR)?;
        let _two_pow_59 = const_target(1 << 59)?;
        if constants.len() != 5 {
            return Err(format!(
                "expected constants {{1, 0, {total_value}, 10000, 2^59}}, found {constants:?}"
            ));
        }

        let mut c = Cursor {
            calls: &t.calls,
            k: 0,
        };

        // A. Dummy checks.
        let mut leaves: Vec<Leaf> = Vec::with_capacity(n);
        let mut dummies: Vec<Target> = Vec::with_capacity(n);
        for i in 0..n {
            let (dummy, d) = c.digest_eq(
                limbs(&pis[i], layout::BLOCK_HASH),
                [zero; 4],
                &format!("dummy check of leaf {i}"),
            )?;
            dummies.push(d);
            leaves.push(Leaf {
                pis: pis[i].clone(),
                dummy,
                is_real: 0,
                not_found: None,
                take: None,
                found: None,
                sel_block: [0; 4],
                sel_number: 0,
                sel_fee: 0,
                block: DigestEq {
                    eq: [0; 4],
                    and: [0; 3],
                },
                block_or: 0,
                block_pin: 0,
                asset_pin: None,
                fee_eq: 0,
                fee_or: 0,
                fee_pin: 0,
                mask_exit: [[0; 4]; 2],
                mask_amount: [0; 2],
                mask_input: 0,
                add_input: None,
                hash_inner: 0,
                hash_outer: 0,
                null_sel: [0; 4],
            });
        }

        // B. First-real header scan.
        let mut found_prev = zero;
        let mut ref_prev: Vec<Target> = vec![zero; 6];
        for i in 0..n {
            let (k_real, is_real) = c.not(dummies[i], &format!("not(is_dummy_{i})"))?;
            let (k_nf, not_found) =
                c.not(found_prev, &format!("not(found_real) before leaf {i}"))?;
            let (k_take, take) =
                c.and(is_real, not_found, &format!("and(is_real_{i}, not_found)"))?;
            let folded = i == 0;
            if folded {
                if not_found != one {
                    return Err("leaf 0: not(false) should fold to the constant one".into());
                }
                if take != is_real {
                    return Err("leaf 0: and(is_real_0, one) should fold to is_real_0".into());
                }
            }
            let src = [
                pis[i][layout::BLOCK_HASH],
                pis[i][layout::BLOCK_HASH + 1],
                pis[i][layout::BLOCK_HASH + 2],
                pis[i][layout::BLOCK_HASH + 3],
                pis[i][layout::BLOCK_NUMBER],
                pis[i][layout::VOLUME_FEE_BPS],
            ];
            let mut ref_now = Vec::with_capacity(6);
            let mut ks = [0usize; 6];
            for (j, s) in src.iter().enumerate() {
                let (k, o) = c.select(
                    take,
                    *s,
                    ref_prev[j],
                    &format!("reference select {j} of leaf {i}"),
                )?;
                ks[j] = k;
                ref_now.push(o);
            }
            let (k_found, found) =
                c.or(found_prev, is_real, &format!("or(found_real, is_real_{i})"))?;
            if folded && found != is_real {
                return Err("leaf 0: or(false, is_real_0) should fold to is_real_0".into());
            }
            let leaf = &mut leaves[i];
            leaf.is_real = k_real;
            if !folded {
                leaf.not_found = Some(k_nf);
                leaf.take = Some(k_take);
            }
            if !folded && i + 1 < n {
                leaf.found = Some(k_found);
            }
            leaf.sel_block = [ks[0], ks[1], ks[2], ks[3]];
            leaf.sel_number = ks[4];
            leaf.sel_fee = ks[5];
            found_prev = found;
            ref_prev = ref_now;
        }
        let block_ref: [Target; 4] = [ref_prev[0], ref_prev[1], ref_prev[2], ref_prev[3]];
        let (number_ref, fee_ref) = (ref_prev[4], ref_prev[5]);
        let asset_ref = pis[0][layout::ASSET_ID];

        // C. Consistency checks.
        for i in 0..n {
            let d = dummies[i];
            let (block, m) = c.digest_eq(
                limbs(&pis[i], layout::BLOCK_HASH),
                block_ref,
                &format!("block consistency of leaf {i}"),
            )?;
            let (k_or, ok) = c.or(d, m, &format!("or(is_dummy_{i}, block_matches)"))?;
            let k_pin = c.connect(ok, one, &format!("connect(block_ok_{i}, one)"))?;
            let k_asset = c.connect(
                pis[i][layout::ASSET_ID],
                asset_ref,
                &format!("asset pin of leaf {i}"),
            )?;
            let (k_feq, feq) = c.is_equal(
                pis[i][layout::VOLUME_FEE_BPS],
                fee_ref,
                &format!("fee is_equal of leaf {i}"),
            )?;
            let (k_for, fok) = c.or(d, feq, &format!("or(is_dummy_{i}, fee_matches)"))?;
            let k_fpin = c.connect(fok, one, &format!("connect(fee_ok_{i}, one)"))?;
            let leaf = &mut leaves[i];
            leaf.block = block;
            leaf.block_or = k_or;
            leaf.block_pin = k_pin;
            leaf.asset_pin = (i > 0).then_some(k_asset);
            leaf.fee_eq = k_feq;
            leaf.fee_or = k_for;
            leaf.fee_pin = k_fpin;
        }

        // D. Masked exit/amount candidates, in slot order.
        let mut cand_exit: Vec<[Target; 4]> = Vec::with_capacity(2 * n);
        let mut cand_amount: Vec<Target> = Vec::with_capacity(2 * n);
        for i in 0..n {
            let d = dummies[i];
            for o in 0..2 {
                let exit_base = if o == 0 {
                    layout::EXIT_ACCOUNT_1
                } else {
                    layout::EXIT_ACCOUNT_2
                };
                let amount = if o == 0 {
                    layout::OUTPUT_AMOUNT_1
                } else {
                    layout::OUTPUT_AMOUNT_2
                };
                let mut e = [zero; 4];
                for l in 0..4 {
                    let (k, out) = c.select(
                        d,
                        zero,
                        pis[i][exit_base + l],
                        &format!("masked exit {o} limb {l} of leaf {i}"),
                    )?;
                    leaves[i].mask_exit[o][l] = k;
                    e[l] = out;
                }
                let (k, m) = c.select(
                    d,
                    zero,
                    pis[i][amount],
                    &format!("masked amount {o} of leaf {i}"),
                )?;
                leaves[i].mask_amount[o] = k;
                cand_exit.push(e);
                cand_amount.push(m);
            }
        }

        // E. Input total.
        let mut acc = zero;
        for i in 0..n {
            let (k, m) = c.select(
                dummies[i],
                zero,
                pis[i][layout::INPUT_AMOUNT],
                &format!("masked input of leaf {i}"),
            )?;
            let (k_add, s) = c.add(acc, m, &format!("input total after leaf {i}"))?;
            leaves[i].mask_input = k;
            if i == 0 {
                if s != m {
                    return Err(
                        "leaf 0: add(zero, masked_input_0) should fold to masked_input_0".into(),
                    );
                }
            } else {
                leaves[i].add_input = Some(k_add);
            }
            acc = s;
        }
        let total_in = acc;

        // F. Output total.
        let mut acc = zero;
        let mut add_output = Vec::with_capacity(2 * n);
        for (k, &m) in cand_amount.iter().enumerate() {
            let (k_add, s) = c.add(acc, m, &format!("output total after slot {k}"))?;
            if k == 0 {
                if s != m {
                    return Err(
                        "slot 0: add(zero, masked_amount_0) should fold to masked_amount_0".into(),
                    );
                }
                add_output.push(None);
            } else {
                add_output.push(Some(k_add));
            }
            acc = s;
        }
        let total_out = acc;

        // G. Fee comparator.
        let (k_comp, comp) = c.sub(ten_thousand, fee_ref, "fee complement")?;
        let k_comp_rc = c.range_check(comp, layout::FEE_COMPLEMENT_BITS, "fee complement")?;
        let (k_rhs, rhs) = c.mul(total_out, ten_thousand, "total_out * 10000")?;
        let (k_lhs, lhs) = c.mul(total_in, comp, "total_in * complement")?;
        let (k_diff, diff) = c.sub(lhs, rhs, "fee difference")?;
        let k_diff_rc = c.range_check(diff, layout::FEE_DIFF_BITS, "fee difference")?;
        let fee = [k_comp, k_comp_rc, k_rhs, k_lhs, k_diff, k_diff_rc];

        // H. Exit-slot grouping.
        let mut slots = Vec::with_capacity(2 * n);
        let mut slot_pis: Vec<Target> = Vec::new();
        for k in 0..2 * n {
            let ek = cand_exit[k];
            let mut dedup = Vec::with_capacity(k);
            let mut dup = zero;
            for j in 0..k {
                let (de, m) = c.digest_eq(
                    cand_exit[j],
                    ek,
                    &format!("dedup slot {k} against slot {j}"),
                )?;
                let (k_or, d) = c.or(dup, m, &format!("dup flag of slot {k} after slot {j}"))?;
                if j == 0 {
                    if d != m {
                        return Err(format!(
                            "slot {k}: or(false, match_0) should fold to match_0"
                        ));
                    }
                    dedup.push((de, None));
                } else {
                    dedup.push((de, Some(k_or)));
                }
                dup = d;
            }
            let mut matches = Vec::with_capacity(2 * n);
            let mut acc = zero;
            for j in 0..2 * n {
                let (me, m) = c.digest_eq(
                    cand_exit[j],
                    ek,
                    &format!("match of slot {k} against candidate {j}"),
                )?;
                let (k_sel, s) = c.select(
                    m,
                    cand_amount[j],
                    zero,
                    &format!("contribution of candidate {j} to slot {k}"),
                )?;
                let (k_add, a) = c.add(acc, s, &format!("slot {k} sum after candidate {j}"))?;
                if j == 0 {
                    if a != s {
                        return Err(format!(
                            "slot {k}: add(zero, contribution_0) should fold to contribution_0"
                        ));
                    }
                    matches.push((me, k_sel, None));
                } else {
                    matches.push((me, k_sel, Some(k_add)));
                }
                acc = a;
            }
            let (k_sum, sum) = c.select(dup, zero, acc, &format!("masked sum of slot {k}"))?;
            let mut exit_sel = [0usize; 4];
            let mut exit_out = [zero; 4];
            for l in 0..4 {
                let (ks, o) = c.select(
                    dup,
                    zero,
                    ek[l],
                    &format!("masked exit limb {l} of slot {k}"),
                )?;
                exit_sel[l] = ks;
                exit_out[l] = o;
            }
            c.range_check(sum, layout::AMOUNT_BITS, &format!("sum of slot {k}"))?;
            slot_pis.push(sum);
            slot_pis.extend(exit_out);
            slots.push(Slot {
                dedup,
                matches,
                sum_sel: k_sum,
                exit_sel,
            });
        }

        // I. Nullifier uniqueness.
        let mut pairs = Vec::new();
        for i in 0..n {
            let (k_not_i, nr_i) =
                c.not(dummies[i], &format!("not(is_dummy_{i}) for uniqueness"))?;
            for j in i + 1..n {
                let (k_not_j, nr_j) =
                    c.not(dummies[j], &format!("not(is_dummy_{j}) against leaf {i}"))?;
                let (k_both, both) = c.and(nr_i, nr_j, &format!("both_real({i}, {j})"))?;
                let (eq, m) = c.digest_eq(
                    limbs(&pis[i], layout::NULLIFIER),
                    limbs(&pis[j], layout::NULLIFIER),
                    &format!("nullifier equality ({i}, {j})"),
                )?;
                let (k_col, col) = c.and(both, m, &format!("collision({i}, {j})"))?;
                let k_pin = c.connect(col, zero, &format!("connect(collision({i}, {j}), zero)"))?;
                pairs.push(Pair {
                    i,
                    j,
                    not_i: k_not_i,
                    not_j: k_not_j,
                    both: k_both,
                    eq,
                    col: k_col,
                    pin: k_pin,
                });
            }
        }

        // J. Dummy nullifiers and the nullifier selects.
        let mut current: Vec<[Target; 4]> = Vec::with_capacity(n);
        for i in 0..n {
            let pre: [Target; 4] = pres[i].clone().try_into().unwrap();
            let (k_in, row_in) = c.poseidon2(pre, &format!("inner dummy hash of leaf {i}"))?;
            let inner: [Target; 4] = [
                Target::wire(row_in, 12),
                Target::wire(row_in, 13),
                Target::wire(row_in, 14),
                Target::wire(row_in, 15),
            ];
            let (k_out, row_out) = c.poseidon2(inner, &format!("outer dummy hash of leaf {i}"))?;
            let mut sel = [0usize; 4];
            let mut outs = [zero; 4];
            for l in 0..4 {
                let (k, o) = c.select(
                    dummies[i],
                    Target::wire(row_out, 12 + l),
                    pis[i][layout::NULLIFIER + l],
                    &format!("nullifier select limb {l} of leaf {i}"),
                )?;
                sel[l] = k;
                outs[l] = o;
            }
            leaves[i].hash_inner = k_in;
            leaves[i].hash_outer = k_out;
            leaves[i].null_sel = sel;
            current.push(outs);
        }

        // K. The odd-even switch network.
        let mut rounds = Vec::with_capacity(n);
        let mut s = 0usize;
        for (r, pos) in positions.iter().enumerate() {
            let mut round = Vec::with_capacity(pos.len());
            for &i in pos {
                let sw = switches[s];
                let assert = c.assert_bool(sw, &format!("switch {s} (round {r}, position {i})"))?;
                let mut sel = [[0usize; 2]; 4];
                let mut lo = [zero; 4];
                let mut hi = [zero; 4];
                for l in 0..4 {
                    let (k0, o0) = c.select(
                        sw,
                        current[i + 1][l],
                        current[i][l],
                        &format!("switch {s} routed {i} limb {l}"),
                    )?;
                    let (k1, o1) = c.select(
                        sw,
                        current[i][l],
                        current[i + 1][l],
                        &format!("switch {s} routed {} limb {l}", i + 1),
                    )?;
                    sel[l] = [k0, k1];
                    lo[l] = o0;
                    hi[l] = o1;
                }
                current[i] = lo;
                current[i + 1] = hi;
                round.push(Switch { assert, sel });
                s += 1;
            }
            rounds.push(round);
        }
        if c.k != t.calls.len() {
            return Err(format!(
                "{} trailing gadget calls after the switch network",
                t.calls.len() - c.k
            ));
        }

        // Public inputs.
        let mut public_inputs: Vec<Target> = vec![total, asset_ref, fee_ref];
        public_inputs.extend(block_ref);
        public_inputs.push(number_ref);
        public_inputs.extend(slot_pis);
        for digest in &current {
            public_inputs.extend(digest);
        }
        let recorded = &t.ex.public_inputs;
        if recorded.len() < public_inputs.len()
            || recorded[..public_inputs.len()] != public_inputs[..]
            || recorded[public_inputs.len()..].iter().any(|x| *x != zero)
        {
            return Err("the recorded public inputs are not [total, asset, fee, block, number, slots…, nullifiers…, zero…]".into());
        }

        Ok(Shape {
            n,
            leaves,
            add_output,
            fee,
            slots,
            pairs,
            rounds,
            switches,
            one,
            zero,
            total,
            ten_thousand,
            fee_ref,
            number_ref,
            block_ref,
            total_in,
            total_out,
            constants,
            calls: t.calls.clone(),
        })
    }

    fn fact(&self, k: usize) -> &Fact {
        &self.calls[k].fact
    }

    fn out(&self, k: usize) -> Target {
        out_of(self.fact(k)).expect("fact with an output")
    }

    /// Whether fact `k` mentions the `zero` constant (so its statement is rewritten with `kzero`).
    fn mentions_zero(&self, k: usize) -> bool {
        let z = self.zero;
        match *self.fact(k) {
            Fact::Select { b, x, y, out } => [b, x, y, out].contains(&z),
            Fact::Not { b, out } => [b, out].contains(&z),
            Fact::And { b1, b2, out } | Fact::Or { b1, b2, out } => [b1, b2, out].contains(&z),
            Fact::Add { x, y, out } | Fact::Sub { x, y, out } | Fact::Mul { x, y, out } => {
                [x, y, out].contains(&z)
            }
            Fact::AssertBool { b } => b == z,
            Fact::IsEqual { x, y, equal, inv } => [x, y, equal, inv].contains(&z),
            Fact::RangeCheck { x, .. } | Fact::SplitLe { x, .. } => x == z,
            Fact::Connect { x, y } => [x, y].contains(&z),
            Fact::Poseidon2 { ref inputs, .. } => inputs.contains(&z),
        }
    }

    /// The per-call decode lemma of fact `k`, applied to the wrapper hypotheses.
    fn projection(&self, k: usize) -> String {
        let name = format!("privateBatchWrapper{}_f{k}", self.n);
        if self.fact(k).is_poseidon2() {
            format!("{name} perm a h hp")
        } else {
            format!("{name} a h")
        }
    }

    fn is_equal_witness(&self, k: usize) -> (Target, Target) {
        match *self.fact(k) {
            Fact::IsEqual { equal, inv, .. } => (equal, inv),
            _ => unreachable!(),
        }
    }
}

fn a(t: Target) -> String {
    format!("a ({})", lean_target(t))
}

/// `![a t0, a t1, a t2, a t3]`.
fn vec4(ts: &[Target]) -> String {
    format!(
        "![{}]",
        ts.iter().map(|t| a(*t)).collect::<Vec<_>>().join(", ")
    )
}

/// `fun j => a (![t0, t1, t2, t3] j)`.
fn digest_fun(ts: &[Target]) -> String {
    format!(
        "fun j => a (![{}] j)",
        ts.iter()
            .map(|t| format!("Target{}", lean_target(*t)))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Render the bridge module for a recorded private-batch wrapper.
pub fn render(shape: &Shape) -> String {
    let n = shape.n;
    let s2 = 2 * n;
    let circuit = format!("privateBatchWrapper{n}");
    let leaves = &shape.leaves;
    let mut o = String::new();
    macro_rules! w {
        ($($arg:tt)*) => { writeln!(o, $($arg)*).unwrap() }
    }

    w!("/-");
    w!("  AUTO-GENERATED by `constraint-exporter` (`src/private_wrapper.rs`) from the recorded");
    w!("  `n = {n}` private-batch wrapper trace `private_batch_wrapper_n{n}.json`. Do not edit;");
    w!("  regenerate with `cargo run -p qp-plonky2-constraint-exporter --bin export-constraints`.");
    w!("");
    w!("  Step 8e — `Generated/PrivateBatchWrapper{n}.lean` proves, from `Satisfies` on the wiring the");
    w!(
        "  real `CircuitBuilder` emitted plus `Poseidon2Rows perm`, the meaning of each of the {}",
        shape.calls.len()
    );
    w!("  gadget calls `build_private_batch_constraints` made over {n} leaves. This module reads the");
    w!("  spec objects off the named targets and public inputs (`row`, `leaves`, `us`, `out`) and");
    w!("  discharges every hypothesis of `private_batch_val_rows` (`Plonky2Bridge/PrivateBatch.lean`),");
    w!("  so `private_batch_end_to_end` is restated on the wiring with `leaf_proof_sound` as its only");
    w!("  axiom (`end_to_end_wired`).");
    w!("-/");
    w!("import Plonky2Bridge.PrivateBatch");
    w!("import Plonky2Spec.Generated.PrivateBatchWrapper{n}");
    w!("");
    w!("namespace Plonky2Bridge.Wrapper{n}");
    w!("");
    w!("open Plonky2Spec (IsBool bselect band bnot bor scanStep rangeCheck feeDen FeeCheck network digestEq");
    w!("  Digest4)");
    w!("open Plonky2Spec.Wiring");
    w!("open Plonky2Spec.Generated");
    w!("open Plonky2Spec.Poseidon2 (St)");
    w!("open Plonky2Spec.Sponge (spongeHash)");
    w!("open WormholeSpec (Digest Felt LeafPublic PrivateBatchOutput ExitSlot inRange RPrivateBatch");
    w!("  maskedOutputTotal realLeaves realNullifiers rawOutputTotal outputExitTotal");
    w!("  RPrivateBatch_value_conservation RPrivateBatch_settles_distinct_spends LeafWitness Rleaf");
    w!("  LeafProofAccepted leaf_proof_sound)");
    w!("");
    w!("variable {{p : ℕ}} [Fact p.Prime]");
    w!("");
    w!("set_option linter.unusedSimpArgs false");
    w!("");
    w!("/-! ### Reading the spec objects off the wiring -/");
    w!("");
    w!("/-- The 22 leaf public inputs of leaf `i` (`leaf_pis_i`). -/");
    w!("def leafPis : Fin {n} → Fin 22 → Target");
    for i in 0..n {
        w!("  | {i} => {circuit}.leaf_pis_{i}");
    }
    w!("");
    w!("/-- The 4-felt dummy-nullifier preimage of leaf `i` (`dummy_pre_image_i`). -/");
    w!("def preimage : Fin {n} → Fin 4 → Target");
    for i in 0..n {
        w!("  | {i} => {circuit}.dummy_pre_image_{i}");
    }
    w!("");
    w!("/-- The field wires of leaf `i` the wrapper reads, with the `is_equal` witnesses of its");
    w!("    `bytes_digest_eq(block_hash, 0)` dummy check, its preimage and the outer dummy-hash");
    w!("    output. -/");
    w!("def row (a : Assignment p) : Fin {n} → LeafRow p");
    for (i, leaf) in leaves.iter().enumerate() {
        let (eqs, invs): (Vec<Target>, Vec<Target>) = leaf
            .dummy
            .eq
            .iter()
            .map(|&k| shape.is_equal_witness(k))
            .unzip();
        let row_out = shape.fact(leaf.hash_outer).digest_row().unwrap();
        w!("  | {i} =>");
        w!(
            "    {{ assetId := a (leafPis {i} {}), out1 := a (leafPis {i} {}), out2 := a (leafPis {i} {}),",
            layout::ASSET_ID,
            layout::OUTPUT_AMOUNT_1,
            layout::OUTPUT_AMOUNT_2
        );
        w!("      fee := a (leafPis {i} {})", layout::VOLUME_FEE_BPS);
        w!(
            "      nullifier := fun j => a (leafPis {i} ⟨{} + j, by omega⟩)",
            layout::NULLIFIER
        );
        w!(
            "      exit1 := fun j => a (leafPis {i} ⟨{} + j, by omega⟩)",
            layout::EXIT_ACCOUNT_1
        );
        w!(
            "      exit2 := fun j => a (leafPis {i} ⟨{} + j, by omega⟩)",
            layout::EXIT_ACCOUNT_2
        );
        w!(
            "      blockHash := fun j => a (leafPis {i} ⟨{} + j, by omega⟩)",
            layout::BLOCK_HASH
        );
        w!(
            "      blockNumber := a (leafPis {i} {}), inputAmount := a (leafPis {i} {})",
            layout::BLOCK_NUMBER,
            layout::INPUT_AMOUNT
        );
        w!("      dummyEq := {}", vec4(&eqs));
        w!("      dummyInv := {}", vec4(&invs));
        w!("      pre := fun j => a (preimage {i} j), dnull := fun j => a (.wire {row_out} (12 + j)) }}");
    }
    w!("");
    let row_list: Vec<String> = (0..n).map(|i| format!("row a {i}")).collect();
    w!(
        "def rows (a : Assignment p) : List (LeafRow p) := [{}]",
        row_list.join(", ")
    );
    let leaf_list: Vec<String> = (0..n).map(|i| format!("(row a {i}).leaf")).collect();
    w!("");
    w!("/-- The {n} children, decoded through `.val`. -/");
    w!(
        "def leaves (a : Assignment p) : List LeafPublic := [{}]",
        leaf_list.join(", ")
    );
    let u_list: Vec<String> = (0..n).map(|i| format!("(row a {i}).u")).collect();
    w!("/-- The {n} dummy-nullifier preimages, decoded through `.val`. -/");
    w!(
        "def us (a : Assignment p) : List (List Felt) := [{}]",
        u_list.join(", ")
    );
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("theorem rows_leaves (a : Assignment p) : (rows a).map LeafRow.leaf = leaves a := rfl");
    w!("omit [Fact p.Prime] in");
    w!("theorem rows_us (a : Assignment p) : (rows a).map LeafRow.u = us a := rfl");
    w!("");
    w!("/-- Public input `k` of the wrapper, decoded through `.val`. -/");
    w!("def pv (a : Assignment p) (k : ℕ) : Felt :=");
    w!("  (a (({circuit} p).publicInputs.getD k (.virt 0))).val");
    w!("");
    let header = 8;
    let slot_pvs: Vec<String> = (0..s2)
        .map(|k| {
            let b = header + 5 * k;
            format!(
                "⟨pv a {b}, ⟨pv a {}, pv a {}, pv a {}, pv a {}⟩⟩",
                b + 1,
                b + 2,
                b + 3,
                b + 4
            )
        })
        .collect();
    let null_base = header + 5 * s2;
    let null_pvs: Vec<String> = (0..n)
        .map(|i| {
            let b = null_base + 4 * i;
            format!("⟨pv a {b}, pv a {}, pv a {}, pv a {}⟩", b + 1, b + 2, b + 3)
        })
        .collect();
    w!("/-- The aggregated output at the `aggregated_output` layout: header `[num_exit_slots,");
    w!("    asset_id, volume_fee_bps, block_hash(4), block_number]`, `2N = {s2}` exit slots");
    w!("    `[sum, account(4)]`, `N = {n}` nullifiers. -/");
    w!("def out (a : Assignment p) : PrivateBatchOutput :=");
    w!("  {{ numExitSlots := pv a 0, assetId := pv a 1, volumeFeeBps := pv a 2,");
    w!("    blockHash := ⟨pv a 3, pv a 4, pv a 5, pv a 6⟩, blockNumber := pv a 7,");
    w!("    exitSlots := [{}],", slot_pvs.join(",\n      "));
    w!("    nullifiers := [{}] }}", null_pvs.join(",\n      "));
    w!("");
    // Candidates and emitted slots on the wires.
    let cand_exit: Vec<[Target; 4]> = leaves
        .iter()
        .flat_map(|leaf| {
            leaf.mask_exit.iter().map(|ks| {
                [
                    shape.out(ks[0]),
                    shape.out(ks[1]),
                    shape.out(ks[2]),
                    shape.out(ks[3]),
                ]
            })
        })
        .collect();
    let cand_amount: Vec<Target> = leaves
        .iter()
        .flat_map(|leaf| leaf.mask_amount.iter().map(|&k| shape.out(k)))
        .collect();
    let cand_list: Vec<String> = (0..s2)
        .map(|k| format!("({}, {})", digest_fun(&cand_exit[k]), a(cand_amount[k])))
        .collect();
    w!("/-- The `2N` masked `(exit, amount)` candidates, in slot order. -/");
    w!("def cands (a : Assignment p) : List (Digest4 p × ZMod p) :=");
    w!("  [{}]", cand_list.join(",\n   "));
    let slot_list: Vec<String> = shape
        .slots
        .iter()
        .map(|slot| {
            let outs: Vec<Target> = slot.exit_sel.iter().map(|&k| shape.out(k)).collect();
            format!("({}, {})", a(shape.out(slot.sum_sel)), digest_fun(&outs))
        })
        .collect();
    w!("");
    w!("/-- The `2N` emitted exit slots `(select(dup, 0, sum), select(dup, 0, exit))`. -/");
    w!("def slots (a : Assignment p) : List (SlotF p) :=");
    w!("  [{}]", slot_list.join(",\n   "));
    w!("");
    let round_list: Vec<String> = shape
        .rounds
        .iter()
        .map(|round| {
            let sws: Vec<String> = round
                .iter()
                .map(|sw| match *shape.fact(sw.assert) {
                    Fact::AssertBool { b } => a(b),
                    _ => unreachable!(),
                })
                .collect();
            format!("[{}]", sws.join(", "))
        })
        .collect();
    w!("/-- The switches of the `n = {n}` odd-even network, by round. -/");
    w!(
        "def rounds (a : Assignment p) : List (List (ZMod p)) := [{}]",
        round_list.join(", ")
    );
    w!("");

    // The scan in normal form.
    w!("/-! ### The {n}-row scan -/");
    w!("");
    w!("/-- `scanRefL` over {n} rows, as the circuit's `select` chain: `take_0` is `is_real_0`");
    w!("    (`and(is_real_0, not(false))` folds), `take_i` is `and(is_real_i, not(found_{{i-1}}))`. -/");
    let ts: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
    w!(
        "theorem scanRefL_{n} ({} : LeafRow p) (f : LeafRow p → ZMod p) :",
        ts.join(" ")
    );
    w!("    scanRefL [{}] f", ts.join(", "));
    let nf = scan_normal_form(n, &|i| format!("r{i}.isDummy"), &|i| format!("(f r{i})"));
    w!("      = {nf} := by");
    w!("  simp only [scanRefL, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, scanStep,");
    w!("    bor_zero_left, bnot_zero, band_one_right]");
    w!("");

    // Constants.
    w!("/-! ### Composition -/");
    w!("");
    w!("theorem consts (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    let cs: Vec<String> = shape
        .constants
        .iter()
        .map(|(t, c)| format!("{} = {c}", a(*t)))
        .collect();
    w!("    {} := by", cs.join(" ∧ "));
    w!("  have hc := h.2.2");
    w!("  simp only [{circuit}, List.forall_mem_cons, List.not_mem_nil, false_implies,");
    w!("    implies_true, and_true] at hc");
    w!("  exact hc");
    w!("");
    let const_names: Vec<String> = shape
        .constants
        .iter()
        .map(|(t, _)| {
            if *t == shape.one {
                "kone".to_string()
            } else if *t == shape.zero {
                "kzero".to_string()
            } else if *t == shape.total {
                "ktot".to_string()
            } else if *t == shape.ten_thousand {
                "kten".to_string()
            } else {
                "-".to_string()
            }
        })
        .collect();

    let d = |i: usize| a(shape.out(leaves[i].dummy.and[2]));
    let block_ref = digest_fun(&shape.block_ref);
    let fee_ref = a(shape.fee_ref);
    let number_ref = a(shape.number_ref);
    let asset_ref = a(leaves[0].pis[layout::ASSET_ID]);
    let total_in = a(shape.total_in);
    let total_out = a(shape.total_out);

    w!("set_option maxHeartbeats 4000000 in");
    w!("/-- **The recorded private-batch wrapper satisfies `RPrivateBatch`.** Every satisfying");
    w!("    assignment whose Poseidon2 rows compute `perm`, with the leaves' 32-bit ranges (which");
    w!("    the leaf proofs certify), decodes to an `RPrivateBatch` instance on the {n} children,");
    w!("    their preimages and the aggregated output. -/");
    w!("theorem sound (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hp : Poseidon2Rows perm ({circuit} p) a)");
    w!("    (h32 : ∀ q ∈ leaves a, inRange 32 q.inputAmount ∧ inRange 32 q.outputAmount1 ∧");
    w!("      inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps) :");
    w!("    RPrivateBatch (spongeRO perm) (leaves a) (us a) (out a) := by");
    w!("  obtain ⟨{}⟩ := consts a h", const_names.join(", "));

    // Projections.
    let mut proj_lines: Vec<String> = Vec::new();
    let mut proj = |name: String, k: usize| {
        proj_lines.push(format!("  have {name} := {}", shape.projection(k)));
        if shape.mentions_zero(k) {
            proj_lines.push(format!("  rw [kzero] at {name}"));
        }
    };
    let digest_eq_proj = |proj: &mut dyn FnMut(String, usize), prefix: &str, de: &DigestEq| {
        for l in 0..4 {
            proj(format!("{prefix}_{l}"), de.eq[l]);
        }
        proj(format!("{prefix}a"), de.and[0]);
        proj(format!("{prefix}b"), de.and[1]);
        proj(format!("{prefix}m"), de.and[2]);
    };
    for (i, leaf) in leaves.iter().enumerate() {
        digest_eq_proj(&mut proj, &format!("e{i}"), &leaf.dummy);
    }
    for (i, leaf) in leaves.iter().enumerate() {
        proj(format!("r{i}"), leaf.is_real);
        if let Some(k) = leaf.not_found {
            proj(format!("nf{i}"), k);
        }
        if let Some(k) = leaf.take {
            proj(format!("tk{i}"), k);
        }
        for j in 0..4 {
            proj(format!("sb{i}_{j}"), leaf.sel_block[j]);
        }
        proj(format!("sn{i}"), leaf.sel_number);
        proj(format!("sf{i}"), leaf.sel_fee);
        if let Some(k) = leaf.found {
            proj(format!("fo{i}"), k);
        }
    }
    for (i, leaf) in leaves.iter().enumerate() {
        digest_eq_proj(&mut proj, &format!("cb{i}"), &leaf.block);
        proj(format!("cbo{i}"), leaf.block_or);
        proj(format!("cbk{i}"), leaf.block_pin);
        if let Some(k) = leaf.asset_pin {
            proj(format!("ca{i}"), k);
        }
        proj(format!("cf{i}"), leaf.fee_eq);
        proj(format!("cfo{i}"), leaf.fee_or);
        proj(format!("cfk{i}"), leaf.fee_pin);
    }
    for (i, leaf) in leaves.iter().enumerate() {
        for o in 0..2 {
            for l in 0..4 {
                proj(format!("mE{}_{l}", 2 * i + o), leaf.mask_exit[o][l]);
            }
            proj(format!("mA{}", 2 * i + o), leaf.mask_amount[o]);
        }
    }
    for (i, leaf) in leaves.iter().enumerate() {
        proj(format!("mi{i}"), leaf.mask_input);
        if let Some(k) = leaf.add_input {
            proj(format!("ai{i}"), k);
        }
    }
    for (k, add) in shape.add_output.iter().enumerate() {
        if let Some(c) = add {
            proj(format!("ao{k}"), *c);
        }
    }
    proj("fc".into(), shape.fee[0]);
    proj("fcr".into(), shape.fee[1]);
    proj("fr".into(), shape.fee[2]);
    proj("fl".into(), shape.fee[3]);
    proj("fd".into(), shape.fee[4]);
    proj("fdr".into(), shape.fee[5]);
    for (k, slot) in shape.slots.iter().enumerate() {
        for (j, (de, or)) in slot.dedup.iter().enumerate() {
            digest_eq_proj(&mut proj, &format!("x{k}_{j}"), de);
            if let Some(c) = or {
                proj(format!("xo{k}_{j}"), *c);
            }
        }
        for (j, (de, sel, add)) in slot.matches.iter().enumerate() {
            digest_eq_proj(&mut proj, &format!("y{k}_{j}"), de);
            proj(format!("ys{k}_{j}"), *sel);
            if let Some(c) = add {
                proj(format!("yd{k}_{j}"), *c);
            }
        }
        proj(format!("os{k}"), slot.sum_sel);
        for l in 0..4 {
            proj(format!("oe{k}_{l}"), slot.exit_sel[l]);
        }
    }
    for pair in &shape.pairs {
        let (i, j) = (pair.i, pair.j);
        proj(format!("un{i}_{j}"), pair.not_i);
        proj(format!("um{i}_{j}"), pair.not_j);
        proj(format!("ub{i}_{j}"), pair.both);
        digest_eq_proj(&mut proj, &format!("q{i}_{j}"), &pair.eq);
        proj(format!("uc{i}_{j}"), pair.col);
        proj(format!("uz{i}_{j}"), pair.pin);
    }
    for (i, leaf) in leaves.iter().enumerate() {
        for l in 0..4 {
            proj(format!("s{i}_{l}"), leaf.null_sel[l]);
        }
    }
    for (r, round) in shape.rounds.iter().enumerate() {
        for (s, sw) in round.iter().enumerate() {
            proj(format!("ab{r}_{s}"), sw.assert);
            for l in 0..4 {
                proj(format!("w{r}_{s}_{l}_0"), sw.sel[l][0]);
                proj(format!("w{r}_{s}_{l}_1"), sw.sel[l][1]);
            }
        }
    }
    for line in &proj_lines {
        w!("{line}");
    }
    for (i, leaf) in leaves.iter().enumerate() {
        w!(
            "  obtain ⟨hi{i}_0, hi{i}_1, hi{i}_2, hi{i}_3⟩ := {}",
            shape.projection(leaf.hash_inner)
        );
        w!(
            "  obtain ⟨ho{i}_0, ho{i}_1, ho{i}_2, ho{i}_3⟩ := {}",
            shape.projection(leaf.hash_outer)
        );
    }
    w!("");
    for i in 0..n {
        w!(
            "  have hd{i} : {} = (row a {i}).isDummy := by rw [e{i}m, e{i}a, e{i}b]; rfl",
            d(i)
        );
    }
    let hds: Vec<String> = (0..n).map(|i| format!("hd{i}")).collect();

    // Membership.
    let alts: Vec<String> = (0..n).map(|i| format!("r = row a {i}")).collect();
    w!("  have hmem : ∀ r ∈ rows a, {} := by", alts.join(" ∨ "));
    w!("    intro r hr");
    w!("    simpa only [rows, List.mem_cons, List.not_mem_nil, or_false] using hr");
    let rfls = vec!["rfl"; n].join(" | ");

    // The scan.
    let xs: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    let take_wire = |i: usize| -> Target {
        match leaves[i].take {
            Some(k) => shape.out(k),
            None => shape.out(leaves[i].is_real),
        }
    };
    let mut chain = format!("bselect ({}) x0 0", a(take_wire(0)));
    for i in 1..n {
        chain = format!("bselect ({}) x{i} ({chain})", a(take_wire(i)));
    }
    let nf_rows = scan_normal_form(n, &|i| format!("(row a {i}).isDummy"), &|i| format!("x{i}"));
    w!("  have hsel : ∀ {} : ZMod p,", xs.join(" "));
    w!("      {chain}");
    w!("        = {nf_rows} := by");
    w!("    intro {}", xs.join(" "));
    let mut rws: Vec<String> = Vec::new();
    for i in (1..n).rev() {
        rws.push(format!("tk{i}"));
    }
    for i in (1..n).rev() {
        rws.push(format!("nf{i}"));
    }
    for i in (1..n).rev() {
        if leaves[i].found.is_some() {
            rws.push(format!("fo{i}"));
        }
    }
    for i in (0..n).rev() {
        rws.push(format!("r{i}"));
    }
    rws.extend(hds.iter().cloned());
    w!("    rw [{}]", rws.join(", "));
    let bw: Vec<String> = shape.block_ref.iter().map(|t| lean_target(*t)).collect();
    w!("  have hblockRef : ({block_ref}) = blockRefL (rows a) := by");
    w!("    funext j");
    w!("    simp only [blockRefL, rows, scanRefL_{n}]");
    w!("    fin_cases j");
    for j in 0..4 {
        let sels: Vec<String> = (0..n).rev().map(|i| format!("sb{i}_{j}")).collect();
        w!("    · show a ({}) = _", bw[j]);
        w!("      rw [{}, hsel]; rfl", sels.join(", "));
    }
    let sns: Vec<String> = (0..n).rev().map(|i| format!("sn{i}")).collect();
    w!("  have hnumRef : {number_ref} = scanRefL (rows a) fun r => r.blockNumber := by");
    w!("    rw [rows, scanRefL_{n}, {}, hsel]; rfl", sns.join(", "));
    let sfs: Vec<String> = (0..n).rev().map(|i| format!("sf{i}")).collect();
    w!("  have hfeeRef : {fee_ref} = scanRefL (rows a) fun r => r.fee := by");
    w!("    rw [rows, scanRefL_{n}, {}, hsel]; rfl", sfs.join(", "));

    // Candidates.
    for k in 0..s2 {
        let i = k / 2;
        let field = if k % 2 == 0 { "exit1" } else { "exit2" };
        let amount = if k % 2 == 0 { "out1" } else { "out2" };
        w!(
            "  have hE{k} : ({}) = fun j => bselect (row a {i}).isDummy 0 ((row a {i}).{field} j) := by",
            digest_fun(&cand_exit[k])
        );
        w!("    funext j");
        w!("    fin_cases j");
        for l in 0..4 {
            w!("    · show a ({}) = _", lean_target(cand_exit[k][l]));
            w!("      rw [mE{k}_{l}, hd{i}]; rfl");
        }
        w!(
            "  have hA{k} : {} = bselect (row a {i}).isDummy 0 (row a {i}).{amount} := by",
            a(cand_amount[k])
        );
        w!("    rw [mA{k}, hd{i}]; rfl");
    }
    let hes: Vec<String> = (0..s2).map(|k| format!("hE{k}")).collect();
    let has: Vec<String> = (0..s2).map(|k| format!("hA{k}")).collect();
    w!("  have hcands : cands a = candsL (rows a) := by");
    w!("    simp only [cands, candsL, rows, List.flatMap_cons, List.flatMap_nil, List.cons_append,");
    w!("      List.nil_append, List.append_nil]");
    w!("    rw [{}, {}]", hes.join(", "), has.join(", "));

    // The grouping loop.
    w!("  have hslots : SlotsOk [] (cands a) (slots a) := by");
    w!("    simp only [SlotsOk, SlotCheck, cands, slots, List.nil_append, List.cons_append]");
    let mut witnesses: Vec<String> = Vec::new();
    for (k, slot) in shape.slots.iter().enumerate() {
        let dup = if k == 0 {
            "0".to_string()
        } else {
            let last = &slot.dedup[k - 1];
            match last.1 {
                Some(c) => a(shape.out(c)),
                None => a(shape.out(last.0.and[2])),
            }
        };
        let acc = match slot.matches[s2 - 1].2 {
            Some(c) => a(shape.out(c)),
            None => a(shape.out(slot.matches[s2 - 1].1)),
        };
        let eqk = a(shape.out(slot.matches[k].0.and[2]));
        let dup_flags: Vec<String> = slot
            .dedup
            .iter()
            .map(|(de, _)| a(shape.out(de.and[2])))
            .collect();
        let m_earlier: Vec<String> = (0..k)
            .map(|j| a(shape.out(slot.matches[j].0.and[2])))
            .collect();
        let m_later: Vec<String> = (k + 1..s2)
            .map(|j| a(shape.out(slot.matches[j].0.and[2])))
            .collect();
        witnesses.push(format!(
            "⟨⟨{dup}, {acc}, {eqk}, [{}], [{}], [{}]⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩",
            dup_flags.join(", "),
            m_earlier.join(", "),
            m_later.join(", ")
        ));
    }
    w!("    refine ⟨{}, trivial⟩", witnesses.join(",\n      "));
    let digest_eq_witness = |shape: &Shape, prefix: &str, de: &DigestEq| -> String {
        let (eqs, invs): (Vec<Target>, Vec<Target>) =
            de.eq.iter().map(|&k| shape.is_equal_witness(k)).unzip();
        format!(
            "⟨{}, {}, fun j => by fin_cases j <;> assumption, by rw [{prefix}m, {prefix}a, {prefix}b]; rfl⟩",
            vec4(&eqs),
            vec4(&invs)
        )
    };
    for (k, slot) in shape.slots.iter().enumerate() {
        let forall2 = |items: Vec<String>| -> String {
            let mut t = "List.Forall₂.nil".to_string();
            for it in items.into_iter().rev() {
                t = format!("List.Forall₂.cons ({it}) ({t})");
            }
            t
        };
        w!("    · -- slot {k}: dedup flags");
        let dedups: Vec<String> = slot
            .dedup
            .iter()
            .enumerate()
            .map(|(j, (de, _))| digest_eq_witness(shape, &format!("x{k}_{j}"), de))
            .collect();
        w!("      exact {}", forall2(dedups));
        w!("    · -- slot {k}: matches against the earlier candidates");
        let earlier: Vec<String> = (0..k)
            .map(|j| digest_eq_witness(shape, &format!("y{k}_{j}"), &slot.matches[j].0))
            .collect();
        w!("      exact {}", forall2(earlier));
        w!("    · -- slot {k}: self match");
        w!(
            "      exact {}",
            digest_eq_witness(shape, &format!("y{k}_{k}"), &slot.matches[k].0)
        );
        w!("    · -- slot {k}: matches against the later candidates");
        let later: Vec<String> = (k + 1..s2)
            .map(|j| digest_eq_witness(shape, &format!("y{k}_{j}"), &slot.matches[j].0))
            .collect();
        w!("      exact {}", forall2(later));
        w!("    · -- slot {k}: duplicate flag");
        if k == 0 {
            w!("      simp only [List.foldl_nil]");
        } else {
            w!("      simp only [List.foldl_cons, List.foldl_nil, bor_zero_left]");
        }
        let ors: Vec<String> = (1..k).rev().map(|j| format!("xo{k}_{j}")).collect();
        if !ors.is_empty() {
            w!("      rw [{}]", ors.join(", "));
        }
        w!("    · -- slot {k}: accumulated sum");
        w!("      simp only [List.zipWith_cons_cons, List.zipWith_nil_left, List.zipWith_nil_right,");
        w!("        List.cons_append, List.nil_append, List.foldl_cons, List.foldl_nil, zero_add]");
        let mut rws: Vec<String> = (1..s2).rev().map(|j| format!("yd{k}_{j}")).collect();
        rws.extend((0..s2).map(|j| format!("ys{k}_{j}")));
        w!("      rw [{}]", rws.join(", "));
        w!("    · -- slot {k}: masked sum");
        w!("      exact os{k}");
        w!("    · -- slot {k}: masked exit");
        w!("      intro j; fin_cases j <;> assumption");
    }

    // Totals.
    for i in 0..n {
        w!(
            "  have hM{i} : {} = bselect (row a {i}).isDummy 0 (row a {i}).inputAmount := by",
            a(shape.out(leaves[i].mask_input))
        );
        w!("    rw [mi{i}, hd{i}]; rfl");
    }
    w!("  have hin : {total_in} = ((rows a).map fun r => bselect r.isDummy 0 r.inputAmount).sum := by");
    w!("    simp only [rows, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]");
    let mut rws: Vec<String> = (1..n).rev().map(|i| format!("ai{i}")).collect();
    rws.extend((0..n).map(|i| format!("hM{i}")));
    w!("    rw [{}]", rws.join(", "));
    if n > 2 {
        w!("    ring");
    }
    w!("  have hout : {total_out} =");
    w!("      ((rows a).map fun r => bselect r.isDummy 0 r.out1 + bselect r.isDummy 0 r.out2).sum := by");
    w!("    simp only [rows, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]");
    let mut rws: Vec<String> = (1..s2).rev().map(|k| format!("ao{k}")).collect();
    rws.extend(has.iter().cloned());
    w!("    rw [{}]", rws.join(", "));
    w!("    ring");

    // Fee.
    w!("  have hfc : FeeCheck ({fee_ref}) ({total_in}) ({total_out}) := by");
    w!("    refine ⟨?_, ?_⟩");
    w!("    · show rangeCheck (feeDen - {fee_ref}) 14");
    w!("      unfold feeDen; rw [Nat.cast_ofNat, ← kten, ← fc]; exact fcr");
    w!("    · show rangeCheck ({total_in} * (feeDen - {fee_ref}) - {total_out} * feeDen) 52");
    w!("      unfold feeDen; rw [Nat.cast_ofNat, ← kten, ← fc, ← fl, ← fr, ← fd]; exact fdr");

    // Nullifier network.
    let final_nulls = network_outputs(shape);
    let null_lits: Vec<String> = final_nulls
        .iter()
        .map(|dg| {
            format!(
                "⟨({}).val, ({}).val, ({}).val, ({}).val⟩",
                a(dg[0]),
                a(dg[1]),
                a(dg[2]),
                a(dg[3])
            )
        })
        .collect();
    w!("  have hnulls : (out a).nullifiers = (network (rounds a) ((rows a).map fun r => r.slot.sel)).map valDigest := by");
    w!(
        "    show ([{}] : List Digest) = _",
        null_lits.join(",\n      ")
    );
    let mut rws: Vec<String> = Vec::new();
    for (r, round) in shape.rounds.iter().enumerate().rev() {
        for (s, _) in round.iter().enumerate().rev() {
            for l in 0..4 {
                rws.push(format!("w{r}_{s}_{l}_0"));
                rws.push(format!("w{r}_{s}_{l}_1"));
            }
        }
    }
    for i in 0..n {
        for l in 0..4 {
            rws.push(format!("s{i}_{l}"));
        }
    }
    rws.extend(hds.iter().cloned());
    w!("    rw [{}]", rws.join(", "));
    w!("    rfl");

    // Uniqueness.
    for pair in &shape.pairs {
        let (i, j) = (pair.i, pair.j);
        w!("  have u{i}_{j} : UniqCheck (row a {i}).nullifier (row a {j}).nullifier (row a {i}).isDummy (row a {j}).isDummy :=");
        w!(
            "    ⟨{}, {}, by",
            a(shape.out(pair.eq.and[2])),
            digest_eq_witness(shape, &format!("q{i}_{j}"), &pair.eq)
        );
        w!("      rw [← hd{i}, ← hd{j}, ← un{i}_{j}, ← um{i}_{j}, ← ub{i}_{j}, ← uc{i}_{j}]; exact uz{i}_{j}⟩");
    }
    w!("  have hcol : (rows a).Pairwise fun r s => UniqCheck r.nullifier s.nullifier r.isDummy s.isDummy := by");
    w!("    unfold rows");
    let mut pw = "List.Pairwise.nil".to_string();
    for _ in 0..n {
        pw = format!("List.Pairwise.cons ?_ ({pw})");
    }
    w!("    refine {pw}");
    for i in 0..n {
        w!("    · intro s hs");
        if i + 1 == n {
            w!("      nomatch hs");
        } else {
            w!("      simp only [List.mem_cons, List.not_mem_nil, or_false] at hs");
            let alts = vec!["rfl"; n - i - 1].join(" | ");
            w!("      rcases hs with {alts}");
            for j in i + 1..n {
                w!("      · exact u{i}_{j}");
            }
        }
    }

    // Assembly.
    let total_value = 2 * n;
    w!("  have hR : RPrivateBatch (spongeRO perm) ((rows a).map LeafRow.leaf) ((rows a).map LeafRow.u) (out a) := by");
    w!("    refine private_batch_val_rows perm hpg (rows a) (rounds a) (out := out a)");
    w!("      (feeRef := {fee_ref}) (totalIn := {total_in}) (totalOut := {total_out})");
    w!("      (assetRef := {asset_ref}) (numRef := {number_ref}) (blockRef := {block_ref})");
    w!("      (cands := cands a) (slots := slots a) ?_ ?_ ?_ (by show {n} ≤ 64; decide) h32");
    w!("      hblockRef hnumRef hfeeRef ?_ ?_ ?_ rfl rfl rfl rfl ?_ hin hout hfc hcands hslots rfl hnulls hcol");
    w!("    · -- the switches are boolean");
    w!("      intro s hs");
    w!("      simp only [rounds, List.flatten_cons, List.flatten_nil, List.cons_append, List.nil_append,");
    w!("        List.append_nil, List.mem_cons, List.not_mem_nil, or_false] at hs");
    let sw_count = shape.switches.len();
    let sw_alts = vec!["rfl"; sw_count].join(" | ");
    if sw_count == 1 {
        w!("      rcases hs with rfl");
        w!("      assumption");
    } else {
        w!("      rcases hs with {sw_alts} <;> assumption");
    }
    w!("    · -- dummy checks");
    w!("      intro r hr");
    w!("      rcases hmem r hr with {rfls}");
    for _ in 0..n {
        w!("      · intro j; fin_cases j <;> assumption");
    }
    w!("    · -- dummy nullifiers");
    w!("      intro r hr");
    w!("      rcases hmem r hr with {rfls}");
    for leaf in leaves.iter() {
        let row_in = shape.fact(leaf.hash_inner).digest_row().unwrap();
        w!("      · exact ⟨fun j => a (.wire {row_in} (12 + j)), fun j => by fin_cases j <;> assumption,");
        w!("          fun j => by fin_cases j <;> assumption⟩");
    }
    w!("    · -- asset consistency");
    w!("      intro r hr");
    w!("      rcases hmem r hr with {rfls}");
    for i in 0..n {
        if i == 0 {
            w!("      · rfl");
        } else {
            w!("      · exact ca{i}");
        }
    }
    w!("    · -- fee consistency");
    w!("      intro r hr");
    w!("      rcases hmem r hr with {rfls}");
    for i in 0..n {
        w!("      · exact ⟨_, _, cf{i}, by rw [← hd{i}, ← cfo{i}, cfk{i}, kone]⟩");
    }
    w!("    · -- block consistency");
    w!("      intro r hr");
    w!("      rcases hmem r hr with {rfls}");
    for (i, leaf) in leaves.iter().enumerate() {
        let (eqs, invs): (Vec<Target>, Vec<Target>) = leaf
            .block
            .eq
            .iter()
            .map(|&k| shape.is_equal_witness(k))
            .unzip();
        w!(
            "      · refine ⟨{}, {}, fun j => by fin_cases j <;> assumption, ?_⟩",
            vec4(&eqs),
            vec4(&invs)
        );
        w!(
            "        show bor (row a {i}).isDummy (band (band ({}) ({})) (band ({}) ({}))) = 1",
            a(eqs[0]),
            a(eqs[1]),
            a(eqs[2]),
            a(eqs[3])
        );
        w!("        rw [← hd{i}, ← cb{i}a, ← cb{i}b, ← cb{i}m, ← cbo{i}, cbk{i}, kone]");
    }
    w!("    · -- the slot count");
    w!("      show ({}).val = 2 * {n}", a(shape.total));
    w!("      rw [ktot]");
    w!("      have hc : ({total_value} : ℕ) < p := lt_of_lt_of_le (by decide) hpg");
    w!("      rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]");
    w!("  rwa [rows_leaves, rows_us] at hR");
    w!("");
    w!("/-- **The capstone on the recorded wiring.** `private_batch_end_to_end` with its decode");
    w!("    hypotheses discharged by `sound`: a satisfying assignment of the `n = {n}` private-batch");
    w!("    wrapper whose recursion gadgets accepted every child leaf proof (i) satisfies");
    w!("    `RPrivateBatch`, (ii) conserves value, (iii) settles distinct spends and (iv) attests");
    w!("    every child's `Rleaf` — through `leaf_proof_sound`, the only axiom. -/");
    w!("theorem end_to_end_wired (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hp : Poseidon2Rows perm ({circuit} p) a)");
    w!("    (hacc : ∀ pub ∈ leaves a, LeafProofAccepted (spongeRO perm) pub) :");
    w!("    RPrivateBatch (spongeRO perm) (leaves a) (us a) (out a)");
    w!("      ∧ outputExitTotal (out a) = maskedOutputTotal (leaves a)");
    w!("      ∧ (outputExitTotal (out a) = rawOutputTotal (realLeaves (leaves a))");
    w!("          ∧ (realNullifiers (leaves a)).Nodup)");
    w!("      ∧ ∀ pub ∈ leaves a, ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w := by");
    w!("  have hleaf := fun pub hq => leaf_proof_sound (spongeRO perm) pub (hacc pub hq)");
    w!("  have hR := sound perm hpg a h hp fun q hq =>");
    w!("    let ⟨_, hw⟩ := hleaf q hq");
    w!("    Rleaf_ranges hw");
    w!("  exact ⟨hR, RPrivateBatch_value_conservation hR, RPrivateBatch_settles_distinct_spends hR, hleaf⟩");
    w!("");
    w!("end Plonky2Bridge.Wrapper{n}");
    o
}

/// The final nullifier digests after the switch network, position by position.
fn network_outputs(shape: &Shape) -> Vec<[Target; 4]> {
    let n = shape.n;
    let mut current: Vec<[Target; 4]> = shape
        .leaves
        .iter()
        .map(|leaf| {
            [
                shape.out(leaf.null_sel[0]),
                shape.out(leaf.null_sel[1]),
                shape.out(leaf.null_sel[2]),
                shape.out(leaf.null_sel[3]),
            ]
        })
        .collect();
    let positions = switch_positions(n);
    for (round, pos) in shape.rounds.iter().zip(positions.iter()) {
        for (sw, &i) in round.iter().zip(pos.iter()) {
            let lo = [
                shape.out(sw.sel[0][0]),
                shape.out(sw.sel[1][0]),
                shape.out(sw.sel[2][0]),
                shape.out(sw.sel[3][0]),
            ];
            let hi = [
                shape.out(sw.sel[0][1]),
                shape.out(sw.sel[1][1]),
                shape.out(sw.sel[2][1]),
                shape.out(sw.sel[3][1]),
            ];
            current[i] = lo;
            current[i + 1] = hi;
        }
    }
    current
}

/// The normal form `simp only [scanRefL, …, scanStep, bor_zero_left, bnot_zero, band_one_right]`
/// leaves `scanRefL [r₀, …, rₙ₋₁] f` in: with `R_i = bnot D_i`, `F_0 = R_0`,
/// `F_i = bor F_{i-1} R_i`, `T_0 = R_0`, `T_i = band R_i (bnot F_{i-1})`, the reference is
/// `bselect T_{n-1} x_{n-1} (… (bselect T_0 x_0 0))`.
fn scan_normal_form(
    n: usize,
    dummy: &dyn Fn(usize) -> String,
    x: &dyn Fn(usize) -> String,
) -> String {
    let r = |i: usize| format!("(bnot {})", dummy(i));
    let mut found = r(0);
    let mut acc = format!("bselect {} {} 0", r(0), x(0));
    for i in 1..n {
        let take = format!("(band {} (bnot {found}))", r(i));
        acc = format!("bselect {take} {} ({acc})", x(i));
        found = format!("(bor {found} {})", r(i));
    }
    acc
}

/// Build `formal/Plonky2Bridge/Generated/Wrapper{n}.lean` from the recorded `n` private-batch
/// wrapper trace.
pub fn generate_private_wrapper_bridge_lean(n: usize) -> Result<String, String> {
    let circuit = format!("private_batch_wrapper_n{n}");
    let t = load(&traces_dir().join(format!("{circuit}.json")))?;
    if t.circuit != circuit {
        return Err(format!("unexpected trace circuit {:?}", t.circuit));
    }
    let shape = Shape::read(&t)?;
    if shape.n != n {
        return Err(format!("trace {circuit} has {} leaves", shape.n));
    }
    Ok(render(&shape))
}
