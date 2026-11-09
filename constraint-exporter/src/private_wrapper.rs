//! Generates `formal/Plonky2Bridge/Generated/Wrapper{N}.lean` — the composition of a recorded
//! `n = N` private-batch wrapper trace into `RPrivateBatch` (PLAN.md Step 8e).
//!
//! `build_private_batch_constraints` makes its gadget calls in a fixed order determined by
//! the shape `(n leaves, m exit slots)`: the per-leaf dummy checks, the first-real block
//! scan, the per-leaf consistency checks, the masked input total, the exit-amount range
//! checks with the output total, the fee comparator on the witnessed `volume_fee_bps`, the
//! pairwise nullifier-uniqueness constraints, and the dummy-nullifier hashes with the
//! nullifier selects. [`Shape::read`] walks the recorded calls with a cursor and checks each
//! one against the role it must play (which named targets it reads, which constants, how
//! the outputs chain), so a change in the wrapper's structure is a generator error rather
//! than a proof that quietly talks about different wires — a free witness that lost its
//! range check, say, leaves `FeeCheck` or `hslots32` undischargeable. [`render`] then emits
//! the decode definitions (`row`, `out`, `slots`, …), the concrete-`N` unfolding of
//! `scanRefL`, and a `sound` proof that discharges every hypothesis of
//! `Plonky2Bridge.private_batch_val_rows` by rewriting with the exporter-generated per-call
//! decode lemmas `privateBatchWrapper{N}_f{k}`.

use core::fmt::Write as _;

use plonky2::field::types::PrimeField64;
use plonky2::iop::target::Target;

use crate::circuit::lean_target;
use crate::gadget::{Call, Fact};
use crate::trace::{check_verifiers, load, traces_dir, LoadedTrace};

/// The leaf public-input layout (`wormhole/aggregator/src/private_batch/circuit/constants.rs`).
mod layout {
    pub const ASSET_ID: usize = 0;
    pub const NULLIFIER: usize = 1;
    pub const BLOCK_HASH: usize = 5;
    pub const BLOCK_NUMBER: usize = 9;
    pub const INPUT_AMOUNT: usize = 10;
    pub const LEAF_PI_LEN: usize = 11;
    pub const PRE_IMAGE_LEN: usize = 4;
    pub const HEADER_LEN: usize = 8;
    pub const EXIT_SLOT_LEN: usize = 5;
    pub const FEE_DENOMINATOR: u64 = 10_000;
    pub const FEE_BITS: usize = 14;
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
    /// The five reference `select`s: block hash limbs, block number.
    sel_block: [usize; 4],
    sel_number: usize,
    /// Consistency: `bytes_digest_eq`, `or`, `connect` for the block hash; `connect` for the
    /// asset id (absent for leaf 0, the reference itself).
    block: DigestEq,
    block_or: usize,
    block_pin: usize,
    asset_pin: Option<usize>,
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

/// One exit slot: the witnessed amount and account targets, the amount's `range_check 32`
/// and its `add` into the output total (absent for the first, folded against `zero`).
#[derive(Debug, Clone, Copy)]
struct ExitSlot {
    amount: Target,
    account: [Target; 4],
    range: usize,
    add: Option<usize>,
}

/// The recorded wrapper, indexed by role.
#[derive(Debug, Clone)]
pub struct Shape {
    pub n: usize,
    /// The number of exit slots `m`.
    pub m: usize,
    /// The `8 + 5m + 4n` registered public inputs.
    public_inputs: Vec<Target>,
    leaves: Vec<Leaf>,
    slots: Vec<ExitSlot>,
    /// `range_check 14` of the fee, then `sub`, `range_check 14`, `mul`, `mul`, `sub`,
    /// `range_check 52` of the fee comparator.
    fee: [usize; 7],
    pairs: Vec<Pair>,
    one: Target,
    zero: Target,
    total: Target,
    ten_thousand: Target,
    /// The witnessed `volume_fee_bps`.
    fee_witness: Target,
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
            return Err("leaf_pis_i is not 11 targets".into());
        }
        check_verifiers(t, "leaf_circuit", &pis)?;
        let pres: Vec<Vec<Target>> = (0..n)
            .map(|i| named(&format!("dummy_pre_image_{i}")).cloned())
            .collect::<Result<_, _>>()?;
        if pres.iter().any(|p| p.len() != layout::PRE_IMAGE_LEN) {
            return Err("dummy_pre_image_i is not 4 targets".into());
        }
        let fee_witness = match named("volume_fee_bps")?.as_slice() {
            [f] => *f,
            _ => return Err("volume_fee_bps is not one target".into()),
        };
        let m =
            t.ex.named
                .iter()
                .filter(|(name, _)| name.starts_with("exit_amount_"))
                .count();
        if m < 2 {
            return Err(format!("expected at least two exit_amount_i, found {m}"));
        }
        let slot_targets: Vec<(Target, [Target; 4])> = (0..m)
            .map(|k| {
                let amount = match named(&format!("exit_amount_{k}"))?.as_slice() {
                    [x] => *x,
                    _ => return Err(format!("exit_amount_{k} is not one target")),
                };
                let account: [Target; 4] = named(&format!("exit_account_{k}"))?
                    .clone()
                    .try_into()
                    .map_err(|_| format!("exit_account_{k} is not 4 targets"))?;
                Ok((amount, account))
            })
            .collect::<Result<_, String>>()?;

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
        let total_value = m as u64;
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
        for (i, pis_i) in pis.iter().enumerate() {
            let (dummy, d) = c.digest_eq(
                limbs(pis_i, layout::BLOCK_HASH),
                [zero; 4],
                &format!("dummy check of leaf {i}"),
            )?;
            dummies.push(d);
            leaves.push(Leaf {
                pis: pis_i.clone(),
                dummy,
                is_real: 0,
                not_found: None,
                take: None,
                found: None,
                sel_block: [0; 4],
                sel_number: 0,
                block: DigestEq {
                    eq: [0; 4],
                    and: [0; 3],
                },
                block_or: 0,
                block_pin: 0,
                asset_pin: None,
                mask_input: 0,
                add_input: None,
                hash_inner: 0,
                hash_outer: 0,
                null_sel: [0; 4],
            });
        }

        // B. First-real block scan.
        let mut found_prev = zero;
        let mut ref_prev: Vec<Target> = vec![zero; 5];
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
            ];
            let mut ref_now = Vec::with_capacity(5);
            let mut ks = [0usize; 5];
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
            found_prev = found;
            ref_prev = ref_now;
        }
        let block_ref: [Target; 4] = [ref_prev[0], ref_prev[1], ref_prev[2], ref_prev[3]];
        let number_ref = ref_prev[4];
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
            let leaf = &mut leaves[i];
            leaf.block = block;
            leaf.block_or = k_or;
            leaf.block_pin = k_pin;
            leaf.asset_pin = (i > 0).then_some(k_asset);
        }

        // D. Input total.
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

        // E. Exit-amount range checks and the output total.
        let mut acc = zero;
        let mut slots = Vec::with_capacity(m);
        for (k, &(amount, account)) in slot_targets.iter().enumerate() {
            let range = c.range_check(amount, layout::AMOUNT_BITS, &format!("exit amount {k}"))?;
            let (k_add, sum) = c.add(acc, amount, &format!("output total after slot {k}"))?;
            let add = if k == 0 {
                if sum != amount {
                    return Err(
                        "slot 0: add(zero, exit_amount_0) should fold to exit_amount_0".into(),
                    );
                }
                None
            } else {
                Some(k_add)
            };
            slots.push(ExitSlot {
                amount,
                account,
                range,
                add,
            });
            acc = sum;
        }
        let total_out = acc;

        // F. Fee comparator. The fee is a free witness: its own range check comes first.
        let k_fee_rc = c.range_check(fee_witness, layout::FEE_BITS, "volume_fee_bps")?;
        let (k_comp, comp) = c.sub(ten_thousand, fee_witness, "fee complement")?;
        let k_comp_rc = c.range_check(comp, layout::FEE_BITS, "fee complement")?;
        let (k_rhs, rhs) = c.mul(total_out, ten_thousand, "total_out * 10000")?;
        let (k_lhs, lhs) = c.mul(total_in, comp, "total_in * complement")?;
        let (k_diff, diff) = c.sub(lhs, rhs, "fee difference")?;
        let k_diff_rc = c.range_check(diff, layout::FEE_DIFF_BITS, "fee difference")?;
        let fee = [k_fee_rc, k_comp, k_comp_rc, k_rhs, k_lhs, k_diff, k_diff_rc];

        // G. Nullifier uniqueness.
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

        // H. Dummy nullifiers and the nullifier selects.
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

        if c.k != t.calls.len() {
            return Err(format!(
                "{} trailing gadget calls after the nullifier selects",
                t.calls.len() - c.k
            ));
        }

        // Public inputs.
        let mut public_inputs: Vec<Target> = vec![total, asset_ref, fee_witness];
        public_inputs.extend(block_ref);
        public_inputs.push(number_ref);
        for slot in &slots {
            public_inputs.push(slot.amount);
            public_inputs.extend(slot.account);
        }
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
            m,
            public_inputs: t.ex.public_inputs.clone(),
            leaves,
            slots,
            fee,
            pairs,
            one,
            zero,
            total,
            ten_thousand,
            fee_witness,
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
    let m = shape.m;
    let pi_total = shape.public_inputs.len();
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
    w!("  Step 8e — `Generated/Wrapper{n}.lean` proves, from `Satisfies` on the wiring the real");
    w!(
        "  `CircuitBuilder` emitted plus `Poseidon2Rows perm`, the meaning of each of the {}",
        shape.calls.len()
    );
    w!("  gadget calls `build_private_batch_constraints` made over {n} leaves and {m} exit slots.");
    w!("  This module reads the spec objects off the named targets and public inputs (`row`,");
    w!("  `leaves`, `us`, `slots`, `out`) and discharges every hypothesis of");
    w!("  `private_batch_val_rows` (`Plonky2Bridge/PrivateBatch.lean`), so `private_batch_end_to_end`");
    w!("  is restated on the wiring (`end_to_end_wired`), with the children's `Rleaf` recovered from");
    w!("  their accepted proofs through the one trusted axiom `proof_sound` and the leaf bridge");
    w!("  (Step 10). `accepted_sound` is the wrapper's own `proof_sound` consequence, for the");
    w!("  public-batch wrapper above it.");
    w!("-/");
    w!("import Plonky2Bridge.PrivateBatch");
    w!("import Plonky2Bridge.Generated.Leaf");
    w!("import Plonky2Spec.Generated.PrivateBatchWrapper{n}");
    w!("");
    w!("namespace Plonky2Bridge.Wrapper{n}");
    w!("");
    w!("open Plonky2Spec (IsBool bselect band bnot bor scanStep rangeCheck feeDen FeeCheck digestEq Digest4)");
    w!("open Plonky2Spec.Wiring");
    w!("open Plonky2Spec.Generated");
    w!("open Plonky2Spec.Poseidon2 (St)");
    w!("open Plonky2Spec.Sponge (spongeHash)");
    w!("open WormholeSpec (Digest Felt LeafPublic PrivateBatchOutput ExitSlot inRange RPrivateBatch");
    w!("  maskedInputTotal rawInputTotal realLeaves realNullifiers outputExitTotal");
    w!("  RPrivateBatch_settlement_le_input RPrivateBatch_settles_distinct_spends LeafWitness Rleaf)");
    w!("");
    w!("variable {{p : ℕ}} [Fact p.Prime]");
    w!("");
    w!("set_option linter.unusedSimpArgs false");
    w!("");
    w!("/-! ### Reading the spec objects off the wiring -/");
    w!("");
    w!(
        "/-- The {} leaf public inputs of leaf `i` (`leaf_pis_i`). -/",
        layout::LEAF_PI_LEN
    );
    w!(
        "def leafPis : Fin {n} → Fin {} → Target",
        layout::LEAF_PI_LEN
    );
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
            "    {{ assetId := a (leafPis {i} {}), nullifier := fun j => a (leafPis {i} ⟨{} + j, by omega⟩),",
            layout::ASSET_ID,
            layout::NULLIFIER
        );
        w!(
            "      blockHash := fun j => a (leafPis {i} ⟨{} + j, by omega⟩),",
            layout::BLOCK_HASH
        );
        w!(
            "      blockNumber := a (leafPis {i} {}), inputAmount := a (leafPis {i} {}),",
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
    let slot_list: Vec<String> = shape
        .slots
        .iter()
        .map(|slot| format!("({}, {})", a(slot.amount), digest_fun(&slot.account)))
        .collect();
    w!("/-- The {m} witnessed exit slots `(exit_amount_k, exit_account_k)`. -/");
    w!("def slots (a : Assignment p) : List (SlotF p) :=");
    w!("  [{}]", slot_list.join(",\n   "));
    w!("");
    w!("/-- Public input `k` of the wrapper, decoded through `.val`. -/");
    w!("def pv (a : Assignment p) (k : ℕ) : Felt :=");
    w!("  (a (({circuit} p).publicInputs.getD k (.virt 0))).val");
    w!("");
    let header = layout::HEADER_LEN;
    let slot_pvs: Vec<String> = (0..m)
        .map(|k| {
            let b = header + layout::EXIT_SLOT_LEN * k;
            format!(
                "⟨pv a {b}, ⟨pv a {}, pv a {}, pv a {}, pv a {}⟩⟩",
                b + 1,
                b + 2,
                b + 3,
                b + 4
            )
        })
        .collect();
    let null_base = header + layout::EXIT_SLOT_LEN * m;
    let null_pvs: Vec<String> = (0..n)
        .map(|i| {
            let b = null_base + 4 * i;
            format!("⟨pv a {b}, pv a {}, pv a {}, pv a {}⟩", b + 1, b + 2, b + 3)
        })
        .collect();
    w!("/-- The aggregated output at the `aggregated_output` layout: header `[num_exit_slots,");
    w!("    asset_id, volume_fee_bps, block_hash(4), block_number]`, `M = {m}` exit slots");
    w!("    `[amount, account(4)]`, `N = {n}` nullifiers. -/");
    w!("def out (a : Assignment p) : PrivateBatchOutput :=");
    w!("  {{ numExitSlots := pv a 0, assetId := pv a 1, volumeFeeBps := pv a 2,");
    w!("    blockHash := ⟨pv a 3, pv a 4, pv a 5, pv a 6⟩, blockNumber := pv a 7,");
    w!("    exitSlots := [{}],", slot_pvs.join(",\n      "));
    w!("    nullifiers := [{}] }}", null_pvs.join(",\n      "));
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
    let fee = a(shape.fee_witness);
    let number_ref = a(shape.number_ref);
    let asset_ref = a(leaves[0].pis[layout::ASSET_ID]);
    let total_in = a(shape.total_in);
    let total_out = a(shape.total_out);

    w!("set_option maxHeartbeats 4000000 in");
    w!("/-- **The recorded private-batch wrapper satisfies `RPrivateBatch`.** Every satisfying");
    w!("    assignment whose Poseidon2 rows compute `perm`, with the leaves' 32-bit input amounts");
    w!("    (which the leaf proofs certify), decodes to an `RPrivateBatch` instance on the {n}");
    w!("    children, their preimages and the aggregated output. -/");
    w!("theorem sound (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hp : Poseidon2Rows perm ({circuit} p) a)");
    w!("    (h32 : ∀ q ∈ leaves a, inRange 32 q.inputAmount) :");
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
    }
    for (i, leaf) in leaves.iter().enumerate() {
        proj(format!("mi{i}"), leaf.mask_input);
        if let Some(k) = leaf.add_input {
            proj(format!("ai{i}"), k);
        }
    }
    for (k, slot) in shape.slots.iter().enumerate() {
        proj(format!("rc{k}"), slot.range);
        if let Some(c) = slot.add {
            proj(format!("ao{k}"), c);
        }
    }
    proj("ffr".into(), shape.fee[0]);
    proj("fc".into(), shape.fee[1]);
    proj("fcr".into(), shape.fee[2]);
    proj("fr".into(), shape.fee[3]);
    proj("fl".into(), shape.fee[4]);
    proj("fd".into(), shape.fee[5]);
    proj("fdr".into(), shape.fee[6]);
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
    w!("  have hout : {total_out} = ((slots a).map Prod.fst).sum := by");
    w!("    simp only [slots, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]");
    let rws: Vec<String> = (1..m).rev().map(|k| format!("ao{k}")).collect();
    w!("    rw [{}]", rws.join(", "));
    if m > 2 {
        w!("    ring");
    }

    // Exit-amount range checks.
    w!("  have hslots32 : ∀ s ∈ slots a, rangeCheck s.1 32 := by");
    w!("    intro s hs");
    w!("    simp only [slots, List.mem_cons, List.not_mem_nil, or_false] at hs");
    let slot_rfls = vec!["rfl"; m].join(" | ");
    w!("    rcases hs with {slot_rfls}");
    for k in 0..m {
        w!("    · exact rc{k}");
    }

    // Fee.
    w!("  have hfc : FeeCheck ({fee}) ({total_in}) ({total_out}) := by");
    w!("    refine ⟨ffr, ?_, ?_⟩");
    w!("    · show rangeCheck (feeDen - {fee}) 14");
    w!("      unfold feeDen; rw [Nat.cast_ofNat, ← kten, ← fc]; exact fcr");
    w!("    · show rangeCheck ({total_in} * (feeDen - {fee}) - {total_out} * feeDen) 52");
    w!("      unfold feeDen; rw [Nat.cast_ofNat, ← kten, ← fc, ← fl, ← fr, ← fd]; exact fdr");

    // Nullifiers.
    let null_lits: Vec<String> = leaves
        .iter()
        .map(|leaf| {
            format!(
                "⟨({}).val, ({}).val, ({}).val, ({}).val⟩",
                a(shape.out(leaf.null_sel[0])),
                a(shape.out(leaf.null_sel[1])),
                a(shape.out(leaf.null_sel[2])),
                a(shape.out(leaf.null_sel[3]))
            )
        })
        .collect();
    w!("  have hnulls : (out a).nullifiers = ((rows a).map fun r => r.slot.sel).map valDigest := by");
    w!(
        "    show ([{}] : List Digest) = _",
        null_lits.join(",\n      ")
    );
    let mut rws: Vec<String> = Vec::new();
    for i in 0..n {
        for l in 0..4 {
            rws.push(format!("s{i}_{l}"));
        }
    }
    rws.extend(hds.iter().cloned());
    w!("    rw [{}]", rws.join(", "));
    w!("    rfl");

    // Uniqueness.
    let digest_eq_witness = |shape: &Shape, prefix: &str, de: &DigestEq| -> String {
        let (eqs, invs): (Vec<Target>, Vec<Target>) =
            de.eq.iter().map(|&k| shape.is_equal_witness(k)).unzip();
        format!(
            "⟨{}, {}, fun j => by fin_cases j <;> assumption, by rw [{prefix}m, {prefix}a, {prefix}b]; rfl⟩",
            vec4(&eqs),
            vec4(&invs)
        )
    };
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
    w!("  have hR : RPrivateBatch (spongeRO perm) ((rows a).map LeafRow.leaf) ((rows a).map LeafRow.u) (out a) := by");
    w!("    refine private_batch_val_rows perm hpg (rows a) (out := out a)");
    w!("      (fee := {fee}) (totalIn := {total_in}) (totalOut := {total_out})");
    w!("      (assetRef := {asset_ref}) (numRef := {number_ref}) (blockRef := {block_ref})");
    w!("      (slots := slots a) ?_ ?_ (by show {n} ≤ 64; decide) h32 hblockRef hnumRef ?_ ?_");
    w!("      rfl rfl rfl rfl ?_ hin hslots32 (by show {m} ≤ 64; decide) hout hfc rfl hnulls hcol");
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
    w!("      show ({}).val = {m}", a(shape.total));
    w!("      rw [ktot]");
    w!("      have hc : ({m} : ℕ) < p := lt_of_lt_of_le (by decide) hpg");
    w!("      rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]");
    w!("  rwa [rows_leaves, rows_us] at hR");
    w!("");
    w!("/-! ### The wrapper as a recursion tree (PLAN.md Step 10) -/");
    w!("");
    w!("/-- The wrapper with its recursion gadgets: `verify_proof` under the leaf verifier key on");
    w!("    each `leaf_pis_i`. -/");
    w!("def tree : Recursive p :=");
    w!("  .node \"private_batch_wrapper_n{n}\" ({circuit} p)");
    let children: Vec<String> = (0..n)
        .map(|i| format!("(LeafCircuit.tree, List.ofFn (leafPis {i}))"))
        .collect();
    w!("    [{}]", children.join(",\n     "));
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("set_option maxRecDepth 8192 in");
    w!("/-- The tree's gadgets are the recorded ones. -/");
    w!("theorem tree_verifiers : (tree (p := p)).verifiers = {circuit}.verifiers := rfl");
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("/-- Child `i`'s public inputs, as the wrapper reads them, are the leaf's `pubOf`. -/");
    w!("theorem row_leaf (a : Assignment p) (i : Fin {n}) :");
    w!("    (row a i).leaf = LeafCircuit.pubOf (a ∘ leafPis i) := by");
    w!("  fin_cases i <;> rfl");
    w!("");
    w!("/-- **The capstone on the recorded wiring.** `private_batch_end_to_end` with its decode");
    w!("    hypotheses discharged by `sound`: a satisfying assignment of the `n = {n}` private-batch");
    w!("    wrapper whose recursion gadgets accepted every child leaf proof (i) satisfies");
    w!("    `RPrivateBatch`, (ii) settles at most the masked input total, (iii) settles at most the");
    w!("    real inputs with distinct spends and (iv) attests every child's `Rleaf` — through");
    w!("    `proof_sound`, the only axiom, and `leaf_wired`. -/");
    w!("theorem end_to_end_wired (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hp : Poseidon2Rows perm ({circuit} p) a)");
    w!("    (hacc : ∀ i : Fin {n}, ProofAccepted perm LeafCircuit.tree ((List.ofFn (leafPis i)).map a)) :");
    w!("    RPrivateBatch (spongeRO perm) (leaves a) (us a) (out a)");
    w!("      ∧ outputExitTotal (out a) ≤ maskedInputTotal (leaves a)");
    w!("      ∧ (outputExitTotal (out a) ≤ rawInputTotal (realLeaves (leaves a))");
    w!("          ∧ (realNullifiers (leaves a)).Nodup)");
    w!("      ∧ ∀ pub ∈ leaves a, ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w := by");
    w!("  have hleaf : ∀ pub ∈ leaves a, ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w := by");
    w!("    intro pub hq");
    w!("    simp only [leaves, List.mem_cons, List.mem_nil_iff, or_false] at hq");
    let alts: Vec<&str> = (0..n).map(|_| "rfl").collect();
    w!("    rcases hq with {}", alts.join(" | "));
    for i in 0..n {
        w!("    · rw [row_leaf]; exact LeafCircuit.accepted_sound perm hpg (hacc {i})");
    }
    w!("  have hR := sound perm hpg a h hp fun q hq =>");
    w!("    let ⟨_, hw⟩ := hleaf q hq");
    w!("    Rleaf_ranges hw");
    w!("  exact ⟨hR, RPrivateBatch_settlement_le_input hR, RPrivateBatch_settles_distinct_spends hR, hleaf⟩");
    w!("");
    w!("/-- The {pi_total} public inputs in registration order. -/");
    w!("def pi : Fin {pi_total} → Target :=");
    let pi_items: Vec<String> = shape
        .public_inputs
        .iter()
        .map(|&t| lean_target(t))
        .collect();
    w!("  ![{}]", pi_items.join(", "));
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("set_option maxRecDepth 8192 in");
    w!("theorem publicInputs_eq : ({circuit} p).publicInputs = List.ofFn pi := rfl");
    w!("");
    w!("/-- `out` as a function of the public-input values alone. -/");
    w!("def outOf (v : Fin {pi_total} → ZMod p) : PrivateBatchOutput :=");
    let vv = |k: usize| format!("(v {k}).val");
    let slot_vs: Vec<String> = (0..m)
        .map(|k| {
            let b = header + layout::EXIT_SLOT_LEN * k;
            format!(
                "⟨{}, ⟨{}, {}, {}, {}⟩⟩",
                vv(b),
                vv(b + 1),
                vv(b + 2),
                vv(b + 3),
                vv(b + 4)
            )
        })
        .collect();
    let null_vs: Vec<String> = (0..n)
        .map(|i| {
            let b = null_base + 4 * i;
            format!("⟨{}, {}, {}, {}⟩", vv(b), vv(b + 1), vv(b + 2), vv(b + 3))
        })
        .collect();
    w!(
        "  {{ numExitSlots := {}, assetId := {}, volumeFeeBps := {},",
        vv(0),
        vv(1),
        vv(2)
    );
    w!(
        "    blockHash := ⟨{}, {}, {}, {}⟩, blockNumber := {},",
        vv(3),
        vv(4),
        vv(5),
        vv(6),
        vv(7)
    );
    w!("    exitSlots := [{}],", slot_vs.join(",\n      "));
    w!("    nullifiers := [{}] }}", null_vs.join(",\n      "));
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("set_option maxRecDepth 8192 in");
    w!("theorem out_eq (a : Assignment p) : out a = outOf (a ∘ pi) := rfl");
    w!("");
    w!("/-- **An accepted private-batch proof attests `RPrivateBatch`.** Through `proof_sound`: the");
    w!("    accepted public inputs are those of an assignment satisfying the wrapper's exported");
    w!("    wiring whose own recursion gadgets accepted every leaf proof, hence (`end_to_end_wired`)");
    w!("    an `RPrivateBatch` instance. This is what `WormholeSpec.private_batch_proof_sound` used");
    w!("    to assume. -/");
    w!("theorem accepted_sound (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    {{ts : Fin {pi_total} → Target}} {{a : Assignment p}}");
    w!("    (h : ProofAccepted perm tree ((List.ofFn ts).map a)) :");
    w!("    ∃ leaves us, RPrivateBatch (spongeRO perm) leaves us (outOf (a ∘ ts)) := by");
    w!("  obtain ⟨a', h', hp', hpis, hch⟩ := proof_sound perm _ _ _ _ h");
    w!("  rw [publicInputs_eq] at hpis");
    w!("  have hacc : ∀ i : Fin {n}, ProofAccepted perm LeafCircuit.tree ((List.ofFn (leafPis i)).map a') := by");
    w!("    intro i");
    w!("    fin_cases i");
    for i in 0..n {
        w!("    · exact hch (LeafCircuit.tree, List.ofFn (leafPis {i})) (by simp)");
    }
    w!("  have hR := (end_to_end_wired perm hpg a' h' hp' hacc).1");
    w!("  rw [out_eq, pis_eq hpis] at hR");
    w!("  exact ⟨_, _, hR⟩");
    w!("");
    w!("end Plonky2Bridge.Wrapper{n}");
    o
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
