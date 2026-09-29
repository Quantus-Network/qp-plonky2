//! Generates `formal/Plonky2Bridge/Generated/PublicWrapper{N}.lean` — the composition of a
//! recorded `n_inner = N` public-batch wrapper trace into `RPublicBatch` (PLAN.md Step 8e).
//!
//! `build_public_batch_constraints` makes its gadget calls in a fixed order determined by
//! `(n_inner, private_batch_num_leaves)`: the per-inner dummy checks, the first-real prefix
//! scan, the per-inner consistency checks, then the masked slot and nullifier forwards.
//! [`Shape::read`] indexes the recorded calls positionally and checks each one against the
//! role it must play (which named targets it reads, which constants, how the outputs chain),
//! so a change in the wrapper's structure is a generator error rather than a proof that
//! quietly talks about different wires. [`render`] then emits the decode definitions
//! (`row`, `inner`, `out`, …), the concrete-`N` unfolding of `scanRef`, and a `sound` proof
//! that discharges every hypothesis of `Plonky2Bridge.public_batch_val` by rewriting with
//! the facts of the exporter-generated `publicBatchWrapper{N}_decode`.

use core::fmt::Write as _;

use plonky2::field::types::PrimeField64;
use plonky2::iop::target::Target;

use crate::circuit::lean_target;
use crate::gadget::{Call, Fact, FACT_GROUP};
use crate::trace::{load, traces_dir, LoadedTrace};

/// The private-batch `aggregated_output` public-input layout each inner proof exposes
/// (`wormhole/aggregator/src/private_batch/circuit/constants.rs`).
mod layout {
    pub const NUM_EXIT_SLOTS: usize = 0;
    pub const ASSET_ID: usize = 1;
    pub const VOLUME_FEE_BPS: usize = 2;
    pub const BLOCK_HASH: usize = 3;
    pub const BLOCK_NUMBER: usize = 7;
    pub const HEADER_LEN: usize = 8;
    pub const EXIT_SLOT_LEN: usize = 5;
    pub const LEAF_PI_LEN: usize = 22;
}

/// One inner proof's calls, by role (indices into the trace's calls).
#[derive(Debug, Clone)]
struct Inner {
    pis: Vec<Target>,
    /// `bytes_digest_eq(block_hash, 0)`: four `is_equal` then the `and` tree.
    dummy_eq: [usize; 4],
    dummy_and: [usize; 3],
    /// `not(is_dummy)`.
    is_real: usize,
    /// `not(found_real)`, `and(is_real, not_found)`: absent for inner 0, where the builder
    /// folds them against the constant `false`.
    not_found: Option<usize>,
    take: Option<usize>,
    /// `or(found_real, is_real)`, when a later inner reads it (absent for inner 0, folded,
    /// and for the last inner).
    found: Option<usize>,
    /// The seven reference `select`s: block hash limbs, block number, asset id, fee.
    sel_block: [usize; 4],
    sel_number: usize,
    sel_asset: usize,
    sel_fee: usize,
    /// Consistency: `is_equal`, `or`, `connect` for asset and fee; `bytes_digest_eq`
    /// (four `is_equal`, three `and`), `or`, `connect` for the block hash.
    asset_eq: usize,
    asset_or: usize,
    asset_pin: usize,
    fee_eq: usize,
    fee_or: usize,
    fee_pin: usize,
    block_eq: [usize; 4],
    block_and: [usize; 3],
    block_or: usize,
    block_pin: usize,
    /// Masked forwards, one `select` per limb.
    slot_fwd: Vec<usize>,
    null_fwd: Vec<usize>,
}

/// The recorded wrapper, indexed by role.
#[derive(Debug, Clone)]
pub struct Shape {
    pub n: usize,
    pub leaves: usize,
    pub pi_len: usize,
    pub slots_per_inner: usize,
    pub nulls_per_inner: usize,
    inners: Vec<Inner>,
    one: Target,
    zero: Target,
    total: Target,
    /// `(target, value)` in the order of the export's `constants`.
    constants: Vec<(Target, u64)>,
    calls: Vec<Call>,
}

fn out_of(f: &Fact) -> Option<Target> {
    match *f {
        Fact::Select { out, .. }
        | Fact::Not { out, .. }
        | Fact::And { out, .. }
        | Fact::Or { out, .. } => Some(out),
        _ => None,
    }
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
                .filter(|(name, _)| name.starts_with("inner_pis_"))
                .count();
        if n < 2 {
            return Err(format!("expected at least two inner_pis_i, found {n}"));
        }
        let pis: Vec<Vec<Target>> = (0..n)
            .map(|i| named(&format!("inner_pis_{i}")).cloned())
            .collect::<Result<_, _>>()?;
        let pi_len = pis[0].len();
        if pis.iter().any(|p| p.len() != pi_len) {
            return Err("inner_pis_i differ in length".into());
        }
        if pi_len < layout::HEADER_LEN || (pi_len - layout::HEADER_LEN) % layout::LEAF_PI_LEN != 0 {
            return Err(format!("inner PI length {pi_len} is not 22·leaves + 8"));
        }
        let leaves = (pi_len - layout::HEADER_LEN) / layout::LEAF_PI_LEN;
        let slots_per_inner = 2 * leaves;
        let nulls_per_inner = leaves;
        let aggregator_address: [Target; 4] = named("aggregator_address")?
            .clone()
            .try_into()
            .map_err(|_| "aggregator_address is not 4 targets".to_string())?;

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
        let total_value = (n * slots_per_inner) as u64;
        let total = const_target(total_value)?;
        if constants.len() != 3 {
            return Err(format!(
                "expected constants {{1, 0, {total_value}}}, found {constants:?}"
            ));
        }

        let calls = &t.calls;
        let expected_calls = 33 * n + 5 * slots_per_inner * n + 4 * nulls_per_inner * n;
        if calls.len() != expected_calls {
            return Err(format!(
                "expected {expected_calls} gadget calls for n_inner={n}, leaves={leaves}; trace has {}",
                calls.len()
            ));
        }
        let fact = |k: usize| -> &Fact { &calls[k].fact };
        let check = |ok: bool, what: String| -> Result<(), String> {
            if ok {
                Ok(())
            } else {
                Err(what)
            }
        };

        let mut inners = Vec::with_capacity(n);
        // Dummy checks.
        for (i, pis_i) in pis.iter().enumerate() {
            let base = 7 * i;
            let mut eqs = [Target::VirtualTarget { index: 0 }; 4];
            for j in 0..4 {
                match *fact(base + j) {
                    Fact::IsEqual { x, y, equal, .. }
                        if x == pis_i[layout::BLOCK_HASH + j] && y == zero =>
                    {
                        eqs[j] = equal;
                    }
                    ref f => {
                        return Err(format!(
                            "call {}: expected dummy is_equal of inner {i} limb {j}, got {f:?}",
                            base + j
                        ))
                    }
                }
            }
            let and_a = match *fact(base + 4) {
                Fact::And { b1, b2, out } if b1 == eqs[0] && b2 == eqs[1] => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected and(eq0, eq1) of inner {i}, got {f:?}",
                        base + 4
                    ))
                }
            };
            let and_b = match *fact(base + 5) {
                Fact::And { b1, b2, out } if b1 == eqs[2] && b2 == eqs[3] => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected and(eq2, eq3) of inner {i}, got {f:?}",
                        base + 5
                    ))
                }
            };
            match *fact(base + 6) {
                Fact::And { b1, b2, .. } if b1 == and_a && b2 == and_b => {}
                ref f => {
                    return Err(format!(
                        "call {}: expected the dummy and-root of inner {i}, got {f:?}",
                        base + 6
                    ))
                }
            }
            inners.push(Inner {
                pis: pis_i.clone(),
                dummy_eq: [base, base + 1, base + 2, base + 3],
                dummy_and: [base + 4, base + 5, base + 6],
                is_real: 0,
                not_found: None,
                take: None,
                found: None,
                sel_block: [0; 4],
                sel_number: 0,
                sel_asset: 0,
                sel_fee: 0,
                asset_eq: 0,
                asset_or: 0,
                asset_pin: 0,
                fee_eq: 0,
                fee_or: 0,
                fee_pin: 0,
                block_eq: [0; 4],
                block_and: [0; 3],
                block_or: 0,
                block_pin: 0,
                slot_fwd: Vec::new(),
                null_fwd: Vec::new(),
            });
        }
        let dummies: Vec<Target> = inners
            .iter()
            .map(|inner| out_of(fact(inner.dummy_and[2])).unwrap())
            .collect();
        let dummy = |i: usize| dummies[i];

        // First-real scan.
        let mut found_prev = zero;
        let mut ref_prev: Vec<Target> = vec![zero; 7];
        for i in 0..n {
            let base = 7 * n + 11 * i;
            let is_real = match *fact(base) {
                Fact::Not { b, out } if b == dummy(i) => out,
                ref f => {
                    return Err(format!(
                        "call {base}: expected not(is_dummy_{i}), got {f:?}"
                    ))
                }
            };
            let not_found = match *fact(base + 1) {
                Fact::Not { b, out } if b == found_prev => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected not(found_real) before inner {i}, got {f:?}",
                        base + 1
                    ))
                }
            };
            let take = match *fact(base + 2) {
                Fact::And { b1, b2, out } if b1 == is_real && b2 == not_found => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected and(is_real_{i}, not_found), got {f:?}",
                        base + 2
                    ))
                }
            };
            let folded = i == 0;
            if folded {
                check(
                    not_found == one,
                    "inner 0: not(false) should fold to the constant one".into(),
                )?;
                check(
                    take == is_real,
                    "inner 0: and(is_real_0, one) should fold to is_real_0".into(),
                )?;
            }
            let src = [
                inners[i].pis[layout::BLOCK_HASH],
                inners[i].pis[layout::BLOCK_HASH + 1],
                inners[i].pis[layout::BLOCK_HASH + 2],
                inners[i].pis[layout::BLOCK_HASH + 3],
                inners[i].pis[layout::BLOCK_NUMBER],
                inners[i].pis[layout::ASSET_ID],
                inners[i].pis[layout::VOLUME_FEE_BPS],
            ];
            let mut ref_now = Vec::with_capacity(7);
            for (k, s) in src.iter().enumerate() {
                match *fact(base + 3 + k) {
                    Fact::Select { b, x, y, out } if b == take && x == *s && y == ref_prev[k] => {
                        ref_now.push(out)
                    }
                    ref f => {
                        return Err(format!(
                            "call {}: expected reference select {k} of inner {i}, got {f:?}",
                            base + 3 + k
                        ))
                    }
                }
            }
            let found = match *fact(base + 10) {
                Fact::Or { b1, b2, out } if b1 == found_prev && b2 == is_real => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected or(found_real, is_real_{i}), got {f:?}",
                        base + 10
                    ))
                }
            };
            if folded {
                check(
                    found == is_real,
                    "inner 0: or(false, is_real_0) should fold to is_real_0".into(),
                )?;
            }
            let inner = &mut inners[i];
            inner.is_real = base;
            if !folded {
                inner.not_found = Some(base + 1);
                inner.take = Some(base + 2);
            }
            if !folded && i + 1 < n {
                inner.found = Some(base + 10);
            }
            inner.sel_block = [base + 3, base + 4, base + 5, base + 6];
            inner.sel_number = base + 7;
            inner.sel_asset = base + 8;
            inner.sel_fee = base + 9;
            found_prev = found;
            ref_prev = ref_now;
        }
        let block_ref: [Target; 4] = [ref_prev[0], ref_prev[1], ref_prev[2], ref_prev[3]];
        let (number_ref, asset_ref, fee_ref) = (ref_prev[4], ref_prev[5], ref_prev[6]);

        // Consistency checks.
        for i in 0..n {
            let base = 18 * n + 15 * i;
            let d = dummy(i);
            let pis_i = inners[i].pis.clone();
            let scalar =
                |k: usize, src: Target, reference: Target, what: &str| -> Result<(), String> {
                    let equal = match *fact(k) {
                        Fact::IsEqual { x, y, equal, .. } if x == src && y == reference => equal,
                        ref f => {
                            return Err(format!(
                                "call {k}: expected {what} is_equal of inner {i}, got {f:?}"
                            ))
                        }
                    };
                    let ok = match *fact(k + 1) {
                        Fact::Or { b1, b2, out } if b1 == d && b2 == equal => out,
                        ref f => {
                            return Err(format!(
                                "call {}: expected or(is_dummy_{i}, {what}_matches), got {f:?}",
                                k + 1
                            ))
                        }
                    };
                    match *fact(k + 2) {
                        Fact::Connect { x, y } if x == ok && y == one => Ok(()),
                        ref f => Err(format!(
                            "call {}: expected connect({what}_ok, one), got {f:?}",
                            k + 2
                        )),
                    }
                };
            scalar(base, pis_i[layout::ASSET_ID], asset_ref, "asset")?;
            scalar(base + 3, pis_i[layout::VOLUME_FEE_BPS], fee_ref, "fee")?;
            let mut eqs = [zero; 4];
            for j in 0..4 {
                match *fact(base + 6 + j) {
                    Fact::IsEqual { x, y, equal, .. }
                        if x == pis_i[layout::BLOCK_HASH + j] && y == block_ref[j] =>
                    {
                        eqs[j] = equal
                    }
                    ref f => {
                        return Err(format!(
                            "call {}: expected block is_equal limb {j} of inner {i}, got {f:?}",
                            base + 6 + j
                        ))
                    }
                }
            }
            let and_a = match *fact(base + 10) {
                Fact::And { b1, b2, out } if b1 == eqs[0] && b2 == eqs[1] => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected block and(eq0, eq1) of inner {i}, got {f:?}",
                        base + 10
                    ))
                }
            };
            let and_b = match *fact(base + 11) {
                Fact::And { b1, b2, out } if b1 == eqs[2] && b2 == eqs[3] => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected block and(eq2, eq3) of inner {i}, got {f:?}",
                        base + 11
                    ))
                }
            };
            let matches = match *fact(base + 12) {
                Fact::And { b1, b2, out } if b1 == and_a && b2 == and_b => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected block and-root of inner {i}, got {f:?}",
                        base + 12
                    ))
                }
            };
            let ok = match *fact(base + 13) {
                Fact::Or { b1, b2, out } if b1 == d && b2 == matches => out,
                ref f => {
                    return Err(format!(
                        "call {}: expected or(is_dummy_{i}, block_matches), got {f:?}",
                        base + 13
                    ))
                }
            };
            match *fact(base + 14) {
                Fact::Connect { x, y } if x == ok && y == one => {}
                ref f => {
                    return Err(format!(
                        "call {}: expected connect(block_ok, one), got {f:?}",
                        base + 14
                    ))
                }
            }
            let inner = &mut inners[i];
            inner.asset_eq = base;
            inner.asset_or = base + 1;
            inner.asset_pin = base + 2;
            inner.fee_eq = base + 3;
            inner.fee_or = base + 4;
            inner.fee_pin = base + 5;
            inner.block_eq = [base + 6, base + 7, base + 8, base + 9];
            inner.block_and = [base + 10, base + 11, base + 12];
            inner.block_or = base + 13;
            inner.block_pin = base + 14;
        }

        // Masked forwards.
        let mut public_inputs: Vec<Target> = aggregator_address.to_vec();
        public_inputs.extend([asset_ref, fee_ref]);
        public_inputs.extend(block_ref);
        public_inputs.extend([number_ref, total]);
        let slot_limbs = slots_per_inner * layout::EXIT_SLOT_LEN;
        for i in 0..n {
            let base = 33 * n + slot_limbs * i;
            let d = dummy(i);
            for k in 0..slot_limbs {
                match *fact(base + k) {
                    Fact::Select { b, x, y, out }
                        if b == d && x == zero && y == inners[i].pis[layout::HEADER_LEN + k] =>
                    {
                        public_inputs.push(out);
                        inners[i].slot_fwd.push(base + k);
                    }
                    ref f => {
                        return Err(format!(
                            "call {}: expected slot forward {k} of inner {i}, got {f:?}",
                            base + k
                        ))
                    }
                }
            }
        }
        let null_limbs = nulls_per_inner * 4;
        let nulls_start = layout::HEADER_LEN + slot_limbs;
        for i in 0..n {
            let base = 33 * n + slot_limbs * n + null_limbs * i;
            let d = dummy(i);
            for k in 0..null_limbs {
                match *fact(base + k) {
                    Fact::Select { b, x, y, out }
                        if b == d && x == zero && y == inners[i].pis[nulls_start + k] =>
                    {
                        public_inputs.push(out);
                        inners[i].null_fwd.push(base + k);
                    }
                    ref f => {
                        return Err(format!(
                            "call {}: expected nullifier forward {k} of inner {i}, got {f:?}",
                            base + k
                        ))
                    }
                }
            }
        }
        check(
            public_inputs == t.ex.public_inputs,
            "the recorded public inputs are not [address, asset, fee, block, number, total, slots…, nullifiers…]".into(),
        )?;

        Ok(Shape {
            n,
            leaves,
            pi_len,
            slots_per_inner,
            nulls_per_inner,
            inners,
            one,
            zero,
            total,
            constants,
            calls: calls.clone(),
        })
    }

    fn fact(&self, k: usize) -> &Fact {
        &self.calls[k].fact
    }

    fn out(&self, k: usize) -> Target {
        out_of(self.fact(k)).expect("fact with an output")
    }

    /// `hf.2.….1` projection path of fact `k` in the grouped decode conjunction.
    fn projection(&self, k: usize) -> String {
        let total = self.calls.len();
        let groups = total.div_ceil(FACT_GROUP);
        let (g, j) = (k / FACT_GROUP, k % FACT_GROUP);
        let group_len = (total - g * FACT_GROUP).min(FACT_GROUP);
        let mut path = String::from("hf");
        if groups > 1 {
            path.push_str(&".2".repeat(g));
            if g + 1 < groups {
                path.push_str(".1");
            }
        }
        path.push_str(&".2".repeat(j));
        if j + 1 < group_len {
            path.push_str(".1");
        }
        path
    }
}

fn a(t: Target) -> String {
    format!("a ({})", lean_target(t))
}

/// Render the bridge module for a recorded public-batch wrapper.
pub fn render(shape: &Shape) -> String {
    let n = shape.n;
    let s = shape.slots_per_inner;
    let u = shape.nulls_per_inner;
    let pi_len = shape.pi_len;
    let circuit = format!("publicBatchWrapper{n}");
    let decode = format!("{circuit}_decode");
    let inners = &shape.inners;
    let mut o = String::new();
    macro_rules! w {
        ($($arg:tt)*) => { writeln!(o, $($arg)*).unwrap() }
    }

    w!("/-");
    w!("  AUTO-GENERATED by `constraint-exporter` (`src/public_wrapper.rs`) from the recorded");
    w!("  `n_inner = {n}` public-batch wrapper trace `public_batch_wrapper_n{n}.json`. Do not edit;");
    w!("  regenerate with `cargo run -p qp-plonky2-constraint-exporter --bin export-constraints`.");
    w!("");
    w!("  Step 8e — `Generated/PublicBatchWrapper{n}.lean` proves, from `Satisfies` on the wiring the");
    w!(
        "  real `CircuitBuilder` emitted, the meaning of each of the {} gadget calls",
        shape.calls.len()
    );
    w!(
        "  `build_public_batch_constraints` made over {n} `{}`-leaf private batches. This module",
        shape.leaves
    );
    w!("  reads the inner outputs and the aggregated output off the named targets and public inputs");
    w!("  and discharges every hypothesis of `public_batch_val` (`Plonky2Bridge/PublicBatch.lean`),");
    w!("  so `public_batch_end_to_end` is restated on the wiring with `private_batch_proof_sound`");
    w!("  as its only axiom (`end_to_end_wired`).");
    w!("-/");
    w!("import Plonky2Bridge.PublicBatch");
    w!("import Plonky2Spec.Generated.PublicBatchWrapper{n}");
    w!("");
    w!("namespace Plonky2Bridge.PublicWrapper{n}");
    w!("");
    w!("open Plonky2Spec (bselect band bnot bor scanStep)");
    w!("open Plonky2Spec.Wiring");
    w!("open Plonky2Spec.Generated ({circuit} {decode})");
    w!("open WormholeSpec (Digest Felt PrivateBatchOutput PublicBatchOutput ExitSlot RandomOracle RPublicBatch");
    w!("  RPrivateBatch PrivateBatchProofAccepted private_batch_proof_sound RPublicBatch_totalExitSlots)");
    w!("");
    w!("variable {{p : ℕ}} [Fact p.Prime]");
    w!("");
    w!("/-! ### Reading the spec objects off the wiring -/");
    w!("");
    w!("/-- The {pi_len} public inputs of inner proof `i` (`inner_pis_i`), at the private-batch");
    w!("    `aggregated_output` layout. -/");
    w!("def innerPis : Fin {n} → Fin {pi_len} → Target");
    for i in 0..n {
        w!("  | {i} => {circuit}.inner_pis_{i}");
    }
    w!("");
    w!("/-- The field wires of inner `i` the wrapper reads, with the `is_equal` witnesses of its");
    w!("    `bytes_digest_eq(block_hash, 0)` dummy check. -/");
    w!("def row (a : Assignment p) : Fin {n} → InnerRow p");
    for (i, inner) in inners.iter().enumerate() {
        let (eqs, invs): (Vec<String>, Vec<String>) = inner
            .dummy_eq
            .iter()
            .map(|&k| match *shape.fact(k) {
                Fact::IsEqual { equal, inv, .. } => (a(equal), a(inv)),
                _ => unreachable!(),
            })
            .unzip();
        w!("  | {i} =>");
        w!(
            "    {{ blockHash := fun j => a (innerPis {i} ⟨{} + j, by omega⟩)",
            layout::BLOCK_HASH
        );
        w!("      dummyEq := ![{}]", eqs.join(", "));
        w!("      dummyInv := ![{}]", invs.join(", "));
        w!(
            "      blockNumber := a (innerPis {i} {}), assetId := a (innerPis {i} {}), fee := a (innerPis {i} {})",
            layout::BLOCK_NUMBER,
            layout::ASSET_ID,
            layout::VOLUME_FEE_BPS
        );
        let slots: Vec<String> = (0..s)
            .map(|k| {
                let b = layout::HEADER_LEN + layout::EXIT_SLOT_LEN * k;
                format!(
                    "(a (innerPis {i} {b}), fun j => a (innerPis {i} ⟨{} + j, by omega⟩))",
                    b + 1
                )
            })
            .collect();
        w!("      slots := [{}]", slots.join(",\n        "));
        let nulls: Vec<String> = (0..u)
            .map(|k| {
                let b = layout::HEADER_LEN + layout::EXIT_SLOT_LEN * s + 4 * k;
                format!("fun j => a (innerPis {i} ⟨{b} + j, by omega⟩)")
            })
            .collect();
        w!("      nulls := [{}] }}", nulls.join(",\n        "));
    }
    w!("");
    w!("/-- Inner `i`'s private-batch output, decoded through `.val`. -/");
    w!("def inner (a : Assignment p) (i : Fin {n}) : PrivateBatchOutput :=");
    w!(
        "  {{ numExitSlots := (a (innerPis i {})).val, assetId := (a (innerPis i {})).val,",
        layout::NUM_EXIT_SLOTS,
        layout::ASSET_ID
    );
    w!(
        "    volumeFeeBps := (a (innerPis i {})).val, blockHash := valDigest (row a i).blockHash,",
        layout::VOLUME_FEE_BPS
    );
    w!(
        "    blockNumber := (a (innerPis i {})).val, exitSlots := (row a i).slots.map valSlot,",
        layout::BLOCK_NUMBER
    );
    w!("    nullifiers := (row a i).nulls.map valDigest }}");
    w!("");
    let pairs: Vec<String> = (0..n)
        .map(|i| format!("(row a {i}, inner a {i})"))
        .collect();
    w!(
        "def rows (a : Assignment p) : List (InnerPair p) := [{}]",
        pairs.join(", ")
    );
    let inner_list: Vec<String> = (0..n).map(|i| format!("inner a {i}")).collect();
    w!(
        "def inners (a : Assignment p) : List PrivateBatchOutput := [{}]",
        inner_list.join(", ")
    );
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("theorem rows_inners (a : Assignment p) : (rows a).map Prod.snd = inners a := rfl");
    w!("");
    w!("/-- The `aggregator_address` witness, decoded. -/");
    w!("def addr (a : Assignment p) : Digest := valDigest fun j => a ({circuit}.aggregator_address j)");
    w!("");
    w!("/-- Public input `k` of the wrapper, decoded through `.val`. -/");
    w!("def pv (a : Assignment p) (k : ℕ) : Felt :=");
    w!("  (a (({circuit} p).publicInputs.getD k (.virt 0))).val");
    w!("");
    // Output slots and nullifiers, as literal lists over the forwarded wires.
    let slot_wires: Vec<Target> = inners
        .iter()
        .flat_map(|i| i.slot_fwd.iter().map(|&k| shape.out(k)))
        .collect();
    let null_wires: Vec<Target> = inners
        .iter()
        .flat_map(|i| i.null_fwd.iter().map(|&k| shape.out(k)))
        .collect();
    let v = |t: Target| format!("({}).val", a(t));
    let slot_list: Vec<String> = slot_wires
        .chunks(layout::EXIT_SLOT_LEN)
        .map(|c| {
            format!(
                "⟨{}, ⟨{}, {}, {}, {}⟩⟩",
                v(c[0]),
                v(c[1]),
                v(c[2]),
                v(c[3]),
                v(c[4])
            )
        })
        .collect();
    let null_list: Vec<String> = null_wires
        .chunks(4)
        .map(|c| format!("⟨{}, {}, {}, {}⟩", v(c[0]), v(c[1]), v(c[2]), v(c[3])))
        .collect();
    let slot_list_s = format!("[{}]", slot_list.join(",\n      "));
    let null_list_s = format!("[{}]", null_list.join(",\n      "));
    w!("/-- The aggregated output at the public-batch layout: `[aggregator_address(4), asset_id,");
    w!("    volume_fee_bps, block_hash(4), block_number, total_exit_slots]`, `{n} · {s}` forwarded exit");
    w!("    slots `[sum, account(4)]`, `{n} · {u}` forwarded nullifiers. -/");
    w!("def out (a : Assignment p) : PublicBatchOutput :=");
    w!("  {{ aggregatorAddress := ⟨pv a 0, pv a 1, pv a 2, pv a 3⟩, assetId := pv a 4, volumeFeeBps := pv a 5,");
    w!("    blockHash := ⟨pv a 6, pv a 7, pv a 8, pv a 9⟩, blockNumber := pv a 10, totalExitSlots := pv a 11,");
    w!("    exitSlots := {slot_list_s},");
    w!("    nullifiers := {null_list_s} }}");
    w!("");

    // The scan in normal form.
    w!("/-! ### The {n}-row scan -/");
    w!("");
    w!("/-- `scanRef` over {n} rows, as the circuit's `select` chain: `take_0` is `is_real_0`");
    w!("    (`and(is_real_0, not(false))` folds), `take_i` is `and(is_real_i, not(found_{{i-1}}))`. -/");
    let ts: Vec<String> = (0..n).map(|i| format!("t{i}")).collect();
    w!(
        "theorem scanRef_{n} ({} : InnerPair p) (f : InnerPair p → ZMod p) :",
        ts.join(" ")
    );
    w!("    scanRef [{}] f", ts.join(", "));
    let nf = scan_normal_form(n, &|i| format!("t{i}.1.isDummy"), &|i| format!("(f t{i})"));
    w!("      = {nf} := by");
    w!("  simp only [scanRef, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, scanStep,");
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
            } else {
                "ktot".to_string()
            }
        })
        .collect();

    w!("/-- **The recorded public-batch wrapper satisfies `RPublicBatch`.** Every satisfying");
    w!("    assignment decodes to an `RPublicBatch` instance on the {n} inner outputs, the");
    w!("    aggregator address and the aggregated output; the oracle is irrelevant (the wrapper");
    w!("    hashes nothing). -/");
    w!("theorem sound (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    w!("    RPublicBatch ro (inners a) (addr a) (out a) := by");
    w!("  obtain ⟨{}⟩ := consts a h", const_names.join(", "));
    w!("  have hf := {decode} a h");
    // Projections.
    let mut proj = |name: String, k: usize| w!("  have {name} := {}", shape.projection(k));
    for (i, inner) in inners.iter().enumerate() {
        for j in 0..4 {
            proj(format!("e{i}_{j}"), inner.dummy_eq[j]);
        }
        proj(format!("da{i}"), inner.dummy_and[0]);
        proj(format!("db{i}"), inner.dummy_and[1]);
        proj(format!("d{i}"), inner.dummy_and[2]);
    }
    for (i, inner) in inners.iter().enumerate() {
        proj(format!("r{i}"), inner.is_real);
        if let Some(k) = inner.not_found {
            proj(format!("nf{i}"), k);
        }
        if let Some(k) = inner.take {
            proj(format!("tk{i}"), k);
        }
        for j in 0..4 {
            proj(format!("sb{i}_{j}"), inner.sel_block[j]);
        }
        proj(format!("sn{i}"), inner.sel_number);
        proj(format!("sa{i}"), inner.sel_asset);
        proj(format!("sf{i}"), inner.sel_fee);
        if let Some(k) = inner.found {
            proj(format!("fo{i}"), k);
        }
    }
    for (i, inner) in inners.iter().enumerate() {
        proj(format!("ca{i}"), inner.asset_eq);
        proj(format!("cao{i}"), inner.asset_or);
        proj(format!("cak{i}"), inner.asset_pin);
        proj(format!("cf{i}"), inner.fee_eq);
        proj(format!("cfo{i}"), inner.fee_or);
        proj(format!("cfk{i}"), inner.fee_pin);
        for j in 0..4 {
            proj(format!("cb{i}_{j}"), inner.block_eq[j]);
        }
        proj(format!("cba{i}"), inner.block_and[0]);
        proj(format!("cbb{i}"), inner.block_and[1]);
        proj(format!("cbc{i}"), inner.block_and[2]);
        proj(format!("cbo{i}"), inner.block_or);
        proj(format!("cbk{i}"), inner.block_pin);
    }
    for (i, inner) in inners.iter().enumerate() {
        for (k, &c) in inner.slot_fwd.iter().enumerate() {
            proj(format!("fs{i}_{k}"), c);
        }
    }
    for (i, inner) in inners.iter().enumerate() {
        for (k, &c) in inner.null_fwd.iter().enumerate() {
            proj(format!("fn{i}_{k}"), c);
        }
    }
    w!("");
    let all_eqs: Vec<String> = (0..n)
        .flat_map(|i| (0..4).map(move |j| format!("e{i}_{j}")))
        .collect();
    w!("  rw [kzero] at {}", all_eqs.join(" "));
    for (i, inner) in inners.iter().enumerate() {
        let d = shape.out(inner.dummy_and[2]);
        w!(
            "  have hd{i} : {} = (row a {i}).isDummy := by rw [d{i}, da{i}, db{i}]; rfl",
            a(d)
        );
    }
    // hsel: the select chain on the circuit's take wires.
    let xs: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    let take_wire = |i: usize| -> Target {
        match inners[i].take {
            Some(k) => shape.out(k),
            None => shape.out(inners[i].is_real),
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
        if inners[i].found.is_some() {
            rws.push(format!("fo{i}"));
        }
    }
    for i in (0..n).rev() {
        rws.push(format!("r{i}"));
    }
    for i in 0..n {
        rws.push(format!("hd{i}"));
    }
    w!("    rw [{}]", rws.join(", "));
    // Block reference limbs.
    let block_wires: Vec<Target> = (0..4)
        .map(|j| shape.out(inners[n - 1].sel_block[j]))
        .collect();
    let bw: Vec<String> = block_wires.iter().map(|t| lean_target(*t)).collect();
    w!(
        "  have hblk : ∀ j, a (![{}] j) = blockRef (rows a) j := by",
        bw.join(", ")
    );
    w!("    intro j");
    w!("    simp only [blockRef, rows, scanRef_{n}]");
    w!("    fin_cases j");
    for j in 0..4 {
        let sels: Vec<String> = (0..n).rev().map(|i| format!("sb{i}_{j}")).collect();
        w!("    · show a ({}) = _", bw[j]);
        w!("      rw [{}, kzero, hsel]; rfl", sels.join(", "));
    }
    for j in 0..4 {
        w!(
            "  have hb{j} : a ({}) = blockRef (rows a) {j} := hblk {j}",
            bw[j]
        );
    }
    let scalar_ref = |name: &str, field: &str, sel: &dyn Fn(usize) -> usize| -> String {
        let wire = shape.out(sel(n - 1));
        let sels: Vec<String> = (0..n).rev().map(|i| format!("{name}{i}")).collect();
        format!(
            "  have h{name}R : {} = scanRef (rows a) fun t => t.1.{field} := by\n    rw [rows, scanRef_{n}, {}, kzero, hsel]; rfl",
            a(wire),
            sels.join(", ")
        )
    };
    w!(
        "{}",
        scalar_ref("sn", "blockNumber", &|i| inners[i].sel_number)
    );
    w!("{}", scalar_ref("sa", "assetId", &|i| inners[i].sel_asset));
    w!("{}", scalar_ref("sf", "fee", &|i| inners[i].sel_fee));
    let number_wire = a(shape.out(inners[n - 1].sel_number));
    let asset_wire = a(shape.out(inners[n - 1].sel_asset));
    let fee_wire = a(shape.out(inners[n - 1].sel_fee));
    // Membership.
    let alts: Vec<String> = (0..n)
        .map(|i| format!("t = (row a {i}, inner a {i})"))
        .collect();
    w!("  have hmem : ∀ t ∈ rows a, {} := by", alts.join(" ∨ "));
    w!("    intro t ht");
    w!("    simpa only [rows, List.mem_cons, List.not_mem_nil, or_false] using ht");
    let ps: Vec<String> = (0..n)
        .map(|i| format!("P (row a {i}, inner a {i})"))
        .collect();
    let hs: Vec<String> = (0..n).map(|i| format!("h{i}")).collect();
    let rfls = vec!["rfl"; n].join(" | ");
    w!(
        "  have hdec : ∀ (P : InnerPair p → Prop), {} →",
        ps.join(" → ")
    );
    w!("      ∀ t ∈ rows a, P t := by");
    w!("    intro P {} t ht", hs.join(" "));
    w!("    rcases hmem t ht with {rfls}");
    for i in 0..n {
        w!("    · exact h{i}");
    }
    let dec_rfls = vec!["rfl"; n].join(" ");
    w!("  refine public_batch_val ro (rows a) (k := {s}) rfl ?_ (hdec _ {dec_rfls}) (hdec _ {dec_rfls})");
    w!("    (hdec _ {dec_rfls}) (hdec _ {dec_rfls}) (hdec _ {dec_rfls}) (hdec _ {dec_rfls}) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_");
    w!("    ?_ ?_");
    w!("  · -- dummy checks");
    w!("    intro t ht");
    w!("    rcases hmem t ht with {rfls}");
    for _ in 0..n {
        w!("    · intro j; fin_cases j <;> assumption");
    }
    w!("  · -- header: block hash");
    w!(
        "    show (⟨{}⟩ : Digest)",
        block_wires
            .iter()
            .map(|t| v(*t))
            .collect::<Vec<_>>()
            .join(", ")
    );
    w!("      = ⟨(blockRef (rows a) 0).val, (blockRef (rows a) 1).val, (blockRef (rows a) 2).val,");
    w!("          (blockRef (rows a) 3).val⟩");
    w!("    rw [hb0, hb1, hb2, hb3]");
    w!("  · show ({number_wire}).val = _");
    w!("    rw [hsnR]");
    w!("  · show ({asset_wire}).val = _");
    w!("    rw [hsaR]");
    w!("  · show ({fee_wire}).val = _");
    w!("    rw [hsfR]");
    w!("  · -- asset consistency");
    w!("    intro t ht");
    w!("    rcases hmem t ht with {rfls}");
    for i in 0..n {
        w!("    · exact ⟨_, _, hsaR ▸ ca{i}, by rw [← hd{i}, ← cao{i}, cak{i}, kone]⟩");
    }
    w!("  · -- fee consistency");
    w!("    intro t ht");
    w!("    rcases hmem t ht with {rfls}");
    for i in 0..n {
        w!("    · exact ⟨_, _, hsfR ▸ cf{i}, by rw [← hd{i}, ← cfo{i}, cfk{i}, kone]⟩");
    }
    w!("  · -- block consistency");
    w!("    intro t ht");
    w!("    rcases hmem t ht with {rfls}");
    for (i, inner) in inners.iter().enumerate() {
        let (eqs, invs): (Vec<String>, Vec<String>) = inner
            .block_eq
            .iter()
            .map(|&k| match *shape.fact(k) {
                Fact::IsEqual { equal, inv, .. } => (a(equal), a(inv)),
                _ => unreachable!(),
            })
            .unzip();
        w!("    · refine ⟨![{}],", eqs.join(", "));
        w!("        ![{}], ?_, ?_⟩", invs.join(", "));
        w!("      · intro j; rw [← hblk j]; fin_cases j <;> assumption");
        w!(
            "      · show bor (row a {i}).isDummy (band (band ({}) ({}))",
            eqs[0],
            eqs[1]
        );
        w!("          (band ({}) ({}))) = 1", eqs[2], eqs[3]);
        w!("        rw [← hd{i}, ← cba{i}, ← cbb{i}, ← cbc{i}, ← cbo{i}, cbk{i}, kone]");
    }
    let hds: Vec<String> = (0..n).map(|i| format!("hd{i}")).collect();
    let fs_names: Vec<String> = inners
        .iter()
        .enumerate()
        .flat_map(|(i, inner)| (0..inner.slot_fwd.len()).map(move |k| format!("fs{i}_{k}")))
        .collect();
    let fn_names: Vec<String> = inners
        .iter()
        .enumerate()
        .flat_map(|(i, inner)| (0..inner.null_fwd.len()).map(move |k| format!("fn{i}_{k}")))
        .collect();
    w!("  · -- forwarded exit slots");
    w!("    show ({slot_list_s} : List ExitSlot) = _");
    w!(
        "    rw [{}, kzero, {}]",
        fs_names.join(", "),
        hds.join(", ")
    );
    w!("    rfl");
    w!("  · -- forwarded nullifiers");
    w!("    show ({null_list_s} : List Digest) = _");
    w!(
        "    rw [{}, kzero, {}]",
        fn_names.join(", "),
        hds.join(", ")
    );
    w!("    rfl");
    w!("  · intro t ht");
    w!("    rcases hmem t ht with {rfls} <;> rfl");
    let total_value = n * s;
    w!("  · show ({}).val = {n} * {s}", a(shape.total));
    w!("    rw [ktot]");
    w!("    have hc : ({total_value} : ℕ) < p := lt_of_lt_of_le (by decide) hpg");
    w!("    rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]");
    w!("");
    w!("/-- **The public-batch capstone on the recorded wiring.** `public_batch_end_to_end` with its");
    w!("    decode hypotheses discharged by `sound`: a satisfying assignment of the `n_inner = {n}`");
    w!("    public-batch wrapper whose recursion gadgets accepted every inner private-batch proof");
    w!("    (i) satisfies `RPublicBatch`, (ii) has the slot-count header equal to the sum of the");
    w!("    inners' slot counts and (iii) attests every inner's `RPrivateBatch` — through");
    w!("    `private_batch_proof_sound`, the only axiom. -/");
    w!("theorem end_to_end_wired (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hacc : ∀ o ∈ inners a, PrivateBatchProofAccepted ro o) :");
    w!("    RPublicBatch ro (inners a) (addr a) (out a)");
    w!("      ∧ (out a).totalExitSlots = ((inners a).map fun o => o.exitSlots.length).sum");
    w!("      ∧ ∀ o ∈ inners a, ∃ leaves us, RPrivateBatch ro leaves us o := by");
    w!("  have hR := sound ro hpg a h");
    w!("  exact ⟨hR, RPublicBatch_totalExitSlots hR, fun o ho => private_batch_proof_sound ro o (hacc o ho)⟩");
    w!("");
    w!("end Plonky2Bridge.PublicWrapper{n}");
    o
}

/// The normal form `simp only [scanRef, …, scanStep, bor_zero_left, bnot_zero, band_one_right]`
/// leaves `scanRef [t₀, …, tₙ₋₁] f` in: with `R_i = bnot D_i`, `F_0 = R_0`,
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

/// Build `formal/Plonky2Bridge/Generated/PublicWrapper{n}.lean` from the recorded
/// `n_inner = n` public-batch wrapper trace.
pub fn generate_public_wrapper_bridge_lean(n: usize) -> Result<String, String> {
    let circuit = format!("public_batch_wrapper_n{n}");
    let t = load(&traces_dir().join(format!("{circuit}.json")))?;
    if t.circuit != circuit {
        return Err(format!("unexpected trace circuit {:?}", t.circuit));
    }
    let shape = Shape::read(&t)?;
    if shape.n != n {
        return Err(format!("trace {circuit} has {} inners", shape.n));
    }
    Ok(render(&shape))
}
