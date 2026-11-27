//! Generates `formal/Plonky2Bridge/Generated/PublicWrapper{N}.lean` — the composition of a
//! recorded `n_inner = N` public-batch wrapper trace into `RPublicBatch` (PLAN.md Step 8e).
//!
//! `build_public_batch_constraints` makes its gadget calls in a fixed order determined by
//! `(n_inner, private_batch_shape)`: the per-inner slot-count pins, the per-inner dummy
//! checks, the first-real prefix scan, the per-inner consistency checks, then the masked
//! slot and nullifier forwards.
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
use crate::gadget::{Call, Fact};
use crate::trace::{check_verifiers, load, traces_dir, LoadedTrace};

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
    pub const NULLIFIER_LEN: usize = 4;
}

/// One inner proof's calls, by role (indices into the trace's calls).
#[derive(Debug, Clone)]
struct Inner {
    pis: Vec<Target>,
    /// `connect(num_exit_slots, constant(slots_per_inner))`.
    count_pin: usize,
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
    /// `constant(slots_per_inner)`, every inner's slot-count header.
    slots_const: Target,
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
        // The inner shape `(leaves, exit slots)` is not recoverable from the PI length alone:
        // the leaf count comes from the child circuit the recursion gadgets verify, the slot
        // count from what remains of the layout `header ‖ slots × 5 ‖ nullifiers × 4`.
        let child =
            t.ex.verifiers
                .first()
                .map(|(tag, _)| tag.clone())
                .ok_or("trace has no verify_proof gadgets")?;
        let leaves: usize = child
            .strip_prefix("private_batch_wrapper_n")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("verifier child {child:?} is not a private-batch wrapper"))?;
        check_verifiers(t, &child, &pis)?;
        let nulls_per_inner = leaves;
        let null_len = nulls_per_inner * layout::NULLIFIER_LEN;
        if pi_len < layout::HEADER_LEN + null_len
            || (pi_len - layout::HEADER_LEN - null_len) % layout::EXIT_SLOT_LEN != 0
        {
            return Err(format!(
                "inner PI length {pi_len} is not 8 + 5·slots + 4·{leaves}"
            ));
        }
        let slots_per_inner = (pi_len - layout::HEADER_LEN - null_len) / layout::EXIT_SLOT_LEN;
        if slots_per_inner < 2 {
            return Err(format!(
                "expected at least two exit slots per inner, found {slots_per_inner}"
            ));
        }
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
        let slots_value = slots_per_inner as u64;
        let slots_const = const_target(slots_value)?;
        let total_value = (n * slots_per_inner) as u64;
        let total = const_target(total_value)?;
        if constants.len() != 4 {
            return Err(format!(
                "expected constants {{1, 0, {slots_value}, {total_value}}}, found {constants:?}"
            ));
        }

        let calls = &t.calls;
        let expected_calls = 34 * n + 5 * slots_per_inner * n + 4 * nulls_per_inner * n;
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

        // Slot-count pins.
        for (i, pis_i) in pis.iter().enumerate() {
            match *fact(i) {
                Fact::Connect { x, y }
                    if x == pis_i[layout::NUM_EXIT_SLOTS] && y == slots_const => {}
                ref f => {
                    return Err(format!(
                        "call {i}: expected connect(num_exit_slots_{i}, slots_per_inner), got {f:?}"
                    ))
                }
            }
        }

        let mut inners = Vec::with_capacity(n);
        // Dummy checks.
        for (i, pis_i) in pis.iter().enumerate() {
            let base = n + 7 * i;
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
                count_pin: i,
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
            let base = 8 * n + 11 * i;
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
            let base = 19 * n + 15 * i;
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
            let base = 34 * n + slot_limbs * i;
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
            let base = 34 * n + slot_limbs * n + null_limbs * i;
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
            slots_const,
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

    /// The per-call decode lemma of fact `k`, parenthesized so it is one `rw` argument.
    /// Projecting the grouped conjunction (`hf.2.….1`) is superlinear once a trace has
    /// tens of thousands of calls, and introducing every lemma as a local hypothesis is too.
    fn projection(&self, k: usize) -> String {
        format!("(publicBatchWrapper{}_f{k} a h)", self.n)
    }
}

fn a(t: Target) -> String {
    format!("a ({})", lean_target(t))
}

fn fact_term(bound: &std::collections::HashMap<String, String>, name: &str) -> String {
    bound
        .get(name)
        .unwrap_or_else(|| panic!("missing projection {name}"))
        .clone()
}

/// Render the bridge module for a recorded public-batch wrapper.
pub fn render(shape: &Shape) -> String {
    let n = shape.n;
    let leaves = shape.leaves;
    let s = shape.slots_per_inner;
    let u = shape.nulls_per_inner;
    let pi_len = shape.pi_len;
    let circuit = format!("publicBatchWrapper{n}");
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
    w!("  so `public_batch_end_to_end` is restated on the wiring (`end_to_end_wired`), with each");
    w!("  inner's `RPrivateBatch` recovered from its accepted proof through `proof_sound`");
    w!("  and the private-batch wrapper bridge (Step 10).");
    w!("-/");
    w!("import Plonky2Bridge.PublicBatch");
    w!("import Plonky2Bridge.Generated.Wrapper{leaves}");
    w!("import Plonky2Spec.Generated.PublicBatchWrapper{n}");
    w!("");
    w!("namespace Plonky2Bridge.PublicWrapper{n}");
    w!("");
    w!("open Plonky2Spec (bselect band bnot bor scanStep)");
    w!("open Plonky2Spec.Wiring");
    w!("open Plonky2Spec.Generated");
    w!("open Plonky2Spec.Poseidon2 (St)");
    w!("open WormholeSpec (Digest Felt PrivateBatchOutput PublicBatchOutput ExitSlot RandomOracle RPublicBatch");
    w!("  RPrivateBatch RPublicBatch_totalExitSlots)");
    w!("");
    w!("variable {{p : ℕ}} [Fact p.Prime]");
    w!("");
    // `Fin n` matches of this size are checked by a unary decision tree. A final
    // catch-all keeps that check from walking all `n` successors, and the lists in
    // `out` are longer than the default recursion and heartbeat budgets.
    if n > 4 {
        w!("set_option maxRecDepth 200000");
        w!("set_option maxHeartbeats 0");
        w!("");
    }
    w!("/-! ### Reading the spec objects off the wiring -/");
    w!("");
    w!("/-- The {pi_len} public inputs of inner proof `i` (`inner_pis_i`), at the private-batch");
    w!("    `aggregated_output` layout. -/");
    w!("noncomputable def innerPis : Fin {n} → Fin {pi_len} → Target");
    for i in 0..n {
        w!("  | {i} => {circuit}.inner_pis_{i}");
    }
    w!("  | _ => {circuit}.inner_pis_0");
    w!("");
    w!("/-- The field wires of inner `i` the wrapper reads, with the `is_equal` witnesses of its");
    w!("    `bytes_digest_eq(block_hash, 0)` dummy check. -/");
    w!("noncomputable def row (a : Assignment p) : Fin {n} → InnerRow p");
    let mut emit_row = |pat: String, i: usize, inner: &Inner| {
        let (eqs, invs): (Vec<String>, Vec<String>) = inner
            .dummy_eq
            .iter()
            .map(|&k| match *shape.fact(k) {
                Fact::IsEqual { equal, inv, .. } => (a(equal), a(inv)),
                _ => unreachable!(),
            })
            .unzip();
        w!("  | {pat} =>");
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
    };
    for (i, inner) in inners.iter().enumerate() {
        emit_row(format!("{i}"), i, inner);
    }
    emit_row("_".to_string(), 0, &inners[0]);
    w!("");
    w!("/-- Inner `i`'s private-batch output, decoded through `.val`. -/");
    w!("noncomputable def inner (a : Assignment p) (i : Fin {n}) : PrivateBatchOutput :=");
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
        "noncomputable def rows (a : Assignment p) : List (InnerPair p) := [{}]",
        pairs.join(", ")
    );
    let inner_list: Vec<String> = (0..n).map(|i| format!("inner a {i}")).collect();
    w!(
        "noncomputable def inners (a : Assignment p) : List PrivateBatchOutput := [{}]",
        inner_list.join(", ")
    );
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("theorem rows_inners (a : Assignment p) : (rows a).map Prod.snd = inners a := rfl");
    w!("");
    w!("/-- The `aggregator_address` witness, decoded. -/");
    w!("noncomputable def addr (a : Assignment p) : Digest := valDigest fun j => a ({circuit}.aggregator_address j)");
    w!("");
    w!("/-- Public input `k` of the wrapper, decoded through `.val`. -/");
    w!("noncomputable def pv (a : Assignment p) (k : ℕ) : Felt :=");
    w!("  (a (({circuit} p).publicInputs.getD k (.virt 0))).val");
    w!("");
    // One list per inner. `out` concatenates them in the same right-nested shape as
    // `List.flatten`, so the forwarding proof rewrites one inner at a time instead of
    // a single goal the size of every forwarded limb (production is 86 · 32 slots).
    let v = |t: Target| format!("({}).val", a(t));
    let slot_lits: Vec<String> = inners
        .iter()
        .map(|inner| {
            let wires: Vec<Target> = inner.slot_fwd.iter().map(|&k| shape.out(k)).collect();
            let slots: Vec<String> = wires
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
            format!("[{}]", slots.join(",\n      "))
        })
        .collect();
    let null_lits: Vec<String> = inners
        .iter()
        .map(|inner| {
            let wires: Vec<Target> = inner.null_fwd.iter().map(|&k| shape.out(k)).collect();
            let nulls: Vec<String> = wires
                .chunks(layout::NULLIFIER_LEN)
                .map(|c| format!("⟨{}, {}, {}, {}⟩", v(c[0]), v(c[1]), v(c[2]), v(c[3])))
                .collect();
            format!("[{}]", nulls.join(",\n      "))
        })
        .collect();
    let nest = |name: &str| {
        let mut acc = "[]".to_string();
        for i in (0..n).rev() {
            acc = format!("{name} a {i} ++ ({acc})");
        }
        acc
    };
    w!("/-- Inner `i`'s forwarded exit slots, read off the wiring. -/");
    w!("noncomputable def exitChunk (a : Assignment p) : Fin {n} → List ExitSlot");
    for (i, lit) in slot_lits.iter().enumerate() {
        w!("  | {i} => {lit}");
    }
    w!("  | _ => []");
    w!("");
    w!("/-- Inner `i`'s forwarded nullifiers, read off the wiring. -/");
    w!("noncomputable def nullChunk (a : Assignment p) : Fin {n} → List Digest");
    for (i, lit) in null_lits.iter().enumerate() {
        w!("  | {i} => {lit}");
    }
    w!("  | _ => []");
    w!("");
    w!("/-- The aggregated output at the public-batch layout: `[aggregator_address(4), asset_id,");
    w!("    volume_fee_bps, block_hash(4), block_number, total_exit_slots]`, `{n} · {s}` forwarded exit");
    w!("    slots `[sum, account(4)]`, `{n} · {u}` forwarded nullifiers. -/");
    w!("noncomputable def out (a : Assignment p) : PublicBatchOutput :=");
    w!("  {{ aggregatorAddress := ⟨pv a 0, pv a 1, pv a 2, pv a 3⟩, assetId := pv a 4, volumeFeeBps := pv a 5,");
    w!("    blockHash := ⟨pv a 6, pv a 7, pv a 8, pv a 9⟩, blockNumber := pv a 10, totalExitSlots := pv a 11,");
    w!("    exitSlots := {},", nest("exitChunk"));
    w!("    nullifiers := {} }}", nest("nullChunk"));
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
            } else if *t == shape.slots_const {
                "kslots".to_string()
            } else {
                "ktot".to_string()
            }
        })
        .collect();

    // Per-call lemmas are applied where they are used. Naming all of them in one
    // context (production is ~20k calls) makes elaboration quadratic in the trace.
    let mut bound: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    {
        let mut bind = |name: String, k: usize| {
            bound.insert(name, shape.projection(k));
        };
        for (i, inner) in inners.iter().enumerate() {
            bind(format!("cp{i}"), inner.count_pin);
        }
        for (i, inner) in inners.iter().enumerate() {
            for j in 0..4 {
                bind(format!("e{i}_{j}"), inner.dummy_eq[j]);
            }
            bind(format!("da{i}"), inner.dummy_and[0]);
            bind(format!("db{i}"), inner.dummy_and[1]);
            bind(format!("d{i}"), inner.dummy_and[2]);
        }
        for (i, inner) in inners.iter().enumerate() {
            bind(format!("r{i}"), inner.is_real);
            if let Some(k) = inner.not_found {
                bind(format!("nf{i}"), k);
            }
            if let Some(k) = inner.take {
                bind(format!("tk{i}"), k);
            }
            for j in 0..4 {
                bind(format!("sb{i}_{j}"), inner.sel_block[j]);
            }
            bind(format!("sn{i}"), inner.sel_number);
            bind(format!("sa{i}"), inner.sel_asset);
            bind(format!("sf{i}"), inner.sel_fee);
            if let Some(k) = inner.found {
                bind(format!("fo{i}"), k);
            }
        }
        for (i, inner) in inners.iter().enumerate() {
            bind(format!("ca{i}"), inner.asset_eq);
            bind(format!("cao{i}"), inner.asset_or);
            bind(format!("cak{i}"), inner.asset_pin);
            bind(format!("cf{i}"), inner.fee_eq);
            bind(format!("cfo{i}"), inner.fee_or);
            bind(format!("cfk{i}"), inner.fee_pin);
            for j in 0..4 {
                bind(format!("cb{i}_{j}"), inner.block_eq[j]);
            }
            bind(format!("cba{i}"), inner.block_and[0]);
            bind(format!("cbb{i}"), inner.block_and[1]);
            bind(format!("cbc{i}"), inner.block_and[2]);
            bind(format!("cbo{i}"), inner.block_or);
            bind(format!("cbk{i}"), inner.block_pin);
        }
        for (i, inner) in inners.iter().enumerate() {
            for (k, &c) in inner.slot_fwd.iter().enumerate() {
                bind(format!("fs{i}_{k}"), c);
            }
        }
        for (i, inner) in inners.iter().enumerate() {
            for (k, &c) in inner.null_fwd.iter().enumerate() {
                bind(format!("fn{i}_{k}"), c);
            }
        }
    }
    let obtain = format!("obtain ⟨{}⟩ := consts a h", const_names.join(", "));
    for (i, inner) in inners.iter().enumerate() {
        let d = shape.out(inner.dummy_and[2]);
        let hd = format!(
            "have hd : {} = (row a {i}).isDummy := by rw [{}, {}, {}]; rfl",
            a(d),
            fact_term(&bound, &format!("d{i}")),
            fact_term(&bound, &format!("da{i}")),
            fact_term(&bound, &format!("db{i}"))
        );
        let slot_rules: Vec<String> = (0..inner.slot_fwd.len())
            .map(|k| fact_term(&bound, &format!("fs{i}_{k}")))
            .collect();
        w!("theorem exitChunk_eq_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    exitChunk a {i} = ((row a {i}).slots.map (maskSlot (row a {i}).isDummy)).map valSlot := by");
        w!("  unfold exitChunk");
        w!("  {obtain}");
        w!("  {hd}");
        if !slot_rules.is_empty() {
            w!("  simp only [{}]", slot_rules.join(", "));
        }
        w!("  rw [kzero, hd]");
        w!("  rfl");
        w!("");
        let null_rules: Vec<String> = (0..inner.null_fwd.len())
            .map(|k| fact_term(&bound, &format!("fn{i}_{k}")))
            .collect();
        w!("theorem nullChunk_eq_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    nullChunk a {i} = ((row a {i}).nulls.map fun x j => bselect (row a {i}).isDummy 0 (x j)).map valDigest := by");
        w!("  unfold nullChunk");
        w!("  {obtain}");
        w!("  {hd}");
        if !null_rules.is_empty() {
            w!("  simp only [{}]", null_rules.join(", "));
        }
        w!("  rw [kzero, hd]");
        w!("  rfl");
        w!("");
    }
    let take_tgt = |i: usize| -> Target {
        match inners[i].take {
            Some(k) => shape.out(k),
            None => shape.out(inners[i].is_real),
        }
    };
    let dummy_expr = |i: usize| format!("(row a {i}).isDummy");
    let acc_expr = |i: usize| -> String {
        let mut acc = format!("(bnot {})", dummy_expr(0));
        for j in 1..=i {
            acc = format!("(bor {acc} (bnot {}))", dummy_expr(j));
        }
        acc
    };
    let take_expr = |i: usize| -> String {
        if i == 0 {
            format!("(bnot {})", dummy_expr(0))
        } else {
            format!("(band (bnot {}) (bnot {}))", dummy_expr(i), acc_expr(i - 1))
        }
    };
    let hd_of = |i: usize| -> String {
        let d = shape.out(inners[i].dummy_and[2]);
        format!(
            "have hd : {} = (row a {i}).isDummy := by rw [{}, {}, {}]; rfl",
            a(d),
            fact_term(&bound, &format!("d{i}")),
            fact_term(&bound, &format!("da{i}")),
            fact_term(&bound, &format!("db{i}"))
        )
    };
    w!("/-- The take wire of inner `i` (`is_real` for inner 0, where the scan folds). -/");
    w!("noncomputable def takeWire : Fin {n} → Target");
    for i in 0..n {
        w!("  | {i} => {}", lean_target(take_tgt(i)));
    }
    w!("  | _ => {}", lean_target(take_tgt(0)));
    w!("");
    for i in 0..n {
        let real = a(shape.out(inners[i].is_real));
        let hd = hd_of(i);
        w!("theorem realEq_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    {real} = (bnot (row a {i}).isDummy) := by");
        w!("  {hd}");
        w!("  rw [{}, hd]", fact_term(&bound, &format!("r{i}")));
        w!("");
        if inners[i].found.is_some() {
            w!("theorem foundEq_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
            w!(
                "    {} = {} := by",
                a(shape.out(inners[i].found.unwrap())),
                acc_expr(i)
            );
            w!("  {hd}");
            w!("  rw [{}]", fact_term(&bound, &format!("fo{i}")));
            w!("  rw [{}, hd]", fact_term(&bound, &format!("r{i}")));
            if i == 1 {
                w!("  rw [realEq_0 a h]");
            } else {
                w!("  rw [foundEq_{} a h]", i - 1);
            }
            w!("");
        }
        w!("theorem takeEq_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    {} = {} := by", a(take_tgt(i)), take_expr(i));
        if i == 0 {
            w!("  exact realEq_0 a h");
        } else {
            w!("  {hd}");
            w!("  rw [{}]", fact_term(&bound, &format!("tk{i}")));
            w!("  rw [{}, hd]", fact_term(&bound, &format!("r{i}")));
            w!("  rw [{}]", fact_term(&bound, &format!("nf{i}")));
            if i == 1 {
                w!("  rw [realEq_0 a h]");
            } else {
                w!("  rw [foundEq_{} a h]", i - 1);
            }
        }
        w!("");
    }
    let xs_sc: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    let mut chain_sc = format!("bselect ({}) x0 0", a(take_tgt(0)));
    for i in 1..n {
        chain_sc = format!("bselect ({}) x{i} ({chain_sc})", a(take_tgt(i)));
    }
    let nf_sc = scan_normal_form(n, &|i| format!("(row a {i}).isDummy"), &|i| format!("x{i}"));
    w!("theorem selChain_eq (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    w!("    ∀ {} : ZMod p, {chain_sc}", xs_sc.join(" "));
    w!("      = {nf_sc} := by");
    w!("  intro {}", xs_sc.join(" "));
    for i in (0..n).rev() {
        w!("  rw [takeEq_{i} a h]");
    }
    w!("");
    // One theorem per scanned output. Inlining the chain in `sound` keeps the
    // normal form in the local context for every later tactic step.
    let block_wires: Vec<Target> = (0..4)
        .map(|j| shape.out(inners[n - 1].sel_block[j]))
        .collect();
    let bw: Vec<String> = block_wires.iter().map(|t| lean_target(*t)).collect();
    for j in 0..4 {
        w!("theorem blockLimb_{j} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    a ({}) = blockRef (rows a) {j} := by", bw[j]);
        w!("  {obtain}");
        for i in (0..n).rev() {
            w!("  rw [{}]", fact_term(&bound, &format!("sb{i}_{j}")));
        }
        w!("  rw [kzero, selChain_eq a h]");
        w!("  rw [blockRef, rows, scanRef_{n}]");
        w!("  rfl");
        w!("");
    }
    let emit_scan = |name: &str, field: &str, last: usize| {
        let wire = a(shape.out(last));
        let mut lines = vec![
            format!("theorem scan_{name} (a : Assignment p) (h : Satisfies ({circuit} p) a) :"),
            format!("    {wire} = scanRef (rows a) fun t => t.1.{field} := by"),
            format!("  {obtain}"),
        ];
        for i in (0..n).rev() {
            lines.push(format!(
                "  rw [{}]",
                fact_term(&bound, &format!("{name}{i}"))
            ));
        }
        lines.push("  rw [kzero, selChain_eq a h]".to_string());
        lines.push(format!("  rw [rows, scanRef_{n}]"));
        lines.push("  rfl".to_string());
        lines.join("\n")
    };
    w!(
        "{}",
        emit_scan("sn", "blockNumber", inners[n - 1].sel_number)
    );
    w!("");
    w!("{}", emit_scan("sa", "assetId", inners[n - 1].sel_asset));
    w!("");
    w!("{}", emit_scan("sf", "fee", inners[n - 1].sel_fee));
    w!("");
    for (i, inner) in inners.iter().enumerate() {
        let d = shape.out(inner.dummy_and[2]);
        let hd = format!(
            "have hd : {} = (row a {i}).isDummy := by rw [{}, {}, {}]; rfl",
            a(d),
            fact_term(&bound, &format!("d{i}")),
            fact_term(&bound, &format!("da{i}")),
            fact_term(&bound, &format!("db{i}"))
        );
        w!("theorem dummyAt_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    DummyCheck (row a {i}) := by");
        w!("  {obtain}");
        w!("  intro j; fin_cases j");
        for j in 0..4 {
            w!(
                "  · have e := {}; rw [kzero] at e; exact e",
                fact_term(&bound, &format!("e{i}_{j}"))
            );
        }
        w!("");
        w!("theorem assetAt_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    ConsistencyCheck (row a {i}).isDummy (row a {i}).assetId");
        w!("      (scanRef (rows a) fun t => t.1.assetId) := by");
        w!("  {obtain}");
        w!("  {hd}");
        w!(
            "  exact ⟨_, _, scan_sa a h ▸ {}, by rw [← hd, ← {}, {}, kone]⟩",
            fact_term(&bound, &format!("ca{i}")),
            fact_term(&bound, &format!("cao{i}")),
            fact_term(&bound, &format!("cak{i}"))
        );
        w!("");
        w!("theorem feeAt_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    ConsistencyCheck (row a {i}).isDummy (row a {i}).fee");
        w!("      (scanRef (rows a) fun t => t.1.fee) := by");
        w!("  {obtain}");
        w!("  {hd}");
        w!(
            "  exact ⟨_, _, scan_sf a h ▸ {}, by rw [← hd, ← {}, {}, kone]⟩",
            fact_term(&bound, &format!("cf{i}")),
            fact_term(&bound, &format!("cfo{i}")),
            fact_term(&bound, &format!("cfk{i}"))
        );
        w!("");
        let (eqs, invs): (Vec<String>, Vec<String>) = inner
            .block_eq
            .iter()
            .map(|&k| match *shape.fact(k) {
                Fact::IsEqual { equal, inv, .. } => (a(equal), a(inv)),
                _ => unreachable!(),
            })
            .unzip();
        w!("theorem blockAt_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
        w!("    DigestConsistencyCheck (row a {i}).isDummy (row a {i}).blockHash (blockRef (rows a)) := by");
        w!("  {obtain}");
        w!("  {hd}");
        w!(
            "  have hblk : ∀ j, a (![{}] j) = blockRef (rows a) j := by",
            bw.join(", ")
        );
        w!("    intro j; fin_cases j");
        for j in 0..4 {
            w!("    · exact blockLimb_{j} a h");
        }
        w!("  refine ⟨![{}],", eqs.join(", "));
        w!("    ![{}], ?_, ?_⟩", invs.join(", "));
        w!("  · intro j; rw [← hblk j]; fin_cases j");
        for j in 0..4 {
            w!("    · exact {}", fact_term(&bound, &format!("cb{i}_{j}")));
        }
        w!(
            "  · show bor (row a {i}).isDummy (band (band ({}) ({}))",
            eqs[0],
            eqs[1]
        );
        w!("      (band ({}) ({}))) = 1", eqs[2], eqs[3]);
        w!(
            "    rw [← hd, ← {}, ← {}, ← {}, ← {}, {}, kone]",
            fact_term(&bound, &format!("cba{i}")),
            fact_term(&bound, &format!("cbb{i}")),
            fact_term(&bound, &format!("cbc{i}")),
            fact_term(&bound, &format!("cbo{i}")),
            fact_term(&bound, &format!("cbk{i}"))
        );
        w!("");
        w!("theorem countAt_{i} (a : Assignment p) (h : Satisfies ({circuit} p) a)");
        w!("    (hpg : WormholeSpec.goldilocks ≤ p) :");
        w!("    (inner a {i}).numExitSlots = {s} := by");
        w!("  {obtain}");
        w!("  have hc : ({s} : ℕ) < p := lt_of_lt_of_le (by decide) hpg");
        w!(
            "  show ({}).val = {s}",
            a(inner.pis[layout::NUM_EXIT_SLOTS])
        );
        w!(
            "  rw [{}, kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]",
            fact_term(&bound, &format!("cp{i}"))
        );
        w!("");
    }
    let mem_head = "  intro t ht\n  rw [show rows a = List.ofFn (fun i => (row a i, inner a i)) from rfl] at ht\n  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ht";
    let cases = |name: &str, ty: &str, extra: &str, arm: &dyn Fn(usize) -> String| {
        let mut lines = vec![
            format!("theorem {name} (a : Assignment p) (h : Satisfies ({circuit} p) a){extra} :"),
            format!("    {ty} := by"),
            mem_head.to_string(),
            "  fin_cases i".to_string(),
        ];
        for i in 0..n {
            lines.push(format!("  · {}", arm(i)));
        }
        lines.join("\n")
    };
    w!(
        "{}",
        cases(
            "dummyAll",
            "∀ t ∈ rows a, DummyCheck t.1",
            "",
            &|i| format!("exact dummyAt_{i} a h")
        )
    );
    w!("");
    w!(
        "{}",
        cases(
            "assetAll",
            "∀ t ∈ rows a, ConsistencyCheck t.1.isDummy t.1.assetId (scanRef (rows a) fun t => t.1.assetId)",
            "",
            &|i| format!("exact assetAt_{i} a h")
        )
    );
    w!("");
    w!(
        "{}",
        cases(
            "feeAll",
            "∀ t ∈ rows a, ConsistencyCheck t.1.isDummy t.1.fee (scanRef (rows a) fun t => t.1.fee)",
            "",
            &|i| format!("exact feeAt_{i} a h")
        )
    );
    w!("");
    w!(
        "{}",
        cases(
            "blockAll",
            "∀ t ∈ rows a, DigestConsistencyCheck t.1.isDummy t.1.blockHash (blockRef (rows a))",
            "",
            &|i| format!("exact blockAt_{i} a h")
        )
    );
    w!("");
    w!(
        "{}",
        cases(
            "countAll",
            &format!("∀ t ∈ rows a, t.2.numExitSlots = {s}"),
            " (hpg : WormholeSpec.goldilocks ≤ p)",
            &|i| format!("exact countAt_{i} a h hpg")
        )
    );
    w!("");
    w!("theorem decBlock (a : Assignment p) :");
    w!("    ∀ t ∈ rows a, valDigest t.1.blockHash = t.2.blockHash := by");
    w!("{mem_head}");
    w!("  rfl");
    w!("");
    w!("theorem decSlots (a : Assignment p) :");
    w!("    ∀ t ∈ rows a, t.1.slots.map valSlot = t.2.exitSlots := by");
    w!("{mem_head}");
    w!("  rfl");
    w!("");
    w!("theorem decNulls (a : Assignment p) :");
    w!("    ∀ t ∈ rows a, t.1.nulls.map valDigest = t.2.nullifiers := by");
    w!("{mem_head}");
    w!("  rfl");
    w!("");
    for (name, ty) in [
        (
            "decNum",
            "∀ t ∈ rows a, t.1.blockNumber.val = t.2.blockNumber",
        ),
        ("decAsset", "∀ t ∈ rows a, t.1.assetId.val = t.2.assetId"),
        ("decFee", "∀ t ∈ rows a, t.1.fee.val = t.2.volumeFeeBps"),
        (
            "shapeAll",
            &format!("∀ t ∈ rows a, t.1.slots.length = {s}") as &str,
        ),
    ] {
        w!("theorem {name} (a : Assignment p) :");
        w!("    {ty} := by");
        w!("{mem_head}");
        w!("  fin_cases i <;> rfl");
        w!("");
    }
    let exit_eqs: Vec<String> = (0..n).map(|i| format!("exitChunk_eq_{i} a h")).collect();
    let null_eqs: Vec<String> = (0..n).map(|i| format!("nullChunk_eq_{i} a h")).collect();
    w!("theorem exitFwd (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    w!("    (out a).exitSlots = (forwardedSlotsF (rows a)).map valSlot := by");
    w!("  simp only [out, forwardedSlotsF, rows, List.map_cons, List.map_nil,");
    w!("    List.flatten_cons, List.flatten_nil]");
    w!("  simp only [{}]", exit_eqs.join(", "));
    w!("  rfl");
    w!("");
    w!("theorem nullFwd (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    w!("    (out a).nullifiers = (forwardedNullsF (rows a)).map valDigest := by");
    w!("  simp only [out, forwardedNullsF, rows, List.map_cons, List.map_nil,");
    w!("    List.flatten_cons, List.flatten_nil]");
    w!("  simp only [{}]", null_eqs.join(", "));
    w!("  rfl");
    w!("");
    if shape.calls.len() > 1000 {
        w!("set_option maxHeartbeats 20000000 in");
    }
    w!("/-- **The recorded public-batch wrapper satisfies `RPublicBatch`.** Every satisfying");
    w!("    assignment decodes to an `RPublicBatch` instance on the {n} inner outputs, the");
    w!("    aggregator address and the aggregated output; the oracle is irrelevant (the wrapper");
    w!("    hashes nothing). -/");
    w!("theorem sound (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    w!("    RPublicBatch ro (inners a) (addr a) (out a) := by");
    w!("  {obtain}");
    w!(
        "  have hblk : ∀ j, a (![{}] j) = blockRef (rows a) j := by",
        bw.join(", ")
    );
    w!("    intro j");
    w!("    fin_cases j");
    for j in 0..4 {
        w!("    · exact blockLimb_{j} a h");
    }
    for j in 0..4 {
        w!(
            "  have hb{j} : a ({}) = blockRef (rows a) {j} := hblk {j}",
            bw[j]
        );
    }
    w!("  have hsnR := scan_sn a h");
    w!("  have hsaR := scan_sa a h");
    w!("  have hsfR := scan_sf a h");
    let number_wire = a(shape.out(inners[n - 1].sel_number));
    let asset_wire = a(shape.out(inners[n - 1].sel_asset));
    let fee_wire = a(shape.out(inners[n - 1].sel_fee));
    w!("  refine public_batch_val ro (rows a) (k := {s}) rfl");
    w!("    (dummyAll a h) (decBlock a) (decNum a) (decAsset a) (decFee a)");
    w!("    (decSlots a) (decNulls a) ?_ ?_ ?_ ?_");
    w!("    (assetAll a h) (feeAll a h) (blockAll a h)");
    w!("    (exitFwd a h) (nullFwd a h) (shapeAll a) (countAll a h hpg) ?_");
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
    let total_value = n * s;
    w!("  · show ({}).val = {n} * {s}", a(shape.total));
    w!("    rw [ktot]");
    w!("    have hc : ({total_value} : ℕ) < p := lt_of_lt_of_le (by decide) hpg");
    w!("    rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]");
    w!("");
    w!("/-! ### The wrapper as a recursion tree (PLAN.md Step 10) -/");
    w!("");
    w!("/-- The wrapper with its recursion gadgets: `verify_proof` under the `{leaves}`-leaf private-batch");
    w!("    verifier key on each `inner_pis_i`. -/");
    w!("noncomputable def tree : Recursive p :=");
    w!("  .node \"public_batch_wrapper_n{n}\" ({circuit} p)");
    let children: Vec<String> = (0..n)
        .map(|i| format!("(Wrapper{leaves}.tree, List.ofFn (innerPis {i}))"))
        .collect();
    w!("    [{}]", children.join(",\n     "));
    w!("");
    for i in 0..n {
        w!("theorem pisEq_{i} :");
        w!("    List.ofFn (innerPis {i}) = ({circuit}.verifiers[{i}]'(by decide)).2 := by");
        w!("  show List.ofFn {circuit}.inner_pis_{i} = _");
        w!("  simp only [{circuit}.inner_pis_{i}, List.ofFn_vecCons, List.ofFn_zero]");
        w!("  rfl");
        w!("");
    }
    w!("/-- The tree's gadgets are the recorded ones. -/");
    w!("theorem tree_verifiers : (tree (p := p)).verifiers = {circuit}.verifiers := by");
    w!("  have htag : Wrapper{leaves}.tree.tag (p := p) = \"private_batch_wrapper_n{leaves}\" := by");
    w!("    simp only [Recursive.tag, Wrapper{leaves}.tree]");
    w!("  rw [Recursive.verifiers, tree, Recursive.children]");
    for i in 0..n {
        w!("  rw [List.map_cons, htag, pisEq_{i}]");
    }
    w!("  rw [List.map_nil]");
    w!("  rfl");
    w!("");
    w!("omit [Fact p.Prime] in");
    w!("set_option maxRecDepth 8192 in");
    w!("/-- Inner `i`'s public inputs, as the wrapper reads them, are the private-batch wrapper's");
    w!("    `outOf`. -/");
    w!("theorem inner_eq (a : Assignment p) (i : Fin {n}) :");
    w!("    inner a i = Wrapper{leaves}.outOf (a ∘ innerPis i) := by");
    w!("  fin_cases i <;> rfl");
    w!("");
    w!("/-- **The public-batch capstone on the recorded wiring.** `public_batch_end_to_end` with its");
    w!("    decode hypotheses discharged by `sound`: a satisfying assignment of the `n_inner = {n}`");
    w!("    public-batch wrapper whose recursion gadgets accepted every inner private-batch proof");
    w!("    (i) satisfies `RPublicBatch`, (ii) has the slot-count header equal to the sum of the");
    w!("    inners' slot counts and (iii) attests every inner's `RPrivateBatch` — through");
    w!("    `proof_sound` on the good events `hgw` for the inner wrapper's proofs and `hgl` for the");
    w!("    leaf proofs, and the wired wrapper bridges below it. -/");
    w!("theorem end_to_end_wired (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (hgw : GoodEvents perm Wrapper{leaves}.tree) (hgl : GoodEvents perm LeafCircuit.tree)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hacc : ∀ i : Fin {n}, ProofAccepted perm Wrapper{leaves}.tree ((List.ofFn (innerPis i)).map a)) :");
    w!("    RPublicBatch (spongeRO perm) (inners a) (addr a) (out a)");
    w!("      ∧ (out a).totalExitSlots = ((inners a).map fun o => o.exitSlots.length).sum");
    w!("      ∧ ∀ o ∈ inners a, ∃ leaves us, RPrivateBatch (spongeRO perm) leaves us o := by");
    w!("  have hR := sound (spongeRO perm) hpg a h");
    w!("  refine ⟨hR, RPublicBatch_totalExitSlots hR, ?_⟩");
    w!("  intro o ho");
    w!("  rw [show inners a = List.ofFn (inner a) from rfl] at ho");
    w!("  obtain ⟨i, rfl⟩ := List.mem_ofFn.mp ho");
    w!("  rw [inner_eq]");
    w!("  exact Wrapper{leaves}.accepted_sound perm hpg hgw hgl (hacc i)");
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

/// One generated bridge file, relative to `formal/Plonky2Bridge/Generated/`.
pub struct BridgeModule {
    pub relative: String,
    pub contents: String,
}

/// The bridge for `n`, split into separately elaborated files once `n` is large
/// enough that one `lean` process cannot hold `row`, the forwarded chunks and
/// every per-inner rewrite at once. Small `n` stays a single file.
pub fn generate_public_wrapper_bridge_modules(n: usize) -> Result<Vec<BridgeModule>, String> {
    let src = generate_public_wrapper_bridge_lean(n)?;
    if n <= 4 {
        return Ok(vec![BridgeModule {
            relative: format!("PublicWrapper{n}.lean"),
            contents: src,
        }]);
    }
    Ok(split_bridge(n, &src))
}

/// Inners per forwarding file. Each inner is two theorems (slots and nullifiers);
/// one inner keeps a single elaboration well under the memory a CI runner gives
/// four parallel `lean` processes.
const INNERS_PER_FWD_FILE: usize = 1;

fn split_bridge(n: usize, src: &str) -> Vec<BridgeModule> {
    let module = format!("PublicWrapper{n}");
    let ns = format!("Plonky2Bridge.PublicWrapper{n}");
    let trimmed = src
        .trim_end()
        .trim_end_matches(&format!("end {ns}"))
        .trim_end();
    let lines: Vec<&str> = trimmed.lines().collect();
    let mut idx = 0;
    let mut header = String::new();
    if lines.first().is_some_and(|l| l.starts_with("/-")) {
        while idx < lines.len() {
            header.push_str(lines[idx]);
            header.push('\n');
            if lines[idx].starts_with("-/") {
                idx += 1;
                break;
            }
            idx += 1;
        }
    }
    while idx < lines.len() && lines[idx].is_empty() {
        idx += 1;
    }
    let mut imports = Vec::new();
    while idx < lines.len() && lines[idx].starts_with("import ") {
        imports.push(lines[idx].to_string());
        idx += 1;
    }
    while idx < lines.len() && lines[idx].is_empty() {
        idx += 1;
    }
    assert!(
        lines[idx].starts_with("namespace "),
        "bridge is missing its namespace"
    );
    idx += 1;
    while idx < lines.len() && lines[idx].is_empty() {
        idx += 1;
    }
    let mut opens = Vec::new();
    while idx < lines.len() && !lines[idx].starts_with("variable ") {
        if !lines[idx].is_empty() {
            opens.push(lines[idx].to_string());
        }
        idx += 1;
    }
    let variable = lines[idx].to_string();
    assert!(
        variable.starts_with("variable "),
        "bridge is missing its variable line"
    );
    idx += 1;
    while idx < lines.len() && lines[idx].is_empty() {
        idx += 1;
    }
    let mut options = Vec::new();
    while idx < lines.len() && lines[idx].starts_with("set_option ") {
        options.push(lines[idx].to_string());
        idx += 1;
    }
    let blocks = decl_blocks(&lines[idx..].join("\n"));
    let mut read: Vec<String> = Vec::new();
    let mut scan: Vec<String> = Vec::new();
    let mut take: Vec<String> = Vec::new();
    let mut fwd: Vec<Vec<String>> = vec![Vec::new(); n];
    let mut limb: Vec<Vec<String>> = vec![Vec::new(); 4];
    let mut scalar: Vec<String> = Vec::new();
    let mut rel: Vec<String> = Vec::new();
    let mut dispatch: Vec<String> = Vec::new();
    let mut fwd_join: Vec<String> = Vec::new();
    let mut tree: Vec<String> = Vec::new();
    let mut sound: Vec<String> = Vec::new();
    let mut seen_fwd = false;
    let mut pending = String::new();
    for mut block in blocks {
        if is_directive_only(&block) {
            pending.push_str(&block);
            continue;
        }
        if !pending.is_empty() {
            pending.push_str(&block);
            block = std::mem::take(&mut pending);
        }
        if let Some(i) = chunk_eq_index(&block) {
            seen_fwd = true;
            fwd[i].push(block);
        } else if let Some(j) = block_limb_index(&block) {
            limb[j].push(block);
        } else if is_scan_scalar(&block) {
            scalar.push(block);
        } else if is_rel_block(&block) {
            rel.push(block);
        } else if is_dispatch_block(&block) {
            dispatch.push(block);
        } else if is_tree_block(&block) {
            tree.push(block);
        } else if is_fwd_join(&block) {
            fwd_join.push(block);
        } else if is_take_block(&block) {
            take.push(block);
        } else if block_starts_with(&block, "theorem scanRef_") {
            scan.push(block);
        } else if seen_fwd {
            sound.push(block);
        } else {
            read.push(block);
        }
    }
    assert!(seen_fwd, "no per-inner forwarding theorems to split");

    let part = |name: &str,
                use_imports: bool,
                deps: &[String],
                body: &str,
                with_header: bool|
     -> BridgeModule {
        BridgeModule {
            relative: format!("{module}/{name}.lean"),
            contents: render_bridge_part(
                &ns,
                if with_header { &header } else { "" },
                if use_imports { &imports } else { &[] },
                deps,
                &opens,
                &variable,
                &options,
                body,
            ),
        }
    };
    let mut files = Vec::new();
    files.push(part("Read", true, &[], &read.join("\n"), true));
    files.push(part(
        "Scan",
        false,
        &[format!("Plonky2Bridge.Generated.{module}.Read")],
        &scan.join("\n"),
        false,
    ));
    files.push(part(
        "Take",
        false,
        &[format!("Plonky2Bridge.Generated.{module}.Read")],
        &take.join("\n"),
        false,
    ));
    let mut limb_deps = Vec::new();
    for (j, blocks) in limb.iter().enumerate() {
        let name = format!("Limb{j}");
        files.push(part(
            &name,
            false,
            &[
                format!("Plonky2Bridge.Generated.{module}.Scan"),
                format!("Plonky2Bridge.Generated.{module}.Take"),
            ],
            &blocks.join("\n"),
            false,
        ));
        limb_deps.push(format!("Plonky2Bridge.Generated.{module}.{name}"));
    }
    files.push(part(
        "Scalar",
        false,
        &[
            format!("Plonky2Bridge.Generated.{module}.Scan"),
            format!("Plonky2Bridge.Generated.{module}.Take"),
        ],
        &scalar.join("\n"),
        false,
    ));
    limb_deps.push(format!("Plonky2Bridge.Generated.{module}.Scalar"));
    files.push(part("Rel", false, &limb_deps, &rel.join("\n"), false));
    files.push(part(
        "Dispatch",
        false,
        &[format!("Plonky2Bridge.Generated.{module}.Rel")],
        &dispatch.join("\n"),
        false,
    ));
    files.push(part(
        "Tree",
        false,
        &[format!("Plonky2Bridge.Generated.{module}.Read")],
        &tree.join("\n"),
        false,
    ));
    limb_deps.push(format!("Plonky2Bridge.Generated.{module}.Tree"));
    files.push(part(
        "FwdJoin",
        false,
        &[format!("Plonky2Bridge.Generated.{module}.Fwd")],
        &fwd_join.join("\n"),
        false,
    ));
    limb_deps.push(format!("Plonky2Bridge.Generated.{module}.Dispatch"));
    limb_deps.push(format!("Plonky2Bridge.Generated.{module}.FwdJoin"));
    let mut fwd_names = Vec::new();
    for (batch, group) in fwd.chunks(INNERS_PER_FWD_FILE).enumerate() {
        let body = group
            .iter()
            .flat_map(|blocks| blocks.iter())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if body.trim().is_empty() {
            continue;
        }
        let name = format!("Fwd{batch}");
        files.push(part(
            &name,
            false,
            &[format!("Plonky2Bridge.Generated.{module}.Read")],
            &body,
            false,
        ));
        fwd_names.push(name);
    }
    let barrel = fwd_names
        .iter()
        .map(|name| format!("import Plonky2Bridge.Generated.{module}.{name}\n"))
        .collect::<String>();
    files.push(BridgeModule {
        relative: format!("{module}/Fwd.lean"),
        contents: barrel,
    });
    files.push(part(
        "Sound",
        false,
        &{
            let mut deps = limb_deps.clone();
            deps.push(format!("Plonky2Bridge.Generated.{module}.Fwd"));
            deps
        },
        &sound.join("\n"),
        false,
    ));
    files.push(BridgeModule {
        relative: format!("{module}.lean"),
        contents: format!("import Plonky2Bridge.Generated.{module}.Sound\n"),
    });
    files
}

fn chunk_eq_index(block: &str) -> Option<usize> {
    for line in block.lines() {
        let Some(rest) = line
            .strip_prefix("theorem exitChunk_eq_")
            .or_else(|| line.strip_prefix("theorem nullChunk_eq_"))
        else {
            continue;
        };
        let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !num.is_empty() && rest[num.len()..].starts_with(' ') {
            return num.parse().ok();
        }
    }
    None
}

fn block_limb_index(block: &str) -> Option<usize> {
    for line in block.lines() {
        let Some(rest) = line.strip_prefix("theorem blockLimb_") else {
            continue;
        };
        let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !num.is_empty() && rest[num.len()..].starts_with(' ') {
            return num.parse().ok();
        }
    }
    None
}

fn is_rel_block(block: &str) -> bool {
    block.lines().any(|line| {
        line.starts_with("theorem dummyAt_")
            || line.starts_with("theorem assetAt_")
            || line.starts_with("theorem feeAt_")
            || line.starts_with("theorem blockAt_")
            || line.starts_with("theorem countAt_")
    })
}

fn is_dispatch_block(block: &str) -> bool {
    block.lines().any(|line| {
        line.starts_with("theorem dummyAll")
            || line.starts_with("theorem assetAll")
            || line.starts_with("theorem feeAll")
            || line.starts_with("theorem blockAll")
            || line.starts_with("theorem countAll")
            || line.starts_with("theorem shapeAll")
            || line.starts_with("theorem decBlock")
            || line.starts_with("theorem decNum")
            || line.starts_with("theorem decAsset")
            || line.starts_with("theorem decFee")
            || line.starts_with("theorem decSlots")
            || line.starts_with("theorem decNulls")
    })
}

fn is_directive_only(block: &str) -> bool {
    block.lines().all(|line| {
        let line = line.trim();
        line.is_empty()
            || line.starts_with("omit ")
            || line.starts_with("set_option ")
            || line.starts_with("/-")
            || line.starts_with("-/")
    })
}

fn is_tree_block(block: &str) -> bool {
    block.lines().any(|line| {
        (line.starts_with("def tree ") || line.starts_with("noncomputable def tree "))
            || line.starts_with("theorem pisEq_")
            || line.starts_with("theorem tree_verifiers")
            || line.starts_with("theorem inner_eq")
    })
}

fn is_fwd_join(block: &str) -> bool {
    block
        .lines()
        .any(|line| line.starts_with("theorem exitFwd") || line.starts_with("theorem nullFwd"))
}

fn is_scan_scalar(block: &str) -> bool {
    block.lines().any(|line| line.starts_with("theorem scan_"))
}

fn is_take_block(block: &str) -> bool {
    block.lines().any(|line| {
        (line.starts_with("def takeWire") || line.starts_with("noncomputable def takeWire"))
            || line.starts_with("theorem realEq_")
            || line.starts_with("theorem foundEq_")
            || line.starts_with("theorem takeEq_")
            || line.starts_with("theorem selChain_eq")
    })
}

fn block_starts_with(block: &str, prefix: &str) -> bool {
    block
        .lines()
        .any(|line| line.starts_with(prefix) || line.trim_start().starts_with(prefix))
}

fn decl_blocks(src: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cur = String::new();
    let mut has_decl = false;
    for line in src.lines() {
        let starts = line.starts_with("def ")
            || line.starts_with("noncomputable def ")
            || line.starts_with("theorem ")
            || line.starts_with("omit ")
            || line.starts_with("set_option ")
            || line.starts_with("/--")
            || line.starts_with("/-!");
        if starts && has_decl {
            blocks.push(std::mem::take(&mut cur));
            has_decl = false;
        }
        if line.starts_with("def ")
            || line.starts_with("noncomputable def ")
            || line.starts_with("theorem ")
            || line.starts_with("omit ")
            || line.starts_with("set_option ")
        {
            has_decl = true;
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        blocks.push(cur);
    }
    blocks
}

fn render_bridge_part(
    ns: &str,
    header: &str,
    imports: &[String],
    deps: &[String],
    opens: &[String],
    variable: &str,
    options: &[String],
    body: &str,
) -> String {
    let mut out = String::new();
    if !header.is_empty() {
        out.push_str(header);
        if !header.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }
    for import in imports {
        out.push_str(import);
        out.push('\n');
    }
    for dep in deps {
        out.push_str("import ");
        out.push_str(dep);
        out.push('\n');
    }
    out.push('\n');
    out.push_str("namespace ");
    out.push_str(ns);
    out.push_str("\n\n");
    for open in opens {
        out.push_str(open);
        out.push('\n');
    }
    out.push('\n');
    out.push_str(variable);
    out.push_str("\n\n");
    for option in options {
        out.push_str(option);
        out.push('\n');
    }
    if !options.is_empty() {
        out.push('\n');
    }
    out.push_str(body.trim_start_matches('\n'));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("end ");
    out.push_str(ns);
    out.push('\n');
    out
}
