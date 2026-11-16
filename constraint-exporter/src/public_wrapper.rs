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
use crate::gadget::{rec_depth_option, Call, Fact};
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
        check_verifiers(t, &format!("private_batch_wrapper_n{leaves}"), &pis)?;
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

    /// The per-call decode lemma of fact `k`, applied to the wrapper hypotheses.
    fn projection(&self, k: usize) -> String {
        format!("publicBatchWrapper{}_f{k} a h", self.n)
    }
}

fn a(t: Target) -> String {
    format!("a ({})", lean_target(t))
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
    w!("  inner's `RPrivateBatch` recovered from its accepted proof through the one trusted axiom");
    w!("  `proof_sound` and the private-batch wrapper bridge (Step 10).");
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
    let rec_depth = rec_depth_option(n * (layout::EXIT_SLOT_LEN * s + 4 * u));
    if !rec_depth.is_empty() {
        w!("{rec_depth}");
    }
    w!("/-! ### Reading the spec objects off the wiring -/");
    w!("");
    w!("/-- The {pi_len} public inputs of inner proof `i` (`inner_pis_i`), at the private-batch");
    w!("    `aggregated_output` layout. -/");
    w!("def innerPis : Fin {n} → Fin {pi_len} → Target");
    for i in 0..n {
        w!("  | ⟨{i}, _⟩ => {circuit}.inner_pis_{i}");
    }
    w!("  | ⟨_ + {n}, h⟩ => absurd h (by omega)");
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
        w!("  | ⟨{i}, _⟩ =>");
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
    w!("  | ⟨_ + {n}, h⟩ => absurd h (by omega)");
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
    w!("    {} :=", cs.join(" ∧ "));
    w!("  Plonky2Spec.Generated.{circuit}_consts a h");
    w!("");
    // `obtain` on `consts`, naming the constants in `used` and clearing the rest.
    let obtain_consts = |used: &[&str]| -> String {
        let names: Vec<&str> = shape
            .constants
            .iter()
            .map(|(t, _)| {
                let name = if *t == shape.one {
                    "kone"
                } else if *t == shape.zero {
                    "kzero"
                } else {
                    "ktot"
                };
                if used.contains(&name) {
                    name
                } else {
                    "-"
                }
            })
            .collect();
        format!("  obtain ⟨{}⟩ := consts a h", names.join(", "))
    };
    // `have name := <decode lemma of call k>`, a dummy check's `zero` operand rewritten to `0`.
    let have = |name: &str, k: usize| -> String {
        let mut s = format!("  have {name} := {}", shape.projection(k));
        if matches!(*shape.fact(k), Fact::IsEqual { y, .. } if y == shape.zero) {
            let _ = write!(s, "\n  rw [kzero] at {name}");
        }
        s
    };
    let hyps = format!("(a : Assignment p) (h : Satisfies ({circuit} p) a)");
    let d = |i: usize| a(shape.out(inners[i].dummy_and[2]));

    w!("/-! One lemma per inner and per reference, each over its own recorded calls, so their");
    w!("    cost does not grow with the wrapper and Lean checks them in parallel. -/");
    w!("");
    for (i, inner) in inners.iter().enumerate() {
        w!("/-- Inner {i}: the `bytes_digest_eq(block_hash, 0)` constraints and the flag they output. -/");
        w!("theorem dummy{i} {hyps} :");
        w!(
            "    DummyCheck (row a {i}) ∧ {} = (row a {i}).isDummy := by",
            d(i)
        );
        w!("{}", obtain_consts(&["kzero"]));
        for j in 0..4 {
            w!("{}", have(&format!("e{j}"), inner.dummy_eq[j]));
        }
        w!("{}", have("da", inner.dummy_and[0]));
        w!("{}", have("db", inner.dummy_and[1]));
        w!("{}", have("d", inner.dummy_and[2]));
        w!("  constructor");
        w!("  · intro j; fin_cases j; exacts [e0, e1, e2, e3]");
        w!("  · rw [d, da, db]; rfl");
        w!("");
    }

    // The select chain on the circuit's take wires.
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
    w!("/-- The circuit's `take` chain is the first-real scan over the rows' dummy flags. -/");
    w!("theorem scan {hyps} :");
    w!("    ∀ {} : ZMod p,", xs.join(" "));
    w!("      {chain}");
    w!("        = {nf_rows} := by");
    for (i, inner) in inners.iter().enumerate() {
        w!("{}", have(&format!("r{i}"), inner.is_real));
        for (name, call) in [
            ("nf", inner.not_found),
            ("tk", inner.take),
            ("fo", inner.found),
        ] {
            if let Some(k) = call {
                w!("{}", have(&format!("{name}{i}"), k));
            }
        }
        w!("  have hd{i} := (dummy{i} a h).2");
    }
    w!("  intro {}", xs.join(" "));
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
    w!("  rw [{}]", rws.join(", "));
    w!("");

    let block_wires: Vec<Target> = (0..4)
        .map(|j| shape.out(inners[n - 1].sel_block[j]))
        .collect();
    let bw: Vec<String> = block_wires.iter().map(|t| lean_target(*t)).collect();
    w!("/-- The block-hash reference wires are the scan's result. -/");
    w!("theorem ref_block {hyps} :");
    w!(
        "    ∀ j, a (![{}] j) = blockRef (rows a) j := by",
        bw.join(", ")
    );
    w!("{}", obtain_consts(&["kzero"]));
    for (i, inner) in inners.iter().enumerate() {
        for j in 0..4 {
            w!("{}", have(&format!("sb{i}_{j}"), inner.sel_block[j]));
        }
    }
    w!("  intro j");
    w!("  simp only [blockRef, rows, scanRef_{n}]");
    w!("  fin_cases j");
    for (j, wire) in bw.iter().enumerate() {
        let sels: Vec<String> = (0..n).rev().map(|i| format!("sb{i}_{j}")).collect();
        w!("  · show a ({wire}) = _");
        w!("    rw [{}, kzero, scan a h]; rfl", sels.join(", "));
    }
    w!("");
    let sels_of = |sel: fn(&Inner) -> usize| -> Vec<usize> { inners.iter().map(sel).collect() };
    for (name, field, calls) in [
        ("number", "blockNumber", sels_of(|inner| inner.sel_number)),
        ("asset", "assetId", sels_of(|inner| inner.sel_asset)),
        ("fee", "fee", sels_of(|inner| inner.sel_fee)),
    ] {
        w!("/-- The `{field}` reference wire is the scan's result. -/");
        w!("theorem ref_{name} {hyps} :");
        w!(
            "    {} = scanRef (rows a) fun t => t.1.{field} := by",
            a(shape.out(calls[n - 1]))
        );
        w!("{}", obtain_consts(&["kzero"]));
        for (i, &k) in calls.iter().enumerate() {
            w!("{}", have(&format!("s{i}"), k));
        }
        let sels: Vec<String> = (0..n).rev().map(|i| format!("s{i}")).collect();
        w!(
            "  rw [rows, scanRef_{n}, {}, kzero, scan a h]; rfl",
            sels.join(", ")
        );
        w!("");
    }
    let number_wire = a(shape.out(inners[n - 1].sel_number));
    let asset_wire = a(shape.out(inners[n - 1].sel_asset));
    let fee_wire = a(shape.out(inners[n - 1].sel_fee));

    for (i, inner) in inners.iter().enumerate() {
        let (eqs, invs): (Vec<String>, Vec<String>) = inner
            .block_eq
            .iter()
            .map(|&k| match *shape.fact(k) {
                Fact::IsEqual { equal, inv, .. } => (a(equal), a(inv)),
                _ => unreachable!(),
            })
            .unzip();
        w!("/-- Inner {i}: its asset, fee and block hash are the references unless it is a dummy. -/");
        w!("theorem consistent{i} {hyps} :");
        w!("    ConsistencyCheck (row a {i}).isDummy (row a {i}).assetId (scanRef (rows a) fun t => t.1.assetId)");
        w!("      ∧ ConsistencyCheck (row a {i}).isDummy (row a {i}).fee (scanRef (rows a) fun t => t.1.fee)");
        w!("      ∧ DigestConsistencyCheck (row a {i}).isDummy (row a {i}).blockHash (blockRef (rows a)) := by");
        w!("{}", obtain_consts(&["kone"]));
        w!("  have hd := (dummy{i} a h).2");
        w!("  have hsaR := ref_asset a h");
        w!("  have hsfR := ref_fee a h");
        for (name, k) in [
            ("ca", inner.asset_eq),
            ("cao", inner.asset_or),
            ("cak", inner.asset_pin),
            ("cf", inner.fee_eq),
            ("cfo", inner.fee_or),
            ("cfk", inner.fee_pin),
            ("cb0", inner.block_eq[0]),
            ("cb1", inner.block_eq[1]),
            ("cb2", inner.block_eq[2]),
            ("cb3", inner.block_eq[3]),
            ("cba", inner.block_and[0]),
            ("cbb", inner.block_and[1]),
            ("cbc", inner.block_and[2]),
            ("cbo", inner.block_or),
            ("cbk", inner.block_pin),
        ] {
            w!("{}", have(name, k));
        }
        w!("  refine ⟨⟨_, _, hsaR ▸ ca, by rw [← hd, ← cao, cak, kone]⟩,");
        w!("    ⟨_, _, hsfR ▸ cf, by rw [← hd, ← cfo, cfk, kone]⟩,");
        w!("    ![{}],", eqs.join(", "));
        w!("    ![{}], ?_, ?_⟩", invs.join(", "));
        w!("  · intro j; rw [← ref_block a h j]; fin_cases j; exacts [cb0, cb1, cb2, cb3]");
        w!(
            "  · show bor (row a {i}).isDummy (band (band ({}) ({}))",
            eqs[0],
            eqs[1]
        );
        w!("        (band ({}) ({}))) = 1", eqs[2], eqs[3]);
        w!("    rw [← hd, ← cba, ← cbb, ← cbc, ← cbo, cbk, kone]");
        w!("");
    }

    // The forwarded regions, one literal list per inner.
    let regions = |items: &[String], per_inner: usize| -> Vec<String> {
        items
            .chunks(per_inner)
            .map(|c| format!("[{}]", c.join(",\n      ")))
            .collect()
    };
    let slot_regions = regions(&slot_list, s);
    let null_regions = regions(&null_list, u);
    // Per forwarded region: lemma name, element type, the literal regions, the calls, the
    // masked field region of a row and its `.val` decode.
    type Forward<'a> = (
        &'a str,
        &'a str,
        &'a [String],
        fn(&Inner) -> &Vec<usize>,
        fn(&str) -> String,
        &'a str,
    );
    let forwards: [Forward; 2] = [
        (
            "slots",
            "ExitSlot",
            &slot_regions,
            |inner| &inner.slot_fwd,
            |r| format!("{r}.slots.map (maskSlot {r}.isDummy)"),
            "valSlot",
        ),
        (
            "nulls",
            "Digest",
            &null_regions,
            |inner| &inner.null_fwd,
            |r| format!("{r}.nulls.map fun x => fun j => bselect {r}.isDummy 0 (x j)"),
            "valDigest",
        ),
    ];
    for (name, ty, region, fwd, masked, val) in &forwards {
        for (i, inner) in inners.iter().enumerate() {
            let calls = fwd(inner);
            let fs: Vec<String> = (0..calls.len()).map(|k| format!("f{k}")).collect();
            w!("/-- Inner {i}: its forwarded `{name}` are its own, masked by its dummy flag. -/");
            w!("theorem {name}{i} {hyps} :");
            w!("    ({} : List {ty})", region[i]);
            w!(
                "      = ({}).map {val} := by",
                masked(&format!("(row a {i})"))
            );
            w!("{}", obtain_consts(&["kzero"]));
            w!("  have hd := (dummy{i} a h).2");
            for (f, &k) in fs.iter().zip(calls) {
                w!("{}", have(f, k));
            }
            w!("  rw [{}, kzero, hd]", fs.join(", "));
            w!("  rfl");
            w!("");
        }
    }

    w!("/-- **The recorded public-batch wrapper satisfies `RPublicBatch`.** Every satisfying");
    w!("    assignment decodes to an `RPublicBatch` instance on the {n} inner outputs, the");
    w!("    aggregator address and the aggregated output; the oracle is irrelevant (the wrapper");
    w!("    hashes nothing). -/");
    w!("theorem sound (ro : RandomOracle) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a) :");
    w!("    RPublicBatch ro (inners a) (addr a) (out a) := by");
    w!("{}", obtain_consts(&["ktot"]));
    for (j, wire) in bw.iter().enumerate() {
        w!("  have hb{j} : a ({wire}) = blockRef (rows a) {j} := ref_block a h {j}");
    }
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
    let per_inner = |lemma: &str, part: &str| -> String {
        (0..n)
            .map(|i| format!("({lemma}{i} a h){part}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    w!("  refine public_batch_val ro (rows a) (k := {s}) rfl ?_ (hdec _ {dec_rfls}) (hdec _ {dec_rfls})");
    w!("    (hdec _ {dec_rfls}) (hdec _ {dec_rfls}) (hdec _ {dec_rfls}) (hdec _ {dec_rfls}) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_");
    w!("    ?_ ?_");
    w!("  · -- dummy checks");
    w!("    exact hdec _ {}", per_inner("dummy", ".1"));
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
    w!("    rw [ref_number a h]");
    w!("  · show ({asset_wire}).val = _");
    w!("    rw [ref_asset a h]");
    w!("  · show ({fee_wire}).val = _");
    w!("    rw [ref_fee a h]");
    w!("  · -- asset consistency");
    w!("    exact hdec _ {}", per_inner("consistent", ".1"));
    w!("  · -- fee consistency");
    w!("    exact hdec _ {}", per_inner("consistent", ".2.1"));
    w!("  · -- block consistency");
    w!("    exact hdec _ {}", per_inner("consistent", ".2.2"));
    for (name, ty, region, _, masked, val) in &forwards {
        let mut all = "List.Forall₂.nil".to_string();
        for i in (0..n).rev() {
            all = format!("List.Forall₂.cons ({name}{i} a h) ({all})");
        }
        w!("  · -- forwarded {name}");
        w!(
            "    show ([{}] : List (List {ty})).flatten = _",
            region.join(",\n      ")
        );
        w!(
            "    exact flatten_eq_of_forall₂ (xs := rows a) (f := fun t => {}) (g := {val})",
            masked("t.1")
        );
        w!("      ({all})");
    }
    w!("  · intro t ht");
    w!("    rcases hmem t ht with {rfls} <;> rfl");
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
    w!("def tree : Recursive p :=");
    w!("  .node \"public_batch_wrapper_n{n}\" ({circuit} p)");
    let children: Vec<String> = (0..n)
        .map(|i| format!("(Wrapper{leaves}.tree, List.ofFn (innerPis {i}))"))
        .collect();
    w!("    [{}]", children.join(",\n     "));
    w!("");
    w!("omit [Fact p.Prime] in");
    if rec_depth.is_empty() {
        w!("set_option maxRecDepth 8192 in");
    } else {
        w!("set_option maxHeartbeats {} in", 4_000 * n * pi_len);
    }
    w!("/-- The tree's gadgets are the recorded ones. -/");
    w!("theorem tree_verifiers : (tree (p := p)).verifiers = {circuit}.verifiers := rfl");
    w!("");
    w!("omit [Fact p.Prime] in");
    if rec_depth.is_empty() {
        w!("set_option maxRecDepth 8192 in");
    }
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
    w!("    `proof_sound`, the only axiom, and the wired wrapper bridges below it. -/");
    w!("theorem end_to_end_wired (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)");
    w!("    (a : Assignment p) (h : Satisfies ({circuit} p) a)");
    w!("    (hacc : ∀ i : Fin {n}, ProofAccepted perm Wrapper{leaves}.tree ((List.ofFn (innerPis i)).map a)) :");
    w!("    RPublicBatch (spongeRO perm) (inners a) (addr a) (out a)");
    w!("      ∧ (out a).totalExitSlots = ((inners a).map fun o => o.exitSlots.length).sum");
    w!("      ∧ ∀ o ∈ inners a, ∃ leaves us, RPrivateBatch (spongeRO perm) leaves us o := by");
    w!("  have hR := sound (spongeRO perm) hpg a h");
    w!("  refine ⟨hR, RPublicBatch_totalExitSlots hR, ?_⟩");
    w!("  intro o ho");
    w!("  simp only [inners, List.mem_cons, List.mem_nil_iff, or_false] at ho");
    let alts: Vec<&str> = (0..n).map(|_| "rfl").collect();
    w!("  rcases ho with {}", alts.join(" | "));
    for i in 0..n {
        w!("  · rw [inner_eq]; exact Wrapper{leaves}.accepted_sound perm hpg (hacc {i})");
    }
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
