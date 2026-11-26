//! The recorded public-batch wrappers (`trace.rs`, `public_wrapper.rs`) at every size with
//! a trace: each vendored trace parses into the export + calls the generators expect, it
//! matches the `wormholeSpec` lake package when that has been fetched, `Shape::read`
//! recognises the wrapper's call structure (and rejects a perturbed one), and the checked-in
//! `Plonky2Spec/Generated/PublicBatchWrapper{N}.lean` and
//! `Plonky2Bridge/Generated/PublicWrapper{N}.lean` are what they generate. See
//! `../formal/PLAN.md` Steps 8f and 8e.

use constraint_exporter::circuit::GateKind;
use constraint_exporter::gadget::{Fact, FACT_GROUP};
use constraint_exporter::public_wrapper::{generate_public_wrapper_bridge_modules, Shape};
use constraint_exporter::trace::{
    assert_vendored_trace_matches_pinned, generate_public_batch_wrapper_modules, load, traces_dir,
};
use plonky2::field::types::PrimeField64;

mod common;

/// Recorded `(n_inner, leaves, exit slots)`. `n_inner = 86` is the production public
/// batch, over the production private batch (`8` leaves, `32` exit slots).
const SIZES: [(usize, usize, usize); 3] = [(2, 2, 2), (4, 2, 2), (86, 8, 32)];

fn inner_pi_len(leaves: usize, slots: usize) -> usize {
    8 + 5 * slots + 4 * leaves
}

fn trace_name(n: usize) -> String {
    format!("public_batch_wrapper_n{n}.json")
}

fn wrapper(n: usize) -> constraint_exporter::trace::LoadedTrace {
    load(&traces_dir().join(trace_name(n))).expect("vendored wrapper trace")
}

/// The vendored copies must be the files at the pinned `wormholeSpec` revision. The lake
/// package only exists after `lake build`/`lake update` in `formal/`, so this skips on the
/// Rust CI runners; the `formal` job runs it with `FORMAL_TRACES_STRICT=1` after
/// `lake build`, where a missing or differing file fails.
#[test]
fn vendored_trace_matches_pinned_package() {
    for (n, _, _) in SIZES {
        assert_vendored_trace_matches_pinned(&trace_name(n));
    }
}

#[test]
fn wrapper_trace_shape() {
    for (n, leaves, slots) in SIZES {
        let t = wrapper(n);
        assert_eq!(t.circuit, format!("public_batch_wrapper_n{n}"));
        // The wrapper is pure boolean/select logic: no hashes, no range checks.
        assert!(t
            .ex
            .rows
            .iter()
            .all(|r| r.0 == GateKind::Arithmetic { num_ops: 20 }));
        let nulls = leaves;
        let pi_len = inner_pi_len(leaves, slots);
        // 12 header felts + n · slots · 5 limbs + n · nulls · 4 limbs.
        assert_eq!(t.ex.public_inputs.len(), 12 + n * slots * 5 + n * nulls * 4);
        let named: Vec<(&str, usize)> =
            t.ex.named
                .iter()
                .map(|(n, v)| (n.as_str(), v.len()))
                .collect();
        let mut expected: Vec<(String, usize)> =
            (0..n).map(|i| (format!("inner_pis_{i}"), pi_len)).collect();
        expected.push(("aggregator_address".into(), 4));
        assert_eq!(
            named,
            expected
                .iter()
                .map(|(s, k)| (s.as_str(), *k))
                .collect::<Vec<_>>()
        );

        let count = |f: fn(&Fact) -> bool| t.calls.iter().filter(|c| f(&c.fact)).count();
        // Per inner: one dummy check + three consistency checks, `bytes_digest_eq`/`is_equal`.
        assert_eq!(count(|f| matches!(f, Fact::IsEqual { .. })), n * 10);
        // Per inner: 7 first-real selects + slot limbs + nullifier limbs.
        assert_eq!(
            count(|f| matches!(f, Fact::Select { .. })),
            n * (7 + slots * 5 + nulls * 4)
        );
        // Per inner: the slot-count header pin and the three `or(is_dummy, eq) = 1`
        // consistency pins.
        assert_eq!(count(|f| matches!(f, Fact::Connect { .. })), 4 * n);
        assert_eq!(count(|f| matches!(f, Fact::Poseidon2 { .. })), 0);
        assert_eq!(count(|f| matches!(f, Fact::RangeCheck { .. })), 0);
        assert_eq!(t.calls.len(), 34 * n + n * slots * 5 + n * nulls * 4);
        assert!(t.calls.len() > FACT_GROUP, "decode theorem is grouped");

        let shape = Shape::read(&t).expect("the trace has the wrapper's call structure");
        assert_eq!((shape.n, shape.leaves), (n, leaves));
        assert_eq!(
            (shape.slots_per_inner, shape.nulls_per_inner),
            (slots, nulls)
        );
        assert_eq!(shape.pi_len, pi_len);
    }
}

/// `Shape::read` indexes the calls positionally, so it must notice when the structure it
/// assumes is not what was recorded — otherwise a changed wrapper would yield a bridge that
/// talks about the wrong wires (and only fails, less legibly, in Lean).
#[test]
fn shape_rejects_perturbed_traces() {
    let t = wrapper(2);
    // Swap the slot-count pins of inners 0 and 1: each must read its own header.
    let mut swapped = wrapper(2);
    swapped.calls.swap(0, 1);
    let err = Shape::read(&swapped).unwrap_err();
    assert!(
        err.contains("call 0: expected connect(num_exit_slots_0, slots_per_inner)"),
        "{err}"
    );
    // A slot-count pin against the total rather than the per-inner constant: a same-length
    // inner of another shape would pass.
    let mut total = wrapper(2);
    let constant = |v: u64| {
        total
            .ex
            .constants
            .iter()
            .find(|(_, c)| c.to_canonical_u64() == v)
            .unwrap()
            .0
    };
    let (kslots, ktot) = (constant(2), constant(4));
    if let Fact::Connect { x, y } = total.calls[0].fact {
        assert_eq!(y, kslots);
        total.calls[0].fact = Fact::Connect { x, y: ktot };
    }
    let err = Shape::read(&total).unwrap_err();
    assert!(
        err.contains("call 0: expected connect(num_exit_slots_0, slots_per_inner)"),
        "{err}"
    );
    // Swap the first two `is_equal`s of inner 0's dummy check: the limbs no longer line up.
    let mut swapped = wrapper(2);
    swapped.calls.swap(2, 3);
    let err = Shape::read(&swapped).unwrap_err();
    assert!(err.contains("call 2: expected dummy is_equal"), "{err}");
    // Drop the last call: the count no longer matches `(n_inner, leaves)`.
    let mut short = wrapper(2);
    short.calls.pop();
    let err = Shape::read(&short).unwrap_err();
    assert!(err.contains("expected 104 gadget calls"), "{err}");
    // Reorder the public inputs: the output layout check fires.
    let mut pis = wrapper(2);
    pis.ex.public_inputs.swap(4, 5);
    let err = Shape::read(&pis).unwrap_err();
    assert!(err.contains("recorded public inputs"), "{err}");
    // A consistency `connect` pinned to something other than `one` (the first `connect`
    // after the two slot-count pins).
    let mut pin = wrapper(2);
    let k = t
        .calls
        .iter()
        .enumerate()
        .position(|(i, c)| i >= 2 && matches!(c.fact, Fact::Connect { .. }))
        .unwrap();
    if let Fact::Connect { x, .. } = pin.calls[k].fact {
        pin.calls[k].fact = Fact::Connect { x, y: x };
    }
    let err = Shape::read(&pin).unwrap_err();
    assert!(err.contains("expected connect(asset_ok, one)"), "{err}");
    // The recursion gadgets: one `verify_proof` per inner, under the private-batch key, on
    // `inner_pis_i`.
    let mut extra = wrapper(2);
    let dup = extra.ex.verifiers[0].clone();
    extra.ex.verifiers.push(dup);
    let err = Shape::read(&extra).unwrap_err();
    assert!(
        err.contains("expected 2 verify_proof gadgets, trace has 3"),
        "{err}"
    );
    // The first verifier names the inner shape; the others must agree with it.
    let mut key = wrapper(2);
    key.ex.verifiers[0].0 = "leaf_circuit".into();
    let err = Shape::read(&key).unwrap_err();
    assert!(
        err.contains("verifier child \"leaf_circuit\" is not a private-batch wrapper"),
        "{err}"
    );
    let mut key = wrapper(2);
    key.ex.verifiers[1].0 = "private_batch_wrapper_n4".into();
    let err = Shape::read(&key).unwrap_err();
    assert!(
        err.contains(
            "verifier 1 is for \"private_batch_wrapper_n4\", expected \"private_batch_wrapper_n2\""
        ),
        "{err}"
    );
    let mut targets = wrapper(2);
    let t0 = targets.ex.verifiers[0].1.clone();
    targets.ex.verifiers[1].1 = t0;
    let err = Shape::read(&targets).unwrap_err();
    assert!(
        err.contains("verifier 1 public inputs are not the slot's pis_1 targets"),
        "{err}"
    );
}

#[test]
fn wrapper_lean_is_current() {
    for (n, leaves, slots) in SIZES {
        let generated_modules = generate_public_batch_wrapper_modules(n).unwrap();
        let generated = generated_modules
            .iter()
            .map(|m| m.contents.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for module in &generated_modules {
            let decode_path = format!(
                "{}/../formal/Plonky2Spec/Generated/{}",
                env!("CARGO_MANIFEST_DIR"),
                module.relative
            );
            let on_disk = std::fs::read_to_string(&decode_path).unwrap_or_default();
            assert_eq!(
                on_disk, module.contents,
                "run export-constraints ({decode_path})"
            );
        }
        common::assert_split_manifest(
            &format!(
                "{}/../formal/Plonky2Spec/Generated",
                env!("CARGO_MANIFEST_DIR")
            ),
            &generated_modules
                .iter()
                .map(|m| m.relative.clone())
                .collect::<Vec<_>>(),
        );
        assert!(generated.contains(&format!(
            "theorem publicBatchWrapper{n}_decode (a : Assignment p) \
             (h : Satisfies (publicBatchWrapper{n} p) a) :"
        )));
        assert!(!generated.contains("Poseidon2Rows"));

        let bridge_modules = generate_public_wrapper_bridge_modules(n).unwrap();
        for module in &bridge_modules {
            let bridge_path = format!(
                "{}/../formal/Plonky2Bridge/Generated/{}",
                env!("CARGO_MANIFEST_DIR"),
                module.relative
            );
            let on_disk = std::fs::read_to_string(&bridge_path).unwrap_or_default();
            assert_eq!(
                on_disk, module.contents,
                "run export-constraints ({bridge_path})"
            );
        }
        common::assert_split_manifest(
            &format!(
                "{}/../formal/Plonky2Bridge/Generated",
                env!("CARGO_MANIFEST_DIR")
            ),
            &bridge_modules
                .iter()
                .map(|m| m.relative.clone())
                .collect::<Vec<_>>(),
        );
        let generated = bridge_modules
            .iter()
            .map(|m| m.contents.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(generated.contains(&format!("namespace Plonky2Bridge.PublicWrapper{n}\n")));
        assert!(generated.contains("RPublicBatch ro (inners a) (addr a) (out a) := by"));
        assert!(generated.contains("theorem end_to_end_wired"));
        assert!(generated.contains(&format!(
            "(hacc : ∀ i : Fin {n}, ProofAccepted perm Wrapper{leaves}.tree ((List.ofFn (innerPis i)).map a))"
        )));
        assert!(generated.contains(&format!(
            "refine public_batch_val ro (rows a) (k := {slots})"
        )));
        // Each inner's slot-count header is read back from its `connect` to the constant.
        assert_eq!(
            generated
                .matches(", kslots, ← Nat.cast_ofNat, ZMod.val_natCast_of_lt hc]")
                .count(),
            n
        );
        assert!(generated.contains(&format!(
            ".node \"public_batch_wrapper_n{n}\" (publicBatchWrapper{n} p)"
        )));
        assert_eq!(
            generated
                .matches(&format!("(Wrapper{leaves}.tree, List.ofFn (innerPis "))
                .count(),
            n
        );
        assert!(generated.contains(&format!(
            "exact Wrapper{leaves}.accepted_sound perm hpg hgw hgl (hacc i)"
        )));
        assert!(!generated.contains("private_batch_proof_sound"));
        assert!(!generated.contains("PrivateBatchProofAccepted"));
    }
}
