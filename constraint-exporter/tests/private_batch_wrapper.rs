//! The recorded private-batch wrappers (`trace.rs`, `private_wrapper.rs`) at every leaf count
//! with a trace: each vendored trace parses into the export + calls the generators expect, it
//! matches the `wormholeSpec` lake package when that has been fetched, `Shape::read`
//! recognises the wrapper's call structure (and rejects a perturbed one), and the checked-in
//! `Plonky2Spec/Generated/PrivateBatchWrapper{N}.lean` and
//! `Plonky2Bridge/Generated/Wrapper{N}.lean` are what they generate. See `../formal/PLAN.md`
//! Steps 8c and 8e.

use constraint_exporter::circuit::GateKind;
use constraint_exporter::gadget::{Fact, FACT_GROUP};
use constraint_exporter::private_wrapper::{generate_private_wrapper_bridge_lean, Shape};
use constraint_exporter::trace::{
    assert_vendored_trace_matches_pinned, generate_private_batch_wrapper_lean, load, parse,
    traces_dir,
};
use plonky2::iop::target::Target;

const SIZES: [usize; 1] = [2];

fn trace_name(n: usize) -> String {
    format!("private_batch_wrapper_n{n}.json")
}

fn wrapper(n: usize) -> constraint_exporter::trace::LoadedTrace {
    load(&traces_dir().join(trace_name(n))).expect("vendored wrapper trace")
}

/// `permute_digests4` over `n` positions: round `r` switches the pairs starting at `r mod 2`.
fn switch_count(n: usize) -> usize {
    (0..n).map(|r| (n - r % 2) / 2).sum()
}

/// The gadget calls `build_private_batch_constraints` makes for `n` leaves: dummy checks
/// (7 per leaf), the header scan (10), consistency (13), candidate masks (10), input total
/// (2), output total (2 per leaf), the fee comparator (6), the grouping loop
/// (`52n² + 4n`), uniqueness (`n + 11·n(n−1)/2`), dummy nullifiers (6 per leaf) and the
/// switch network (9 per switch).
fn expected_calls(n: usize) -> usize {
    7 * n
        + 10 * n
        + 13 * n
        + 10 * n
        + 2 * n
        + 2 * n
        + 6
        + (52 * n * n + 4 * n)
        + (n + 11 * n * (n - 1) / 2)
        + 6 * n
        + 9 * switch_count(n)
}

/// The vendored copies must be the files at the pinned `wormholeSpec` revision. The lake
/// package only exists after `lake build`/`lake update` in `formal/`, so this skips on the
/// Rust CI runners; the `formal-bridge` job runs it with `FORMAL_TRACES_STRICT=1` after
/// `lake build`, where a missing or differing file fails.
#[test]
fn vendored_trace_matches_pinned_package() {
    for n in SIZES {
        assert_vendored_trace_matches_pinned(&trace_name(n));
    }
}

#[test]
fn wrapper_trace_shape() {
    for n in SIZES {
        let t = wrapper(n);
        assert_eq!(t.circuit, format!("private_batch_wrapper_n{n}"));
        let kinds = |k: &GateKind| t.ex.rows.iter().filter(|r| r.0 == *k).count();
        assert_eq!(kinds(&GateKind::Poseidon2), 2 * n);
        assert_eq!(kinds(&GateKind::BaseSum2 { num_limbs: 59 }), 2 + 2 * n);
        assert!(t.ex.rows.iter().all(|r| !matches!(r.0, GateKind::Other(_))));
        assert!(t.ex.copies.len() > 2 * 64, "copies are chunked");
        // 8 header felts + 2n slots · 5 limbs + n nullifiers · 4 limbs, zero-padded.
        assert!(t.ex.public_inputs.len() >= 8 + 10 * n + 4 * n);
        let named: Vec<(&str, usize)> =
            t.ex.named
                .iter()
                .map(|(n, v)| (n.as_str(), v.len()))
                .collect();
        let mut expected: Vec<(String, usize)> =
            (0..n).map(|i| (format!("leaf_pis_{i}"), 22)).collect();
        expected.extend((0..n).map(|i| (format!("dummy_pre_image_{i}"), 4)));
        expected.push(("switches".into(), switch_count(n)));
        assert_eq!(
            named,
            expected
                .iter()
                .map(|(s, k)| (s.as_str(), *k))
                .collect::<Vec<_>>()
        );

        // Each `hash_dummy_nullifier_pre_image` is a sponge on the preimage then a sponge on
        // that digest.
        let hashes: Vec<(usize, [Target; 4])> = t
            .calls
            .iter()
            .filter_map(|c| match c.fact {
                Fact::Poseidon2 { row, inputs } => Some((row, inputs)),
                _ => None,
            })
            .collect();
        assert_eq!(hashes.len(), 2 * n);
        for pair in hashes.chunks(2) {
            let [(r0, pre), (r1, mid)] = pair else {
                unreachable!()
            };
            assert!(pre
                .iter()
                .all(|t| matches!(t, Target::VirtualTarget { .. })));
            assert_eq!(*mid, core::array::from_fn(|i| Target::wire(*r0, 12 + i)));
            assert_eq!(*r1, r0 + 1);
        }
        let count = |f: fn(&Fact) -> bool| t.calls.iter().filter(|c| f(&c.fact)).count();
        // Fee complement and difference, plus one per exit slot.
        assert_eq!(count(|f| matches!(f, Fact::RangeCheck { .. })), 2 + 2 * n);
        // Per leaf: block, asset and fee pins; per pair: the collision pin.
        assert_eq!(
            count(|f| matches!(f, Fact::Connect { .. })),
            3 * n + n * (n - 1) / 2
        );
        assert_eq!(
            count(|f| matches!(f, Fact::AssertBool { .. })),
            switch_count(n)
        );
        assert_eq!(t.calls.len(), expected_calls(n));
        assert!(t.calls.len() > FACT_GROUP, "decode theorem is grouped");

        let shape = Shape::read(&t).expect("the trace has the wrapper's call structure");
        assert_eq!(shape.n, n);
    }
}

/// `Shape::read` walks the calls positionally, so it must notice when the structure it
/// assumes is not what was recorded — otherwise a changed wrapper would yield a bridge that
/// talks about the wrong wires (and only fails, less legibly, in Lean).
#[test]
fn shape_rejects_perturbed_traces() {
    let t = wrapper(2);
    // Swap the first two `is_equal`s of leaf 0's dummy check: the limbs no longer line up.
    let mut swapped = wrapper(2);
    swapped.calls.swap(0, 1);
    let err = Shape::read(&swapped).unwrap_err();
    assert!(
        err.contains("call 0: expected is_equal for dummy check of leaf 0 limb 0"),
        "{err}"
    );
    // Drop the last call: the switch network is incomplete.
    let mut short = wrapper(2);
    short.calls.pop();
    let err = Shape::read(&short).unwrap_err();
    assert!(
        err.contains("call 343: expected switch 0 routed 1 limb 3, but the trace ends"),
        "{err}"
    );
    // Reorder the public inputs: the output layout check fires.
    let mut pis = wrapper(2);
    pis.ex.public_inputs.swap(1, 2);
    let err = Shape::read(&pis).unwrap_err();
    assert!(err.contains("recorded public inputs"), "{err}");
    // A consistency `connect` pinned to something other than `one`.
    let mut pin = wrapper(2);
    let k = t
        .calls
        .iter()
        .position(|c| matches!(c.fact, Fact::Connect { .. }))
        .unwrap();
    if let Fact::Connect { x, .. } = pin.calls[k].fact {
        pin.calls[k].fact = Fact::Connect { x, y: x };
    }
    let err = Shape::read(&pin).unwrap_err();
    assert!(
        err.contains("expected connect for connect(block_ok_0, one)"),
        "{err}"
    );
    // A folded `add(zero, x)` that does not return `x`.
    let mut fold = wrapper(2);
    let k = t
        .calls
        .iter()
        .position(|c| matches!(c.fact, Fact::Add { .. }))
        .unwrap();
    if let Fact::Add { x, y, .. } = fold.calls[k].fact {
        fold.calls[k].fact = Fact::Add { x, y, out: x };
    }
    let err = Shape::read(&fold).unwrap_err();
    assert!(err.contains("should fold to masked_input_0"), "{err}");
}

#[test]
fn wrapper_lean_is_current() {
    for n in SIZES {
        let decode_path = format!(
            "{}/../formal/Plonky2Spec/Generated/PrivateBatchWrapper{n}.lean",
            env!("CARGO_MANIFEST_DIR")
        );
        let on_disk = std::fs::read_to_string(&decode_path).unwrap();
        let generated = generate_private_batch_wrapper_lean(n).unwrap();
        assert_eq!(on_disk, generated, "run export-constraints ({decode_path})");
        assert!(generated.contains("import Plonky2Spec.WiringSponge\n"));
        assert!(generated.contains(&format!(
            "theorem privateBatchWrapper{n}_decode (perm : St p → St p) (a : Assignment p) \
             (h : Satisfies (privateBatchWrapper{n} p) a)\n    \
             (hp : Poseidon2Rows perm (privateBatchWrapper{n} p) a) :"
        )));
        assert_eq!(
            generated
                .matches("exact poseidon2Row_hash4 perm hp")
                .count(),
            2 * n
        );

        let bridge_path = format!(
            "{}/../formal/Plonky2Bridge/Generated/Wrapper{n}.lean",
            env!("CARGO_MANIFEST_DIR")
        );
        let on_disk = std::fs::read_to_string(&bridge_path).unwrap();
        let generated = generate_private_wrapper_bridge_lean(n).unwrap();
        assert_eq!(on_disk, generated, "run export-constraints ({bridge_path})");
        assert!(generated.contains(&format!("namespace Plonky2Bridge.Wrapper{n}\n")));
        assert!(generated.contains("RPrivateBatch (spongeRO perm) (leaves a) (us a) (out a) := by"));
        assert!(generated.contains("theorem end_to_end_wired"));
        assert!(generated.contains("refine private_batch_val_rows perm hpg (rows a) (rounds a)"));
        assert_eq!(generated.matches("· -- slot ").count(), 8 * 2 * n);
    }
}

#[test]
fn malformed_traces_are_rejected() {
    let base = r#"{"circuit":"t","num_routed_wires":80,"num_virtual_targets":2,
        "rows":[],"copies":[],"constants":[],"public_inputs":[],"named":[],"calls":[CALLS]}"#;
    let with = |calls: &str| parse(&base.replace("CALLS", calls));
    assert!(with("").is_ok());
    let err = with(
        r#"{"kind":"xor","args":["v0","v1"],"outs":["v2"],"fresh":[],"rows":[0,0],"copies":[0,0]}"#,
    )
    .unwrap_err();
    assert!(err.contains("unknown gadget call kind"), "{err}");
    let err = with(
        r#"{"kind":"add","args":["v0"],"outs":["v2"],"fresh":[],"rows":[0,0],"copies":[0,0]}"#,
    )
    .unwrap_err();
    assert!(err.contains("expected 2 targets"), "{err}");
    let err = with(
        r#"{"kind":"range_check","args":["v0"],"outs":[],"fresh":[],"rows":[0,0],"copies":[0,0]}"#,
    )
    .unwrap_err();
    assert!(err.contains("without bits"), "{err}");
    let err = with(
        r#"{"kind":"poseidon2_hash","args":["v0","v1","v2","v3"],"outs":["w0:12","w0:13","w0:14","w0:15"],"fresh":[],"rows":[0,1],"copies":[0,0]}"#,
    )
    .unwrap_err();
    assert!(err.contains("row range [0, 1] is out of bounds"), "{err}");
    let err = with(
        r#"{"kind":"poseidon2_hash","args":["v0","v1","v2","v3"],"outs":["w0:12","w0:13","w0:14","w0:15"],"fresh":[],"rows":[0,0],"copies":[0,0]}"#,
    )
    .unwrap_err();
    assert!(err.contains("expected one Poseidon2Gate row"), "{err}");
    let err = parse(
        &base
            .replace("CALLS", "")
            .replace(r#""public_inputs":[]"#, r#""public_inputs":["x7"]"#),
    )
    .unwrap_err();
    assert!(err.contains("bad target"), "{err}");
}
