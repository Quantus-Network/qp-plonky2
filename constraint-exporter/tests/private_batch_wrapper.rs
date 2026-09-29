//! The recorded `n = 2` private-batch wrapper (`trace.rs`): the vendored trace parses into
//! the export + calls the generator expects, it matches the `wormholeSpec` lake package
//! when that has been fetched, and the checked-in `Generated/PrivateBatchWrapper2.lean` is
//! what it generates. See `../formal/PLAN.md` Step 8c.

use constraint_exporter::circuit::GateKind;
use constraint_exporter::gadget::{Fact, FACT_GROUP};
use constraint_exporter::trace::{
    assert_vendored_trace_matches_pinned, generate_private_batch_wrapper_lean, load, parse,
    traces_dir,
};
use plonky2::iop::target::Target;

const TRACE: &str = "private_batch_wrapper_n2.json";

fn wrapper() -> constraint_exporter::trace::LoadedTrace {
    load(&traces_dir().join(TRACE)).expect("vendored wrapper trace")
}

/// The vendored copy must be the file at the pinned `wormholeSpec` revision. The lake
/// package only exists after `lake build`/`lake update` in `formal/`, so this skips on the
/// Rust CI runners; the `formal-bridge` job runs it with `FORMAL_TRACES_STRICT=1` after
/// `lake build`, where a missing or differing file fails.
#[test]
fn vendored_trace_matches_pinned_package() {
    assert_vendored_trace_matches_pinned(TRACE);
}

#[test]
fn wrapper_trace_shape() {
    let t = wrapper();
    assert_eq!(t.circuit, "private_batch_wrapper_n2");
    let kinds = |k: &GateKind| t.ex.rows.iter().filter(|r| r.0 == *k).count();
    assert_eq!(kinds(&GateKind::Arithmetic { num_ops: 15 }), 55);
    assert_eq!(kinds(&GateKind::BaseSum2 { num_limbs: 59 }), 6);
    assert_eq!(kinds(&GateKind::Poseidon2), 4);
    assert_eq!(t.ex.rows.len(), 65);
    assert!(t.ex.rows.iter().all(|r| !matches!(r.0, GateKind::Other(_))));
    assert!(t.ex.copies.len() > 2 * 64, "copies are chunked");
    // 8 header/exit-slot felts + 2 · LEAF_PI_LEN.
    assert_eq!(t.ex.public_inputs.len(), 8 + 2 * 22);
    let named: Vec<&str> = t.ex.named.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        named,
        [
            "leaf_pis_0",
            "leaf_pis_1",
            "dummy_pre_image_0",
            "dummy_pre_image_1",
            "switches"
        ]
    );
    assert_eq!(t.ex.named[0].1.len(), 22);
    assert_eq!(t.ex.named[4].1.len(), 1);

    // Two double hashes: each `hash_dummy_nullifier_pre_image` is a sponge on the preimage
    // then a sponge on that digest.
    let hashes: Vec<(usize, [Target; 4])> = t
        .calls
        .iter()
        .filter_map(|c| match c.fact {
            Fact::Poseidon2 { row, inputs } => Some((row, inputs)),
            _ => None,
        })
        .collect();
    assert_eq!(hashes.len(), 4);
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
    assert_eq!(
        t.calls
            .iter()
            .filter(|c| matches!(c.fact, Fact::RangeCheck { .. }))
            .count(),
        6
    );
    assert!(t.calls.len() > FACT_GROUP, "decode theorem is grouped");
}

#[test]
fn wrapper_lean_is_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../formal/Plonky2Spec/Generated/PrivateBatchWrapper2.lean"
    );
    let on_disk = std::fs::read_to_string(path).unwrap();
    let generated = generate_private_batch_wrapper_lean().unwrap();
    assert_eq!(on_disk, generated, "run export-constraints");
    assert!(generated.contains("import Plonky2Spec.WiringSponge\n"));
    assert!(generated.contains(
        "theorem privateBatchWrapper2_decode (perm : St p → St p) (a : Assignment p) \
         (h : Satisfies (privateBatchWrapper2 p) a)\n    \
         (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :"
    ));
    assert_eq!(
        generated
            .matches("exact poseidon2Row_hash4 perm hp")
            .count(),
        4
    );
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
