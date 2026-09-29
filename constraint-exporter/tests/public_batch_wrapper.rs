//! The recorded `n_inner = 2` public-batch wrapper (`trace.rs`): the vendored trace parses
//! into the export + calls the generator expects, it matches the `wormholeSpec` lake package
//! when that has been fetched, and the checked-in `Generated/PublicBatchWrapper2.lean` is
//! what it generates. See `../formal/PLAN.md` Step 8f.

use constraint_exporter::circuit::GateKind;
use constraint_exporter::gadget::{Fact, FACT_GROUP};
use constraint_exporter::trace::{
    assert_vendored_trace_matches_pinned, generate_public_batch_wrapper_lean, load, traces_dir,
};

const TRACE: &str = "public_batch_wrapper_n2.json";

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
    assert_eq!(t.circuit, "public_batch_wrapper_n2");
    // The wrapper is pure boolean/select logic: no hashes, no range checks.
    assert!(t
        .ex
        .rows
        .iter()
        .all(|r| r.0 == GateKind::Arithmetic { num_ops: 20 }));
    assert_eq!(t.ex.rows.len(), 15);
    // 12 header felts + 2 · 4 slots · 5 limbs + 2 · 2 nullifiers · 4 limbs.
    assert_eq!(t.ex.public_inputs.len(), 12 + 40 + 16);
    let named: Vec<(&str, usize)> =
        t.ex.named
            .iter()
            .map(|(n, v)| (n.as_str(), v.len()))
            .collect();
    assert_eq!(
        named,
        [
            ("inner_pis_0", 52),
            ("inner_pis_1", 52),
            ("aggregator_address", 4)
        ]
    );

    let count = |f: fn(&Fact) -> bool| t.calls.iter().filter(|c| f(&c.fact)).count();
    // 2 dummy checks + 2 · 3 consistency checks, each a `bytes_digest_eq`/`is_equal`.
    assert_eq!(count(|f| matches!(f, Fact::IsEqual { .. })), 20);
    // Per inner: 7 first-real selects + 20 slot limbs + 8 nullifier limbs.
    assert_eq!(count(|f| matches!(f, Fact::Select { .. })), 70);
    // Each `or(is_dummy, eq) = 1` consistency check is pinned with a `connect`.
    assert_eq!(count(|f| matches!(f, Fact::Connect { .. })), 6);
    assert_eq!(count(|f| matches!(f, Fact::Poseidon2 { .. })), 0);
    assert_eq!(count(|f| matches!(f, Fact::RangeCheck { .. })), 0);
    assert_eq!(t.calls.len(), 122);
    assert!(t.calls.len() > FACT_GROUP, "decode theorem is grouped");
}

#[test]
fn wrapper_lean_is_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../formal/Plonky2Spec/Generated/PublicBatchWrapper2.lean"
    );
    let on_disk = std::fs::read_to_string(path).unwrap();
    let generated = generate_public_batch_wrapper_lean().unwrap();
    assert_eq!(on_disk, generated, "run export-constraints");
    assert!(generated.contains(
        "theorem publicBatchWrapper2_decode (a : Assignment p) \
         (h : Satisfies (publicBatchWrapper2 p) a) :"
    ));
    assert!(!generated.contains("Poseidon2Rows"));
}
