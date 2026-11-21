//! The certified leaf build (`built.rs`, PLAN.md Step 16): the vendored leaf trace replays
//! into a `CircuitBuilder` and builds to the deployed leaf circuit — same digest as
//! `WormholeCircuit::build_circuit()` in qp-zk-circuits on `qp-plonky2 = "=1.6.0"` — the
//! exporter's certificate of that build passes the Rust check of `Built.Agrees`, and the
//! checked-in `Plonky2Bridge/Generated/LeafBuilt.lean` is what it generates.

use constraint_exporter::built::{
    certify, digest, generate_leaf_built_lean, leaf_cert, verify_agrees, LEAF_CIRCUIT_DIGEST,
};

#[test]
fn leaf_build_matches_deployed_digest() {
    let (_, data, view, _) = leaf_cert().unwrap();
    assert_eq!(data.common.degree_bits(), 8);
    assert_eq!(view.gate_instances.len(), 256);
    assert_eq!(view.num_routed_wires, 60);
    assert_eq!(view.num_public_inputs, 11);
    assert_eq!(
        digest(&data),
        LEAF_CIRCUIT_DIGEST,
        "the replayed leaf build is not the deployed leaf circuit"
    );
}

#[test]
fn leaf_certificate_agrees_with_build() {
    let (t, _, view, cert) = leaf_cert().unwrap();
    // `certify` already ran `verify_agrees`; run it again on its output, and check the
    // certificate's shape against the export.
    verify_agrees(&cert, &t.ex, &view).unwrap();
    assert_eq!(cert.export_rows, t.ex.rows.len());
    assert_eq!(cert.export_rows, 223);
    assert_eq!(cert.routed, 60);
    assert_eq!(cert.num_public_inputs, 11);
    assert_eq!(cert.virt.len(), 434);
    assert!(
        cert.virt.iter().all(Option::is_some),
        "every leaf virtual target is routed"
    );
    assert_eq!(cert.copy_wits.len(), t.ex.copies.len());
    assert_eq!(cert.const_wits.len(), t.ex.constants.len());
    assert_eq!(cert.pins.len(), t.ex.constants.len());
    // Each cycle is a `σ`-cycle: every position of the built system it names is in the
    // routed domain, and the witnesses point into it.
    for &(ci, iu, iv) in &cert.copy_wits {
        let cyc = &cert.cycles[ci];
        assert!(iu < cyc.len() && iv < cyc.len());
    }
    // A corrupted certificate is refused.
    let mut bad = certify(&t.ex, &view).unwrap();
    bad.cycles[0].swap(0, 1);
    assert!(verify_agrees(&bad, &t.ex, &view).is_err());
    let mut bad = certify(&t.ex, &view).unwrap();
    bad.pins[0].1 .1 += 1;
    assert!(verify_agrees(&bad, &t.ex, &view).is_err());
}

#[test]
fn leaf_built_lean_is_current() {
    let path = format!(
        "{}/../formal/Plonky2Bridge/Generated/LeafBuilt.lean",
        env!("CARGO_MANIFEST_DIR")
    );
    let on_disk = std::fs::read_to_string(&path).unwrap();
    let generated = generate_leaf_built_lean().unwrap();
    assert_eq!(on_disk, generated, "run export-constraints ({path})");
    assert!(generated.contains("namespace Plonky2Bridge.LeafCircuit\n"));
    assert!(generated.contains("def buildCert : BuildCert where\n  exportRows := 223\n"));
    assert!(generated
        .contains("theorem checks_copies : buildCert.copiesOk ((((((((leafCircuit.copies0 ++ "));
    assert!(generated.contains(
        "theorem build_refines_of_agrees (b : Built p) (h : b.Agrees (leafCircuit p) buildCert) :\n    \
         b.Refines (leafCircuit p) :="
    ));
    // Nothing in the generated module is assumed.
    assert!(!generated.contains("axiom "));
    assert!(!generated.contains("sorry"));
}
