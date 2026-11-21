//! The certified leaf build (`built.rs`, PLAN.md Step 16): the vendored leaf trace replays
//! into a `CircuitBuilder` and builds to the deployed leaf circuit — same digest as
//! `WormholeCircuit::build_circuit()` in qp-zk-circuits on `qp-plonky2 = "=1.6.0"` — the
//! exporter's certificate of that build passes the Rust check of `Built.Agrees`, it names
//! the public-input hash `build` laid down (Step 17), and the checked-in
//! `Plonky2Bridge/Generated/LeafBuilt.lean` is what it generates.

use constraint_exporter::built::{
    certify, digest, generate_leaf_built_lean, leaf_cert, verify_agrees, LEAF_CIRCUIT_DIGEST,
};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;

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
fn leaf_certificate_names_the_public_input_hash() {
    let (t, _, view, cert) = leaf_cert().unwrap();
    // `build` lays the hash down right after the export's 223 rows: 11 public inputs are
    // two rate-8 blocks, so two `PoseidonGate` rows, then the `PublicInputGate`.
    assert_eq!(cert.hash_rows, vec![223, 224]);
    assert_eq!(cert.pi_row, 225);
    assert_eq!(cert.hash_wits.len(), 2);
    assert!(cert.hash_wits.iter().all(|w| w.lanes.len() == 12));
    assert_eq!(cert.out_wits.len(), 4);
    // One wire is fixed to `0`: the `zero` target's, on the first `ConstantGate` row.
    assert_eq!(cert.zeros, vec![(226, 0)]);
    assert!(view
        .fixed
        .iter()
        .any(|&(r, c, v)| (r, c) == (226, 0) && v == GoldilocksField::ZERO));
    // Every witness points into its cycle, and the cycles it names tie the sources to the
    // positions `BuildCert.feedOk` reads: the swap wire at `iu` and a zero wire at `iv`;
    // for a lane, the source at `iu` and the gate's input at `iv`.
    for (k, (&g, w)) in cert.hash_rows.iter().zip(&cert.hash_wits).enumerate() {
        let (ci, iu, iv) = w.swap;
        assert_eq!(cert.cycles[ci][iu], (g, 24));
        assert!(cert.zeros.contains(&cert.cycles[ci][iv]));
        for (i, &(ci, iu, iv)) in w.lanes.iter().enumerate() {
            let (x, y) = (cert.cycles[ci][iu], cert.cycles[ci][iv]);
            match (k, i) {
                // First block: lanes `0..8` are the public inputs, `8..12` the `zero`.
                (0, i) if i < 8 => assert_eq!(y, (g, i)),
                (0, _) => {
                    assert_eq!(x, (g, i));
                    assert!(cert.zeros.contains(&y));
                }
                // Second block: lanes `0..3` are public inputs `8..11`, the rest the
                // previous row's outputs.
                (_, i) if i < 3 => assert_eq!(y, (g, i)),
                (_, i) => assert_eq!((x, y), ((g - 1, 12 + i), (g, i))),
            }
        }
    }
    for (i, &(ci, iu, iv)) in cert.out_wits.iter().enumerate() {
        assert_eq!(cert.cycles[ci][iu], (224, 12 + i));
        assert_eq!(cert.cycles[ci][iv], (225, i));
    }
    // A certificate naming the wrong rows is refused.
    let mut bad = certify(&t.ex, &view).unwrap();
    bad.pi_row = 224;
    assert!(verify_agrees(&bad, &t.ex, &view).is_err());
    let mut bad = certify(&t.ex, &view).unwrap();
    bad.hash_rows[0] = 225;
    assert!(verify_agrees(&bad, &t.ex, &view).is_err());
    let mut bad = certify(&t.ex, &view).unwrap();
    bad.zeros[0] = (226, 1);
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
    assert!(generated.contains("  zeros := [(226, 0)]\n  piRow := 225\n  hashRows := [223, 224]\n"));
    assert!(generated.contains(
        "theorem checks_pi : buildCert.piOk piTargets buildCert.hashRows hashWits outWits = true := by\n  \
         decide +kernel\n"
    ));
    assert!(generated.contains(
        "theorem build_hashes_of_agrees (b : Built p) (h : b.Agrees (leafCircuit p) buildCert) :\n    \
         b.HashesPublicInputs (leafCircuit p) :="
    ));
    // Nothing in the generated module is assumed.
    assert!(!generated.contains("axiom "));
    assert!(!generated.contains("sorry"));
}
