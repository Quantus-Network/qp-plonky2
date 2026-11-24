//! The certified builds (`built.rs`, PLAN.md Steps 16, 17, 19 and 21): each vendored trace
//! replays into a `CircuitBuilder` — over its child circuit built the same way — to the
//! production builder's state digest, is completed to the whole builder and builds; the
//! exporter's certificate of that build passes the Rust check of `Built.Agrees` and names
//! the public-input hash `build` laid down. For the leaf and the private wrappers, whose
//! proofs a bridge consumes, the checked-in `Plonky2Bridge/Generated/*Built.lean` is what
//! the exporter generates.
//!
//! The production public batch (`public_batch_wrapper_n89`) is a 2^19-row build whose
//! replay needs more than 14 GB even without the `σ` commitment; its test is `#[ignore]`d
//! (`cargo test --release -- --ignored`).

use constraint_exporter::built::{
    build, certify, certify_circuit, digest, generate_built_modules, names, replay_circuit,
    verify_agrees, Certified,
};
use constraint_exporter::trace::{load, traces_dir};

mod common;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;

/// The deployed leaf's `circuit_digest`: `WormholeCircuit::new(wormhole_leaf_circuit_config())
/// .build_circuit()` in qp-zk-circuits over the published `qp-plonky2 1.6.0`.
const LEAF_CIRCUIT_DIGEST: [u64; 4] = [
    15101204307867406391,
    4984110629781477994,
    15964602978973519077,
    1260699525966399192,
];

#[test]
fn leaf_build_matches_deployed_digest() {
    let (_, builder) = replay_circuit("leaf_circuit").unwrap();
    let (data, view) = build(builder, true).unwrap();
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

/// The certificate's shape against the export, and that `verify_agrees` refuses a
/// corrupted certificate.
fn check_certificate(circuit: &str) -> Certified {
    let certified = certify_circuit(circuit).unwrap();
    let Certified {
        trace: t,
        view,
        cert,
        ..
    } = &certified;
    verify_agrees(cert, t, view).unwrap();
    assert_eq!(cert.row_at, t.row_at);
    assert_eq!(cert.routed, t.config.num_routed_wires);
    assert_eq!(cert.num_public_inputs, t.ex.public_inputs.len());
    assert_eq!(cert.virt.len(), t.virt_at.len());
    assert_eq!(cert.copy_wits.len(), t.ex.copies.len());
    assert_eq!(cert.const_wits.len(), t.ex.constants.len());
    assert_eq!(cert.pins.len(), t.ex.constants.len());
    for &(ci, iu, iv) in &cert.copy_wits {
        let cyc = &cert.cycles[ci];
        assert!(iu < cyc.len() && iv < cyc.len());
    }
    // The hash rows end at the `PublicInputGate`, one per rate block, each with its twelve
    // lanes, and the witnesses point into their cycles at the positions
    // `BuildCert.feedOk` reads.
    let blocks = t.ex.public_inputs.len().div_ceil(8);
    assert_eq!(
        cert.hash_rows,
        (cert.pi_row - blocks..cert.pi_row).collect::<Vec<_>>()
    );
    assert!(cert.hash_wits.iter().all(|w| w.lanes.len() == 12));
    assert_eq!(cert.out_wits.len(), 4);
    for (k, (&g, w)) in cert.hash_rows.iter().zip(&cert.hash_wits).enumerate() {
        let (ci, iu, iv) = w.swap;
        assert_eq!(cert.cycles[ci][iu], (g, 24));
        assert!(cert.zeros.contains(&cert.cycles[ci][iv]));
        let block = t.ex.public_inputs[8 * k..].len().min(8);
        for (i, &(ci, iu, iv)) in w.lanes.iter().enumerate() {
            let (x, y) = (cert.cycles[ci][iu], cert.cycles[ci][iv]);
            if i < block {
                assert_eq!(y, (g, i));
            } else if k == 0 {
                assert_eq!(x, (g, i));
                assert!(cert.zeros.contains(&y));
            } else {
                assert_eq!((x, y), ((g - 1, 12 + i), (g, i)));
            }
        }
    }
    let last = *cert.hash_rows.last().unwrap();
    for (i, &(ci, iu, iv)) in cert.out_wits.iter().enumerate() {
        assert_eq!(cert.cycles[ci][iu], (last, 12 + i));
        assert_eq!(cert.cycles[ci][iv], (cert.pi_row, i));
    }
    for &(r, c) in &cert.zeros {
        assert!(view
            .fixed
            .iter()
            .any(|&(fr, fc, v)| (fr, fc) == (r, c) && v == GoldilocksField::ZERO));
    }
    let corrupt = |f: &dyn Fn(&mut constraint_exporter::built::Cert)| {
        let mut bad = certify(t, view).unwrap();
        f(&mut bad);
        assert!(verify_agrees(&bad, t, view).is_err());
    };
    corrupt(&|bad| bad.cycles[0].swap(0, 1));
    corrupt(&|bad| bad.pins[0].1 .1 += 1);
    corrupt(&|bad| bad.pi_row -= 1);
    corrupt(&|bad| bad.hash_rows[0] = bad.pi_row);
    corrupt(&|bad| bad.zeros[0].1 += 1);
    corrupt(&|bad| bad.row_at[0] += 1);
    certified
}

fn check_lean_is_current(circuit: &str) {
    let root = format!(
        "{}/../formal/Plonky2Bridge/Generated",
        env!("CARGO_MANIFEST_DIR")
    );
    let modules = generate_built_modules(circuit).unwrap();
    for module in &modules {
        let path = format!("{root}/{}", module.relative);
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(on_disk, module.contents, "run export-constraints ({path})");
    }
    common::assert_split_manifest(
        &root,
        &modules
            .iter()
            .map(|m| m.relative.clone())
            .collect::<Vec<_>>(),
    );
    let generated = modules
        .iter()
        .map(|m| m.contents.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(generated.contains(&format!(
        "namespace {}\n",
        names(circuit).unwrap().namespace
    )));
    assert!(generated.contains("def tree : Recursive p :="));
    assert!(generated.contains("def buildCert : BuildCert where\n  rowAt := "));
    assert!(generated.contains("theorem copies_linked : ∀ xy ∈ ("));
    assert!(generated.contains("theorem build_refines_of_agrees (b : Built p) (h : b.Agrees ("));
    assert!(generated.contains("theorem checks_pi : buildCert.piOk piTargets buildCert.hashRows hashWits outWits = true := by\n  decide +kernel\n"));
    assert!(generated.contains("theorem build_hashes_of_agrees (b : Built p) (h : b.Agrees ("));
    // Nothing in the generated module is assumed.
    assert!(!generated.contains("axiom "));
    assert!(!generated.contains("sorry"));
}

#[test]
fn leaf_certificate_agrees_with_build() {
    let Certified { trace, cert, .. } = check_certificate("leaf_circuit");
    assert_eq!(cert.row_at, (0..223).collect::<Vec<_>>());
    assert_eq!(cert.virt.len(), 434);
    assert!(
        cert.virt.iter().all(Option::is_some),
        "every leaf virtual target is routed"
    );
    // `build` lays the hash down right after the export's 223 rows: 11 public inputs are
    // two rate-8 blocks, so two `PoseidonGate` rows, then the `PublicInputGate`.
    assert_eq!(cert.hash_rows, vec![223, 224]);
    assert_eq!(cert.pi_row, 225);
    // One wire is fixed to `0`: the `zero` target's, on the first `ConstantGate` row.
    assert_eq!(cert.zeros, vec![(226, 0)]);
    assert_eq!(trace.ex.copies.len(), cert.copy_wits.len());
}

#[test]
fn leaf_built_lean_is_current() {
    check_lean_is_current("leaf_circuit");
}

/// A wrapper's complete export is the whole builder: the recorded trace first — the
/// verifier rows whose free slots the wrapper's ops took, then the rows it added — then
/// the verifier rows the trace did not list, so that every builder row is placed once
/// and `build`'s hash rows follow them all.
fn check_wrapper(circuit: &str) {
    let Certified { trace, cert, .. } = check_certificate(circuit);
    let recorded = load(&traces_dir().join(format!("{circuit}.json"))).unwrap();
    assert_eq!(&cert.row_at[..recorded.row_at.len()], &recorded.row_at[..]);
    assert!(cert.row_at[recorded.row_at.len()..]
        .iter()
        .all(|&r| r < trace.start.rows));
    let mut placed = cert.row_at.clone();
    placed.sort_unstable();
    assert_eq!(placed, (0..cert.row_at.len()).collect::<Vec<_>>());
    assert!(cert.virt.iter().all(Option::is_some));
    assert_eq!(cert.hash_rows[0], cert.row_at.len());
}

#[test]
fn private_batch_wrapper_n2_certificate_agrees_with_build() {
    check_wrapper("private_batch_wrapper_n2");
}

#[test]
fn private_batch_wrapper_n2_built_lean_is_current() {
    check_lean_is_current("private_batch_wrapper_n2");
}

#[test]
fn private_batch_wrapper_n4_built_lean_is_current() {
    check_lean_is_current("private_batch_wrapper_n4");
}

#[test]
fn private_batch_wrapper_n8_certificate_agrees_with_build() {
    check_wrapper("private_batch_wrapper_n8");
}

#[test]
fn private_batch_wrapper_n8_built_lean_is_current() {
    check_lean_is_current("private_batch_wrapper_n8");
}

#[test]
fn public_batch_wrapper_n2_certificate_agrees_with_build() {
    check_wrapper("public_batch_wrapper_n2");
}

#[test]
fn public_batch_wrapper_n4_certificate_agrees_with_build() {
    check_wrapper("public_batch_wrapper_n4");
}

#[test]
#[ignore = "the production public batch is 2^19 rows and over 14 GB; run with --release --ignored"]
fn public_batch_wrapper_n89_certificate_agrees_with_build() {
    check_wrapper("public_batch_wrapper_n89");
}
