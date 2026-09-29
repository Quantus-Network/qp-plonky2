//! Generates the auto-extracted Lean files under `formal/Plonky2Spec/Generated/` (and the
//! bridge compositions under `formal/Plonky2Bridge/Generated/`) from the live gate code:
//!   * `Gates.lean`          — ArithmeticGate + BaseSumGate<2>     (Step 2b)
//!   * `Poseidon2.lean`      — Poseidon2Gate permutation (flat)    (Step 3a)
//!   * `Poseidon2Prims.lean` — Poseidon2 sbox7/mdsLight/internalMix (Step 3b)
//!   * `NullifierSelectCircuit.lean` — pre-`build` wiring of the nullifier-select path (Step 8)
//!   * `PrivateBatchWrapper2.lean` — the recorded `n = 2` private-batch wrapper (Step 8c)
//!   * `PublicBatchWrapper{2,4}.lean` — the recorded `n_inner = 2, 4` public-batch wrappers
//!     (Steps 8f, 8e)
//!   * `Plonky2Bridge/Generated/PublicWrapper{2,4}.lean` — their compositions into
//!     `RPublicBatch` (Step 8e)
//!
//!     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
//!
//! Files are written to their canonical paths next to the Lean spec; contents
//! are also echoed to stdout.

use std::fs;
use std::path::{Path, PathBuf};

/// The public-batch wrapper sizes with recorded traces.
const PUBLIC_WRAPPER_SIZES: [usize; 2] = [2, 4];

fn formal_dir() -> PathBuf {
    // <repo>/qp-plonky2/constraint-exporter -> <repo>/qp-plonky2/formal
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("formal")
}

fn generated_dir() -> PathBuf {
    formal_dir().join("Plonky2Spec").join("Generated")
}

fn bridge_generated_dir() -> PathBuf {
    formal_dir().join("Plonky2Bridge").join("Generated")
}

fn write(dir: &Path, name: &str, contents: &str) -> std::io::Result<()> {
    let path = dir.join(name);
    fs::write(&path, contents)?;
    eprintln!("wrote {} ({} bytes)", path.display(), contents.len());
    Ok(())
}

fn main() -> std::io::Result<()> {
    let dir = generated_dir();
    fs::create_dir_all(&dir)?;

    let gates = constraint_exporter::generate_lean();
    let poseidon2 = constraint_exporter::generate_poseidon2_lean();
    let poseidon2_prims = constraint_exporter::generate_poseidon2_prims_lean();
    let nullifier_select = constraint_exporter::circuit::generate_nullifier_select_lean();
    let gadget_zoo = constraint_exporter::gadget::generate_gadget_zoo_lean();
    let gadget_edge_cases = constraint_exporter::gadget::generate_gadget_edge_cases_lean();
    let wrapper = constraint_exporter::trace::generate_private_batch_wrapper_lean()
        .map_err(std::io::Error::other)?;
    let mut public_wrappers = Vec::new();
    for n in PUBLIC_WRAPPER_SIZES {
        let decode = constraint_exporter::trace::generate_public_batch_wrapper_lean(n)
            .map_err(std::io::Error::other)?;
        let bridge = constraint_exporter::public_wrapper::generate_public_wrapper_bridge_lean(n)
            .map_err(std::io::Error::other)?;
        public_wrappers.push((n, decode, bridge));
    }

    write(&dir, "Gates.lean", &gates)?;
    write(&dir, "Poseidon2.lean", &poseidon2)?;
    write(&dir, "Poseidon2Prims.lean", &poseidon2_prims)?;
    write(&dir, "NullifierSelectCircuit.lean", &nullifier_select)?;
    write(&dir, "GadgetZooCircuit.lean", &gadget_zoo)?;
    write(&dir, "GadgetEdgeCasesCircuit.lean", &gadget_edge_cases)?;
    write(&dir, "PrivateBatchWrapper2.lean", &wrapper)?;
    let bridge_dir = bridge_generated_dir();
    fs::create_dir_all(&bridge_dir)?;
    for (n, decode, bridge) in &public_wrappers {
        write(&dir, &format!("PublicBatchWrapper{n}.lean"), decode)?;
        write(&bridge_dir, &format!("PublicWrapper{n}.lean"), bridge)?;
    }

    print!("{gates}");
    println!("\n-- ===== Poseidon2.lean =====");
    print!("{poseidon2}");
    println!("\n-- ===== Poseidon2Prims.lean =====");
    print!("{poseidon2_prims}");
    println!("\n-- ===== NullifierSelectCircuit.lean =====");
    print!("{nullifier_select}");
    println!("\n-- ===== GadgetZooCircuit.lean =====");
    print!("{gadget_zoo}");
    println!("\n-- ===== GadgetEdgeCasesCircuit.lean =====");
    print!("{gadget_edge_cases}");
    println!("\n-- ===== PrivateBatchWrapper2.lean =====");
    print!("{wrapper}");
    for (n, decode, bridge) in &public_wrappers {
        println!("\n-- ===== PublicBatchWrapper{n}.lean =====");
        print!("{decode}");
        println!("\n-- ===== Plonky2Bridge/Generated/PublicWrapper{n}.lean =====");
        print!("{bridge}");
    }
    Ok(())
}
