//! Generates the auto-extracted Lean files under `formal/Plonky2Spec/Generated/`
//! from the live gate code:
//!   * `Gates.lean`     — ArithmeticGate + BaseSumGate<2>  (Step 2b)
//!   * `Poseidon2.lean` — Poseidon2Gate permutation        (Step 3a)
//!
//!     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
//!
//! Files are written to their canonical paths next to the Lean spec; contents
//! are also echoed to stdout.

use std::fs;
use std::path::{Path, PathBuf};

fn generated_dir() -> PathBuf {
    // <repo>/qp-plonky2/constraint-exporter -> <repo>/qp-plonky2/formal/Plonky2Spec/Generated
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("formal")
        .join("Plonky2Spec")
        .join("Generated")
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

    write(&dir, "Gates.lean", &gates)?;
    write(&dir, "Poseidon2.lean", &poseidon2)?;

    print!("{gates}");
    println!("\n-- ===== Poseidon2.lean =====");
    print!("{poseidon2}");
    Ok(())
}
