//! Generates `formal/Plonky2Spec/Generated/Gates.lean` from the live gate code.
//!
//!     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
//!
//! With no argument it writes the canonical path (next to the Lean spec) and
//! also echoes the contents to stdout. Pass a path to override the destination,
//! or `-` to only print.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

fn default_output() -> PathBuf {
    // <repo>/qp-plonky2/constraint-exporter -> <repo>/qp-plonky2/formal/...
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("formal")
        .join("Plonky2Spec")
        .join("Generated")
        .join("Gates.lean")
}

fn main() -> std::io::Result<()> {
    let lean = constraint_exporter::generate_lean();

    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("-") => {
            print!("{lean}");
            std::io::stdout().flush()?;
        }
        other => {
            let path = other.map(PathBuf::from).unwrap_or_else(default_output);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, &lean)?;
            eprintln!("wrote {} ({} bytes)", path.display(), lean.len());
            print!("{lean}");
        }
    }
    Ok(())
}
