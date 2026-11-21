//! Generates the auto-extracted Lean files under `formal/Plonky2Spec/Generated/` (and the
//! bridge compositions under `formal/Plonky2Bridge/Generated/`) from the live gate code:
//!   * `Gates.lean`          — ArithmeticGate + BaseSumGate<2>     (Step 2b)
//!   * `Poseidon2.lean`      — Poseidon2Gate permutation (flat)    (Step 3a)
//!   * `Poseidon2Prims.lean` — Poseidon2 sbox7/mdsLight/internalMix (Step 3b)
//!   * `NullifierSelectCircuit.lean` — pre-`build` wiring of the nullifier-select path (Step 8)
//!   * `PrivateBatchWrapper{2,4}.lean` — the recorded `n = 2, 4` private-batch wrappers
//!     (Steps 8c, 8e)
//!   * `PublicBatchWrapper{2,4}.lean` — the recorded `n_inner = 2, 4` public-batch wrappers
//!     (Steps 8f, 8e)
//!   * `Plonky2Bridge/Generated/Wrapper{2,4}.lean` and `PublicWrapper{2,4}.lean` — their
//!     compositions into `RPrivateBatch` / `RPublicBatch` (Step 8e)
//!   * `LeafCircuit.lean`    — the recorded leaf circuit                (Step 9b)
//!   * `Plonky2Bridge/Generated/Leaf.lean` — its composition into `Rleaf` (Step 9d)
//!   * `Plonky2Bridge/Generated/LeafBuilt.lean` — the certificate of its build (Step 16)
//!
//!     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
//!
//! Files are written to their canonical paths next to the Lean spec; contents
//! are also echoed to stdout.

use std::fs;
use std::path::{Path, PathBuf};

/// The private-batch wrapper leaf counts with recorded traces. `n = 2` and `4` are
/// recorded with as many exit slots as leaves; `n = 8` is the production shape
/// (`NUM_EXIT_SLOTS = 32`), read off the trace.
const PRIVATE_WRAPPER_SIZES: [usize; 3] = [2, 4, 8];
/// The public-batch wrapper arities with recorded traces. `n = 2` and `4` are over
/// 2-leaf private batches; `n = 89` is the production arity over the `n = 8` one.
const PUBLIC_WRAPPER_SIZES: [usize; 3] = [2, 4, 89];

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

/// Write modules for one public-batch wrapper.
///
/// A split wrapper has pieces under `{dir_name}/` plus a root file next to that
/// directory. Pieces are owned through `{dir_name}/.generated-files`: only a
/// basename on the previous manifest is eligible for deletion. Files outside
/// the directory, such as `PublicBatchWrapper2.lean`, are written in place.
fn write_split(parent: &Path, dir_name: &str, modules: &[(&str, &str)]) -> std::io::Result<()> {
    let prefix = format!("{dir_name}/");
    let mut inside = Vec::new();
    for &(relative, contents) in modules {
        if let Some(name) = relative.strip_prefix(&prefix) {
            inside.push((name, contents));
        } else {
            write(parent, relative, contents)?;
        }
    }
    if !inside.is_empty() {
        let sub = parent.join(dir_name);
        let removed = constraint_exporter::lean_split::sync_owned_files(&sub, &inside)?;
        for name in removed {
            eprintln!("removed {}", sub.join(name).display());
        }
        for (name, contents) in &inside {
            eprintln!(
                "wrote {} ({} bytes)",
                sub.join(name).display(),
                contents.len()
            );
        }
    }
    Ok(())
}

fn print_exported(wrapper_arity: usize, relative: &str, contents: &str) {
    if constraint_exporter::lean_split::stdout_includes_source(wrapper_arity, contents.len()) {
        println!("-- ===== {relative} =====");
        print!("{contents}");
    } else {
        println!("-- {relative} ({} bytes)", contents.len());
    }
}

fn write(dir: &Path, name: &str, contents: &str) -> std::io::Result<()> {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
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
    let mut private_wrappers = Vec::new();
    for n in PRIVATE_WRAPPER_SIZES {
        let decode = constraint_exporter::trace::generate_private_batch_wrapper_lean(n)
            .map_err(std::io::Error::other)?;
        let bridge = constraint_exporter::private_wrapper::generate_private_wrapper_bridge_lean(n)
            .map_err(std::io::Error::other)?;
        private_wrappers.push((n, decode, bridge));
    }
    let leaf =
        constraint_exporter::trace::generate_leaf_circuit_lean().map_err(std::io::Error::other)?;
    let leaf_bridge =
        constraint_exporter::leaf::generate_leaf_bridge_lean().map_err(std::io::Error::other)?;
    let leaf_built =
        constraint_exporter::built::generate_leaf_built_lean().map_err(std::io::Error::other)?;
    let mut public_wrappers = Vec::new();
    for n in PUBLIC_WRAPPER_SIZES {
        let decode = constraint_exporter::trace::generate_public_batch_wrapper_modules(n)
            .map_err(std::io::Error::other)?;
        let bridge = constraint_exporter::public_wrapper::generate_public_wrapper_bridge_modules(n)
            .map_err(std::io::Error::other)?;
        public_wrappers.push((n, decode, bridge));
    }

    write(&dir, "Gates.lean", &gates)?;
    write(&dir, "Poseidon2.lean", &poseidon2)?;
    write(&dir, "Poseidon2Prims.lean", &poseidon2_prims)?;
    write(&dir, "NullifierSelectCircuit.lean", &nullifier_select)?;
    write(&dir, "GadgetZooCircuit.lean", &gadget_zoo)?;
    write(&dir, "GadgetEdgeCasesCircuit.lean", &gadget_edge_cases)?;
    let bridge_dir = bridge_generated_dir();
    fs::create_dir_all(&bridge_dir)?;
    for (n, decode, bridge) in &private_wrappers {
        write(&dir, &format!("PrivateBatchWrapper{n}.lean"), decode)?;
        write(&bridge_dir, &format!("Wrapper{n}.lean"), bridge)?;
    }
    for (n, modules, bridge) in &public_wrappers {
        write_split(
            &dir,
            &format!("PublicBatchWrapper{n}"),
            &modules
                .iter()
                .map(|m| (m.relative.as_str(), m.contents.as_str()))
                .collect::<Vec<_>>(),
        )?;
        write_split(
            &bridge_dir,
            &format!("PublicWrapper{n}"),
            &bridge
                .iter()
                .map(|m| (m.relative.as_str(), m.contents.as_str()))
                .collect::<Vec<_>>(),
        )?;
    }
    write(&dir, "LeafCircuit.lean", &leaf)?;
    write(&bridge_dir, "Leaf.lean", &leaf_bridge)?;
    write(&bridge_dir, "LeafBuilt.lean", &leaf_built)?;

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
    for (n, decode, bridge) in &private_wrappers {
        println!("\n-- ===== PrivateBatchWrapper{n}.lean =====");
        print!("{decode}");
        println!("\n-- ===== Plonky2Bridge/Generated/Wrapper{n}.lean =====");
        print!("{bridge}");
    }
    for (n, modules, bridge) in &public_wrappers {
        println!(
            "\n-- ===== PublicBatchWrapper{n} ({} files) =====",
            modules.len()
        );
        for module in modules {
            print_exported(*n, &module.relative, &module.contents);
        }
        println!(
            "\n-- ===== Plonky2Bridge/Generated/PublicWrapper{n} ({} files) =====",
            bridge.len()
        );
        for module in bridge {
            print_exported(*n, &module.relative, &module.contents);
        }
    }
    println!("\n-- ===== LeafCircuit.lean =====");
    print!("{leaf}");
    println!("\n-- ===== Plonky2Bridge/Generated/Leaf.lean =====");
    print!("{leaf_bridge}");
    println!("\n-- ===== Plonky2Bridge/Generated/LeafBuilt.lean =====");
    print!("{leaf_built}");
    Ok(())
}
