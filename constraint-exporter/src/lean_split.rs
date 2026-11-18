//! Split a generated Lean module that is too large to elaborate as one file.
//!
//! A production public-batch wrapper is tens of megabytes of per-call lemmas. One
//! `lean` process holding all of them runs out of memory, so the module is cut on
//! declaration boundaries into layers that import only what they need:
//! copy-list defs, the circuit, copy lemmas, per-call facts, then the decode
//! conjunction. The original module path becomes an import of that chain.

use std::fmt::Write as _;

pub struct LeanModule {
    /// Path relative to `formal/Plonky2Spec/Generated/`.
    pub relative: String,
    pub contents: String,
}

/// Modules larger than this are split. Leaf (~2.5MB) elaborates comfortably under it.
const MAX_MODULE_BYTES: usize = 2_000_000;

pub fn split_module(module: &str, src: &str) -> Vec<LeanModule> {
    split_module_max(module, src, MAX_MODULE_BYTES)
}

fn split_module_max(module: &str, src: &str, max_bytes: usize) -> Vec<LeanModule> {
    if src.len() <= max_bytes {
        return vec![LeanModule {
            relative: format!("{module}.lean"),
            contents: src.to_string(),
        }];
    }
    let imports = lines_with(src, "import ");
    let opens = src
        .lines()
        .filter(|l| l.starts_with("open "))
        .map(|l| l.to_string())
        .collect::<Vec<_>>();
    let namespace = src
        .lines()
        .find(|l| l.starts_with("namespace "))
        .unwrap_or_else(|| panic!("{module}: no namespace"))
        .to_string();
    let variable = src
        .lines()
        .find(|l| l.starts_with("variable "))
        .unwrap_or_else(|| panic!("{module}: no variable line"))
        .to_string();
    let body = body_after_variable(src, &variable);
    let blocks = declaration_blocks(&body);
    let mut layers: [Vec<String>; 5] = Default::default();
    for block in blocks {
        let layer = decl_layer(&block);
        layers[layer].push(block);
    }
    let layer_names = ["CopyData", "Circuit", "CopyLemma", "Fact", "Decode"];
    let mut files: Vec<LeanModule> = Vec::new();
    let mut prev: Option<String> = None;
    for (layer, blocks) in layers.iter().enumerate() {
        if blocks.is_empty() {
            continue;
        }
        let parts = pack(blocks, max_bytes);
        let count = parts.len();
        let stem = layer_names[layer];
        let deps: Vec<String> = prev.iter().cloned().collect();
        for (i, part) in parts.iter().enumerate() {
            let name = if count == 1 {
                stem.to_string()
            } else {
                format!("{stem}{i}")
            };
            files.push(LeanModule {
                relative: format!("{module}/{name}.lean"),
                contents: render_part(module, &imports, &opens, &namespace, &variable, &deps, part),
            });
        }
        let imported = if count == 1 {
            stem.to_string()
        } else {
            let barrel = render_barrel(
                module,
                &(0..count).map(|i| format!("{stem}{i}")).collect::<Vec<_>>(),
            );
            files.push(LeanModule {
                relative: format!("{module}/{stem}.lean"),
                contents: barrel,
            });
            stem.to_string()
        };
        prev = Some(imported);
    }
    let root_dep = prev.unwrap_or_else(|| panic!("{module}: nothing to split"));
    files.push(LeanModule {
        relative: format!("{module}.lean"),
        contents: render_root(module, &root_dep),
    });
    files
}

fn lines_with(src: &str, prefix: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| l.strip_prefix(prefix).map(|r| r.to_string()))
        .collect()
}

fn body_after_variable(src: &str, variable: &str) -> String {
    let marker = format!("{variable}\n");
    let at = src
        .find(&marker)
        .unwrap_or_else(|| panic!("variable line not found as its own line"));
    let mut body = &src[at + marker.len()..];
    if let Some(end) = body.rfind("\nend ") {
        body = &body[..end + 1];
    }
    body.to_string()
}

fn is_decl(line: &str) -> bool {
    line.starts_with("def ")
        || line.starts_with("noncomputable def ")
        || line.starts_with("theorem ")
        || line.starts_with("lemma ")
}

fn is_prelude(line: &str) -> bool {
    line.starts_with("set_option ")
        || line.starts_with("/--")
        || line.starts_with("/-!")
        || line.starts_with("/-")
}

fn declaration_blocks(body: &str) -> Vec<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut has_decl = false;
    for line in body.lines() {
        if (is_decl(line) || is_prelude(line)) && has_decl {
            blocks.push(std::mem::take(&mut cur));
            has_decl = false;
        }
        cur.push_str(line);
        cur.push('\n');
        if is_decl(line) {
            has_decl = true;
        }
    }
    if !cur.trim().is_empty() {
        if has_decl || blocks.is_empty() {
            blocks.push(cur);
        } else {
            blocks.last_mut().unwrap().push_str(&cur);
        }
    }
    blocks
}

/// 0 copy-list defs, 1 circuit and other defs, 2 copy/const lemmas, 3 per-call facts,
/// 4 the decode conjunction.
fn decl_layer(block: &str) -> usize {
    for line in block.lines() {
        let rest = line
            .strip_prefix("noncomputable def ")
            .or_else(|| line.strip_prefix("def "))
            .or_else(|| line.strip_prefix("theorem "))
            .or_else(|| line.strip_prefix("lemma "));
        let Some(rest) = rest else { continue };
        let name = rest.split_whitespace().next().unwrap_or("");
        if name.contains(".copies") {
            return 0;
        }
        if name.contains("_copies") || name.ends_with("_consts") {
            return 2;
        }
        if let Some(suffix) = name.rsplit("_f").next() {
            if name.contains("_f")
                && !suffix.is_empty()
                && suffix.chars().all(|c| c.is_ascii_digit())
                && line.starts_with("theorem ")
            {
                return 3;
            }
        }
        if name.ends_with("_decode") {
            return 4;
        }
        return 1;
    }
    1
}

fn pack(blocks: &[String], max_bytes: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut cur = String::new();
    for block in blocks {
        if !cur.is_empty() && cur.len() + block.len() > max_bytes {
            parts.push(std::mem::take(&mut cur));
        }
        cur.push_str(block);
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

fn render_part(
    module: &str,
    imports: &[String],
    opens: &[String],
    namespace: &str,
    variable: &str,
    deps: &[String],
    body: &str,
) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Part of `Plonky2Spec.Generated.{module}`, split out of the recorded wrapper\n\
         \x20 so each Lean module elaborates in bounded memory.\n\
         \x20 Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\
         -/\n"
    );
    for import in imports {
        let _ = writeln!(out, "import {import}");
    }
    for dep in deps {
        let _ = writeln!(out, "import Plonky2Spec.Generated.{module}.{dep}");
    }
    let _ = writeln!(out, "\n{namespace}\n");
    for open in opens {
        let _ = writeln!(out, "{open}");
    }
    if !opens.is_empty() {
        out.push('\n');
    }
    // A production circuit literal nests ~2^14 public inputs inside the same `where`
    // as the row list, and the verifier list is ~10^4 targets. Lean elaborates list
    // syntax with one stack frame per element, past the default 1024, and reducing the
    // chunked `copies` append exceeds the default heartbeat budget.
    let _ = writeln!(out, "set_option maxRecDepth 200000\n");
    if body.contains(" : Circuit ") {
        let _ = writeln!(out, "set_option maxHeartbeats 0\n");
    }
    let _ = writeln!(out, "set_option linter.all false\n");
    let _ = writeln!(out, "{variable}\n");
    out.push_str(body);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    let end = namespace
        .strip_prefix("namespace ")
        .unwrap_or("Plonky2Spec.Generated");
    let _ = writeln!(out, "end {end}");
    out
}

fn render_barrel(module: &str, parts: &[String]) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 Re-exports one layer of `Plonky2Spec.Generated.{module}`.\n\
         -/\n"
    );
    for part in parts {
        let _ = writeln!(out, "import Plonky2Spec.Generated.{module}.{part}");
    }
    out
}

/// Name of the manifest written into a split output directory. It lists the
/// basenames the generator owns. Regeneration deletes only names on the
/// previous manifest that this run no longer emits.
pub const GENERATED_MANIFEST: &str = ".generated-files";

/// Byte length above which a production public-batch module (`n > 4`) is
/// summarized on stdout. The recorded `n = 2` and `n = 4` modules stay printed
/// in full, which is what the exporter did before the production split.
pub const STDOUT_FULL_LIMIT: usize = 100_000;

/// Whether `export-constraints` prints `contents` of a public wrapper of arity `n`.
pub fn stdout_includes_source(wrapper_arity: usize, len: usize) -> bool {
    wrapper_arity <= 4 || len <= STDOUT_FULL_LIMIT
}

fn owned_basename(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && name != GENERATED_MANIFEST
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
}

fn read_manifest(path: &std::path::Path) -> std::io::Result<std::collections::BTreeSet<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(std::collections::BTreeSet::new())
        }
        Err(err) => Err(err),
    }
}

/// Write `files` into `dir` and delete generator-owned files this run no longer emits.
///
/// Ownership is the previous `.generated-files` manifest in `dir`. A name that is
/// not on that manifest is left alone, including notes and anything that is not a
/// regular file. A nested directory therefore neither disappears nor aborts the
/// caller. The manifest is rewritten to the basenames in `files`.
pub fn sync_owned_files(
    dir: &std::path::Path,
    files: &[(&str, &str)],
) -> std::io::Result<Vec<String>> {
    for (name, _) in files {
        if !owned_basename(name) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("refusing generated path {name}"),
            ));
        }
    }
    std::fs::create_dir_all(dir)?;
    let manifest_path = dir.join(GENERATED_MANIFEST);
    let owned = read_manifest(&manifest_path)?;
    let new_names: std::collections::BTreeSet<&str> = files.iter().map(|(name, _)| *name).collect();
    let mut removed = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name == GENERATED_MANIFEST || !entry.file_type()?.is_file() {
            continue;
        }
        if owned.contains(name) && !new_names.contains(name) {
            std::fs::remove_file(entry.path())?;
            removed.push(name.to_string());
        }
    }
    for (name, contents) in files {
        std::fs::write(dir.join(name), contents)?;
    }
    let manifest = new_names
        .iter()
        .copied()
        .fold(String::new(), |mut out, name| {
            out.push_str(name);
            out.push('\n');
            out
        });
    std::fs::write(manifest_path, manifest)?;
    Ok(removed)
}

fn render_root(module: &str, dep: &str) -> String {
    format!(
        "/-\n\
         \x20 AUTO-GENERATED — do not edit by hand.\n\n\
         \x20 The recorded wrapper `{module}` is split across `{module}/` so Lean can\n\
         \x20 elaborate it. Importing this module brings every part into scope.\n\
         \x20 Regenerate with:\n\n\
         \x20     cargo run -p qp-plonky2-constraint-exporter --bin export-constraints\n\
         -/\n\
         import Plonky2Spec.Generated.{module}.{dep}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> String {
        let mut s = String::from(
            "/-\n  header\n-/\n\
             import Mathlib.Tactic.IntervalCases\n\
             import Plonky2Spec.WiringGadgets\n\
             \n\
             namespace Plonky2Spec.Generated\n\
             \n\
             open Plonky2Spec.Wiring\n\
             \n\
             set_option linter.all false\n\
             \n\
             variable {p : ℕ} [Fact p.Prime]\n\
             \n",
        );
        for i in 0..4 {
            let _ = writeln!(
                s,
                "def demo.copies{i} : List (Target × Target) := [\n  (.virt {i}, .virt 0)\n  ]\n"
            );
        }
        s.push_str("def demo (p : ℕ) : Circuit p where\n  rows := []\n\n");
        s.push_str(
            "theorem demo_copies0 (a : Assignment p) (h : Satisfies (demo p) a) :\n    True := by\n  trivial\n\n",
        );
        s.push_str(
            "theorem demo_consts (a : Assignment p) (h : Satisfies (demo p) a) :\n    True := by\n  trivial\n\n",
        );
        s.push_str(
            "theorem demo_f0 (a : Assignment p) (h : Satisfies (demo p) a) :\n    True := by\n  trivial\n\n",
        );
        s.push_str(
            "theorem demo_f1 (a : Assignment p) (h : Satisfies (demo p) a) :\n    True := by\n  trivial\n\n",
        );
        s.push_str(
            "set_option maxHeartbeats 20 in\n\
             theorem demo_decode (a : Assignment p) (h : Satisfies (demo p) a) :\n    True := by\n  trivial\n\n\
             end Plonky2Spec.Generated\n",
        );
        s
    }

    #[test]
    fn small_module_is_kept_whole() {
        let src = "theorem t : True := trivial\n";
        let files = split_module_max("Demo", src, 10_000);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative, "Demo.lean");
        assert_eq!(files[0].contents, src);
    }

    #[test]
    fn large_module_splits_by_layer() {
        let src = sample();
        let files = split_module_max("Demo", &src, 80);
        let names: Vec<_> = files.iter().map(|f| f.relative.as_str()).collect();
        assert!(names.contains(&"Demo.lean"), "{names:?}");
        assert!(
            names.iter().any(|n| n.starts_with("Demo/CopyData")),
            "{names:?}"
        );
        assert!(names.contains(&"Demo/Circuit.lean"), "{names:?}");
        assert!(names.iter().any(|n| n.contains("CopyLemma")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("Fact")), "{names:?}");
        assert!(names.iter().any(|n| n.contains("Decode")), "{names:?}");
        let root = files.iter().find(|f| f.relative == "Demo.lean").unwrap();
        assert!(root
            .contents
            .contains("import Plonky2Spec.Generated.Demo.Decode"));
        let circuit = files
            .iter()
            .find(|f| f.relative == "Demo/Circuit.lean")
            .unwrap();
        assert!(circuit.contents.contains("def demo (p : ℕ)"));
        assert!(circuit
            .contents
            .contains("import Plonky2Spec.Generated.Demo.CopyData"));
        let decode = files
            .iter()
            .find(|f| f.relative.contains("Decode") && f.relative.ends_with(".lean"))
            .unwrap();
        assert!(decode.contents.contains("theorem demo_decode"));
        assert!(decode.contents.contains("set_option maxHeartbeats 20 in"));
        assert!(!decode.contents.contains("theorem demo_f0"));
    }

    #[test]
    fn recorded_shapes_stay_on_stdout() {
        assert!(stdout_includes_source(2, 205_239));
        assert!(stdout_includes_source(4, 420_580));
        assert!(stdout_includes_source(89, 367));
        assert!(!stdout_includes_source(89, STDOUT_FULL_LIMIT + 1));
    }

    struct Tmp(std::path::PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tmp(label: &str) -> Tmp {
        let dir = std::env::temp_dir().join(format!(
            "qp-plonky2-sync-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Tmp(dir)
    }

    #[test]
    fn sync_keeps_unrelated_files_and_drops_only_owned_ones() {
        let dir = tmp("owned");
        std::fs::write(dir.0.join(GENERATED_MANIFEST), "Kept.lean\nStale.lean\n").unwrap();
        std::fs::write(dir.0.join("Kept.lean"), "old").unwrap();
        std::fs::write(dir.0.join("Stale.lean"), "stale").unwrap();
        std::fs::write(dir.0.join("local-notes.txt"), "notes").unwrap();
        std::fs::create_dir(dir.0.join("nested")).unwrap();
        std::fs::write(dir.0.join("nested").join("extra.txt"), "x").unwrap();

        let removed = sync_owned_files(&dir.0, &[("Kept.lean", "new")]).unwrap();
        assert_eq!(removed, vec!["Stale.lean".to_string()]);
        assert_eq!(
            std::fs::read_to_string(dir.0.join("Kept.lean")).unwrap(),
            "new"
        );
        assert!(!dir.0.join("Stale.lean").exists());
        assert_eq!(
            std::fs::read_to_string(dir.0.join("local-notes.txt")).unwrap(),
            "notes"
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join("nested").join("extra.txt")).unwrap(),
            "x"
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join(GENERATED_MANIFEST)).unwrap(),
            "Kept.lean\n"
        );
    }

    #[test]
    fn sync_without_a_manifest_does_not_delete_or_abort() {
        let dir = tmp("untracked");
        std::fs::write(dir.0.join("local-notes.txt"), "notes").unwrap();
        std::fs::write(dir.0.join("Old.lean"), "old").unwrap();
        std::fs::create_dir(dir.0.join("nested")).unwrap();
        std::fs::write(dir.0.join("nested").join("extra.txt"), "x").unwrap();

        let removed = sync_owned_files(&dir.0, &[("New.lean", "new")]).unwrap();
        assert!(removed.is_empty());
        assert_eq!(
            std::fs::read_to_string(dir.0.join("New.lean")).unwrap(),
            "new"
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join("local-notes.txt")).unwrap(),
            "notes"
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join("Old.lean")).unwrap(),
            "old"
        );
        assert!(dir.0.join("nested").is_dir());
    }
}
