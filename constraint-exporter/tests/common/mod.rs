//! Checks shared by several test binaries; not every binary uses each one.
#![allow(dead_code)]

use constraint_exporter::gadget::{Call, Fact};
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::poseidon2::Poseidon2Hash;
use plonky2::iop::target::Target;
use plonky2::plonk::config::Hasher;

/// Every recorded fact holds numerically on the witness `val`.
pub fn assert_facts_hold(calls: &[Call], val: impl Fn(Target) -> F) {
    for c in calls {
        match c.fact {
            Fact::Select { b, x, y, out } => {
                assert_eq!(val(out), val(b) * (val(x) - val(y)) + val(y))
            }
            Fact::Not { b, out } => assert_eq!(val(out), F::ONE - val(b)),
            Fact::And { b1, b2, out } => assert_eq!(val(out), val(b1) * val(b2)),
            Fact::Or { b1, b2, out } => {
                assert_eq!(val(out), val(b1) + val(b2) - val(b1) * val(b2))
            }
            Fact::Add { x, y, out } => assert_eq!(val(out), val(x) + val(y)),
            Fact::Sub { x, y, out } => assert_eq!(val(out), val(x) - val(y)),
            Fact::Mul { x, y, out } => assert_eq!(val(out), val(x) * val(y)),
            Fact::AssertBool { b } => assert!(val(b) == F::ZERO || val(b) == F::ONE),
            Fact::IsEqual { x, y, equal, inv } => {
                let d = val(x) - val(y);
                assert_eq!(val(equal) * d, F::ZERO);
                assert_eq!(d * val(inv) - (F::ONE - val(equal)), F::ZERO);
            }
            Fact::RangeCheck { x, bits } => assert!(val(x).to_canonical_u64() < 1 << bits),
            Fact::Connect { x, y } => assert_eq!(val(x), val(y)),
            Fact::SplitLe { x, row, bits } => {
                let limbs: Vec<u64> = (1..=bits)
                    .map(|col| val(Target::wire(row, col)).to_canonical_u64())
                    .collect();
                assert!(limbs.iter().all(|&b| b < 2));
                let sum: u64 = limbs.iter().rev().fold(0, |acc, &b| 2 * acc + b);
                assert_eq!(val(x).to_canonical_u64(), sum);
            }
            Fact::Poseidon2 {
                ref rows,
                ref inputs,
            } => {
                let digest =
                    Poseidon2Hash::hash_no_pad(&inputs.iter().map(|&t| val(t)).collect::<Vec<_>>());
                let last = *rows.last().unwrap();
                for (i, d) in digest.elements.iter().enumerate() {
                    assert_eq!(val(Target::wire(last, 12 + i)), *d);
                }
            }
        }
    }
}

/// A split wrapper's `.generated-files` manifest is exactly the basenames the
/// generator owns in that directory. Regeneration deletes only those names.
pub fn assert_split_manifest(root: &str, relatives: &[String]) {
    let mut by_dir: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for relative in relatives {
        if let Some((dir, name)) = relative.split_once('/') {
            assert!(!name.contains('/'), "{relative}");
            by_dir
                .entry(dir.to_string())
                .or_default()
                .insert(name.to_string());
        }
    }
    for (dir, names) in by_dir {
        let mut expected = names.into_iter().collect::<Vec<_>>().join("\n");
        expected.push('\n');
        let path = format!(
            "{root}/{dir}/{}",
            constraint_exporter::lean_split::GENERATED_MANIFEST
        );
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(on_disk, expected, "run export-constraints ({path})");
    }
}
