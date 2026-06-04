//! Selector information shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ops::Range;

use serde::Serialize;

/// Placeholder value to indicate that a gate doesn't use a selector polynomial.
pub const UNUSED_SELECTOR: usize = u32::MAX as usize;

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct SelectorsInfo {
    pub selector_indices: Vec<usize>,
    pub groups: Vec<Range<usize>>,
}

impl SelectorsInfo {
    pub fn num_selectors(&self) -> usize {
        self.groups.len()
    }

    /// Validate selector metadata against the gate set and constant budget.
    ///
    /// Intended for deserialization of untrusted data: prevents out-of-bounds selector
    /// and group indexing and oversized selector-group loops during constraint evaluation.
    pub fn check_valid(
        &self,
        num_gates: usize,
        num_lookup_selectors: usize,
        num_constants: usize,
    ) -> Result<(), &'static str> {
        if self.selector_indices.len() != num_gates {
            return Err("selector_indices length does not match the number of gates");
        }
        for group in &self.groups {
            if group.start > group.end || group.end > num_gates {
                return Err("selector group range out of bounds");
            }
        }
        // A gate's selector index must point at the group that actually contains it: the
        // filter for gate `i` is built from `groups[selector_indices[i]]` with `i` as the
        // active row, and a group that excludes `i` mis-evaluates that filter (skipping the
        // gate's constraints). `compute_filter` relies on this via a release-stripped assert.
        for (gate_idx, &index) in self.selector_indices.iter().enumerate() {
            let group = self.groups.get(index).ok_or("selector index out of range")?;
            if !group.contains(&gate_idx) {
                return Err("selector group does not contain its gate");
            }
        }
        let prefix = self
            .groups
            .len()
            .checked_add(num_lookup_selectors)
            .ok_or("selector count overflows")?;
        if prefix > num_constants {
            return Err("selectors and lookup selectors exceed num_constants");
        }
        Ok(())
    }
}

/// Enum listing the different selectors for lookup constraints:
/// - `TransSre` is for Sum and RE transition constraints.
/// - `TransLdc` is for LDC transition constraints.
/// - `InitSre` is for the initial constraint of Sum and Re.
/// - `LastLdc` is for the final LDC (and Sum) constraint.
/// - `StartEnd` indicates where lookup end selectors begin.
pub enum LookupSelectors {
    TransSre = 0,
    TransLdc,
    InitSre,
    LastLdc,
    StartEnd,
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::SelectorsInfo;

    #[test]
    fn check_valid_accepts_containing_groups() {
        let info = SelectorsInfo {
            selector_indices: vec![0, 0, 1],
            groups: vec![0..2, 2..3],
        };
        assert!(info.check_valid(3, 0, 8).is_ok());
    }

    #[test]
    fn check_valid_rejects_group_not_containing_its_gate() {
        // Gate 2 is reassigned to group 0 (0..2), which does not contain index 2.
        let info = SelectorsInfo {
            selector_indices: vec![0, 0, 0],
            groups: vec![0..2, 2..3],
        };
        assert!(info.check_valid(3, 0, 8).is_err());
    }

    #[test]
    fn check_valid_rejects_out_of_range_index() {
        let info = SelectorsInfo {
            selector_indices: vec![0, 5],
            groups: vec![0..2],
        };
        assert!(info.check_valid(2, 0, 8).is_err());
    }
}
