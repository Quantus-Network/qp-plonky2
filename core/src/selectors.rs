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
        for &index in &self.selector_indices {
            if index >= self.groups.len() {
                return Err("selector index out of range");
            }
        }
        for group in &self.groups {
            if group.start > group.end || group.end > num_gates {
                return Err("selector group range out of bounds");
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
