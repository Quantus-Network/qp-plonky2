//! Selector information shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
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

    /// Partitions gates, given by degree in ascending order, into the fewest consecutive groups
    /// such that `|group| + max degree in group <= max_degree`, so that each group's filter
    /// polynomial times its gate's constraints stays within the quotient degree.
    pub fn for_gate_degrees(degrees: &[usize], max_degree: usize) -> Result<Self, &'static str> {
        let num_gates = degrees.len();
        let max_gate_degree = *degrees.iter().max().ok_or("no gates")?;
        if max_gate_degree.saturating_add(num_gates - 1) <= max_degree {
            #[allow(clippy::single_range_in_vec_init)]
            return Ok(Self {
                selector_indices: vec![0; num_gates],
                groups: vec![0..num_gates],
            });
        }
        if max_gate_degree >= max_degree {
            return Err("gate degree too high for the quotient degree factor");
        }

        let mut groups = Vec::new();
        let mut selector_indices = Vec::with_capacity(num_gates);
        let mut start = 0;
        while start < num_gates {
            let mut size = 0;
            while start + size < num_gates && size + degrees[start + size] < max_degree {
                size += 1;
            }
            selector_indices.resize(start + size, groups.len());
            groups.push(start..start + size);
            start += size;
        }
        Ok(Self {
            selector_indices,
            groups,
        })
    }
}
