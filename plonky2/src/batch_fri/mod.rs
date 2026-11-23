pub mod oracle;
pub mod prover;
pub mod recursive_verifier;
pub mod verifier;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::field::extension::Extendable;
use crate::fri::structure::{FriOpeningExpression, FriOracleInfo};
use crate::hash::hash_types::RichField;
use crate::plonk::plonk_common::salt_size;

/// How a batch-FRI initial-tree leaf is laid out for one oracle: one `polys ‖ salt` group per
/// instance, in instance order. Opening terms number polynomials globally across the groups, so
/// a term's leaf position is its global index plus the salts of every preceding group.
pub(crate) struct BatchLeafLayout {
    /// `(num_polys, salt_len)` per instance.
    groups: Vec<(usize, usize)>,
}

impl BatchLeafLayout {
    /// One layout per oracle, from each instance's oracle list (all instances share the oracles).
    pub(crate) fn per_oracle(
        oracles_per_instance: &[&[FriOracleInfo]],
        leaf_hiding: bool,
    ) -> Vec<Self> {
        let oracle_count = oracles_per_instance
            .first()
            .map_or(0, |oracles| oracles.len());
        (0..oracle_count)
            .map(|oracle_index| Self {
                groups: oracles_per_instance
                    .iter()
                    .map(|oracles| {
                        let oracle = oracles[oracle_index];
                        (oracle.num_polys, salt_size(leaf_hiding && oracle.blinding))
                    })
                    .collect(),
            })
            .collect()
    }

    /// The `(start, num_polys, salt_len)` of each instance's group within the leaf.
    fn groups(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.groups.iter().scan(0, |start, &(num_polys, salt)| {
            let group = (*start, num_polys, salt);
            *start += num_polys + salt;
            Some(group)
        })
    }

    /// The leaf slice holding each instance's group, salt included.
    pub(crate) fn group_ranges(&self) -> impl Iterator<Item = core::ops::Range<usize>> + '_ {
        self.groups()
            .map(|(start, num_polys, salt)| start..start + num_polys + salt)
    }

    fn position(&self, polynomial_index: usize) -> usize {
        let mut remaining = polynomial_index;
        for (start, num_polys, _) in self.groups() {
            if remaining < num_polys {
                return start + remaining;
            }
            remaining -= num_polys;
        }
        panic!("polynomial index {polynomial_index} exceeds the batch leaf")
    }
}

/// Re-indexes `openings` from global polynomial numbering to batch-leaf positions (one layout per
/// oracle), so the plain-FRI leaf accessors read the right evaluation even when groups are salted.
pub(crate) fn openings_at_leaf_positions<F: RichField + Extendable<D>, const D: usize>(
    openings: &[FriOpeningExpression<F, D>],
    layouts: &[BatchLeafLayout],
) -> Vec<FriOpeningExpression<F, D>> {
    openings
        .iter()
        .map(|expression| {
            let mut expression = expression.clone();
            for term in &mut expression.terms {
                let polynomial = &mut term.polynomial;
                polynomial.polynomial_index =
                    layouts[polynomial.oracle_index].position(polynomial.polynomial_index);
            }
            expression
        })
        .collect()
}
