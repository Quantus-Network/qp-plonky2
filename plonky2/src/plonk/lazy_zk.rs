//! Lazy masked polynomial helpers for memory-efficient PLONK commitments.
//!
//! This module provides lazy versions of the polynomial commitment types that
//! don't store LDE values, saving significant memory at the cost of recomputation.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::lazy_oracle::LazyPolynomialBatch;
use crate::hash::hash_types::RichField;
use crate::plonk::config::GenericConfig;
use crate::plonk::zk::LogicalPolynomialLayout;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{cached_point_power, log2_strict};

/// A lazy polynomial commitment that stores only coefficients and Merkle digests.
///
/// This is the lazy equivalent of `LogicalPolynomialBatch`, saving memory by not
/// storing LDE values (leaves). LDE values are recomputed on-demand.
#[derive(Debug)]
pub struct LazyLogicalPolynomialBatch<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub raw: LazyPolynomialBatch<F, C, D>,
    pub logical_layouts: Vec<LogicalPolynomialLayout>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for LazyLogicalPolynomialBatch<F, C, D>
{
    fn default() -> Self {
        Self {
            raw: LazyPolynomialBatch::default(),
            logical_layouts: Vec::new(),
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    LazyLogicalPolynomialBatch<F, C, D>
{
    /// Evaluate a logical polynomial at a point.
    fn logical_eval_with_point_powers(
        &self,
        logical_index: usize,
        point: F::Extension,
        point_power_cache: &mut Vec<(usize, F::Extension)>,
    ) -> F::Extension {
        match &self.logical_layouts[logical_index] {
            LogicalPolynomialLayout::Raw { raw_index } => self.raw.polynomials[*raw_index]
                .to_extension::<D>()
                .eval(point),
            LogicalPolynomialLayout::SplitMask {
                low_index,
                high_index,
                split_power,
            } => {
                let low = self.raw.polynomials[*low_index]
                    .to_extension::<D>()
                    .eval(point);
                let high = self.raw.polynomials[*high_index]
                    .to_extension::<D>()
                    .eval(point);
                low + cached_point_power(point, *split_power, point_power_cache) * high
            }
        }
    }

    pub fn logical_eval(&self, logical_index: usize, point: F::Extension) -> F::Extension {
        let mut point_power_cache = Vec::new();
        self.logical_eval_with_point_powers(logical_index, point, &mut point_power_cache)
    }

    pub fn logical_evals(&self, point: F::Extension) -> Vec<F::Extension> {
        let mut point_power_cache = Vec::new();
        (0..self.logical_layouts.len())
            .map(|i| self.logical_eval_with_point_powers(i, point, &mut point_power_cache))
            .collect()
    }

    pub fn raw_polys_len(&self) -> usize {
        self.raw.polynomials.len()
    }

    /// Recompute all LDE leaves from coefficients.
    /// 
    /// This is called during quotient computation when we need access to all LDE values.
    pub fn recompute_lde_leaves(&self, fft_root_table: Option<&FftRootTable<F>>) -> Vec<Vec<F>> {
        self.raw.recompute_lde_leaves(fft_root_table)
    }

    /// Get the Merkle cap for the commitment.
    pub fn cap(&self) -> &crate::hash::lazy_merkle_tree::MerkleCap<F, C::Hasher> {
        self.raw.cap()
    }

    /// Materialize this lazy batch into a regular LogicalPolynomialBatch.
    ///
    /// This recomputes all LDE values and creates a regular batch with stored leaves.
    /// Use this when you need repeated random access to LDE values (e.g., quotient computation).
    ///
    /// This is the key method for integration: commit lazily, then materialize when needed.
    pub fn materialize(
        self,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> crate::plonk::zk::LogicalPolynomialBatch<F, C, D> {
        use crate::fri::oracle::PolynomialBatch;
        use crate::hash::merkle_tree::MerkleTree;

        // Recompute LDE leaves
        let leaves = timed!(
            timing,
            "recompute LDE for materialize",
            self.raw.recompute_lde_leaves(fft_root_table)
        );

        // Create a regular MerkleTree with the recomputed leaves
        // We already have the digests and cap, but MerkleTree::new recomputes them
        // For efficiency, we could create a constructor that takes existing digests,
        // but for the PoC we'll just rebuild
        let merkle_tree = timed!(
            timing,
            "rebuild MerkleTree with leaves",
            MerkleTree::new(leaves, log2_strict(self.raw.merkle_tree.cap.len()))
        );

        // Verify the cap matches (sanity check)
        debug_assert_eq!(merkle_tree.cap, self.raw.merkle_tree.cap);

        let raw = PolynomialBatch {
            polynomials: self.raw.polynomials,
            merkle_tree,
            degree_log: self.raw.degree_log,
            rate_bits: self.raw.rate_bits,
            blinding: self.raw.blinding,
        };

        crate::plonk::zk::LogicalPolynomialBatch {
            raw,
            logical_layouts: self.logical_layouts,
        }
    }

    /// Create a lazy batch from a regular LogicalPolynomialBatch, dropping the leaves.
    ///
    /// This allows converting to lazy storage after quotient computation is done,
    /// freeing memory while retaining the ability to answer FRI queries.
    pub fn from_logical_batch(
        batch: crate::plonk::zk::LogicalPolynomialBatch<F, C, D>,
    ) -> Self {
        Self {
            raw: LazyPolynomialBatch::from_polynomial_batch(batch.raw),
            logical_layouts: batch.logical_layouts,
        }
    }
}

/// Commit polynomial values using the lazy approach (no LDE storage).
pub fn commit_values_lazy<F, C, const D: usize>(
    values: Vec<PolynomialValues<F>>,
    target_len: Option<usize>,
    rate_bits: usize,
    leaf_hiding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    fft_root_table: Option<&FftRootTable<F>>,
) -> LazyLogicalPolynomialBatch<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let coeffs = timed!(
        timing,
        "IFFT for lazy commit",
        values
            .into_iter()
            .map(PolynomialValues::ifft)
            .collect::<Vec<_>>()
    );

    commit_coeffs_lazy(
        coeffs,
        target_len,
        rate_bits,
        leaf_hiding,
        cap_height,
        timing,
        fft_root_table,
    )
}

/// Commit polynomial coefficients using the lazy approach (no LDE storage).
pub fn commit_coeffs_lazy<F, C, const D: usize>(
    coeffs: Vec<PolynomialCoeffs<F>>,
    target_len: Option<usize>,
    rate_bits: usize,
    leaf_hiding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    fft_root_table: Option<&FftRootTable<F>>,
) -> LazyLogicalPolynomialBatch<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let logical_layouts = (0..coeffs.len())
        .map(|raw_index| LogicalPolynomialLayout::Raw { raw_index })
        .collect();

    let target_len = target_len.unwrap_or_else(|| {
        coeffs
            .iter()
            .map(|poly| poly.len())
            .max()
            .expect("lazy commitment requires at least one polynomial")
            .next_power_of_two()
    });

    let mut coeffs = coeffs;
    for poly in &mut coeffs {
        poly.pad(target_len).unwrap();
    }

    let raw = LazyPolynomialBatch::from_coeffs(
        coeffs,
        rate_bits,
        leaf_hiding,
        cap_height,
        timing,
        fft_root_table,
    );

    LazyLogicalPolynomialBatch {
        raw,
        logical_layouts,
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::zk::commit_values_with_split_mask;
    use crate::util::timing::TimingTree;
    use plonky2_field::types::Sample;

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    #[test]
    fn test_lazy_logical_batch_matches_regular() {
        let mut timing = TimingTree::default();

        // Create random polynomial values
        let num_polys = 16;
        let degree = 64;
        let rate_bits = 3;
        let cap_height = 2;
        let blinding = false;

        let values: Vec<PolynomialValues<F>> = (0..num_polys)
            .map(|_| PolynomialValues::new((0..degree).map(|_| F::rand()).collect()))
            .collect();

        // Build regular batch
        let regular_batch = commit_values_with_split_mask::<F, C, D>(
            values.clone(),
            None, // no mask plan
            None,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Build lazy batch
        let lazy_batch = commit_values_lazy::<F, C, D>(
            values,
            None,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Verify caps match
        assert_eq!(regular_batch.raw.merkle_tree.cap, lazy_batch.raw.merkle_tree.cap);

        // Verify digests match
        assert_eq!(regular_batch.raw.merkle_tree.digests, lazy_batch.raw.merkle_tree.digests);

        // Verify recomputed LDE values match
        let recomputed_leaves = lazy_batch.recompute_lde_leaves(None);
        assert_eq!(regular_batch.raw.merkle_tree.leaves, recomputed_leaves);

        // Verify logical evals match at a random base field point
        use crate::field::extension::{Extendable, FieldExtension};
        type FE = <F as Extendable<D>>::Extension;
        let x = F::rand();
        let test_point: FE = FieldExtension::<D>::from_basefield(x);
        let regular_evals = regular_batch.logical_evals(test_point);
        let lazy_evals = lazy_batch.logical_evals(test_point);
        assert_eq!(regular_evals, lazy_evals);
    }

    #[test]
    fn test_lazy_materialize() {
        let mut timing = TimingTree::default();

        // Create random polynomial values
        let num_polys = 16;
        let degree = 64;
        let rate_bits = 3;
        let cap_height = 2;
        let blinding = false;

        let values: Vec<PolynomialValues<F>> = (0..num_polys)
            .map(|_| PolynomialValues::new((0..degree).map(|_| F::rand()).collect()))
            .collect();

        // Build regular batch directly
        let regular_batch = commit_values_with_split_mask::<F, C, D>(
            values.clone(),
            None,
            None,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Build lazy batch and then materialize
        let lazy_batch = commit_values_lazy::<F, C, D>(
            values,
            None,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Materialize the lazy batch
        let materialized_batch = lazy_batch.materialize(&mut timing, None);

        // Verify the materialized batch matches the regular batch
        assert_eq!(regular_batch.raw.merkle_tree.cap, materialized_batch.raw.merkle_tree.cap);
        assert_eq!(regular_batch.raw.merkle_tree.digests, materialized_batch.raw.merkle_tree.digests);
        assert_eq!(regular_batch.raw.merkle_tree.leaves, materialized_batch.raw.merkle_tree.leaves);

        // Verify get_lde_values works correctly on the materialized batch
        let lde_size = degree << rate_bits;
        for i in [0, 1, lde_size / 2, lde_size - 1] {
            let regular_lde = regular_batch.raw.get_lde_values(i, 1);
            let materialized_lde = materialized_batch.raw.get_lde_values(i, 1);
            assert_eq!(regular_lde, materialized_lde);
        }
    }

    #[test]
    fn test_memory_savings_with_lazy() {
        let mut timing = TimingTree::default();

        // Use larger parameters to see meaningful memory differences
        let num_polys = 64;
        let degree = 1 << 12; // 4K rows
        let rate_bits = 3;
        let cap_height = 4;
        let blinding = false;

        let values: Vec<PolynomialValues<F>> = (0..num_polys)
            .map(|_| PolynomialValues::new((0..degree).map(|_| F::rand()).collect()))
            .collect();

        // Build lazy batch (memory efficient during this phase)
        let lazy_batch = commit_values_lazy::<F, C, D>(
            values.clone(),
            None,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // At this point, we've committed without storing all LDE values
        // The lazy_batch has the cap and digests but no leaves

        // Calculate theoretical memory savings
        let lde_size = degree << rate_bits;
        let leaf_size = num_polys * 8; // 8 bytes per field element
        let total_leaves_memory = lde_size * leaf_size;
        let coeffs_memory = num_polys * degree * 8;
        let digests_memory = lazy_batch.raw.merkle_tree.digests.len() * 32;

        eprintln!("\n=== Memory Analysis ===");
        eprintln!("LDE size: {} points", lde_size);
        eprintln!("Theoretical leaves memory: {} KB", total_leaves_memory / 1024);
        eprintln!("Coefficients memory: {} KB", coeffs_memory / 1024);
        eprintln!("Digests memory: {} KB", digests_memory / 1024);
        eprintln!("Lazy batch stores: coeffs + digests = {} KB", (coeffs_memory + digests_memory) / 1024);
        eprintln!("Regular batch would store: coeffs + digests + leaves = {} KB", 
            (coeffs_memory + digests_memory + total_leaves_memory) / 1024);
        eprintln!("Memory savings: {} KB ({:.1}%)",
            total_leaves_memory / 1024,
            100.0 * total_leaves_memory as f64 / (coeffs_memory + digests_memory + total_leaves_memory) as f64);

        // Now materialize when we need to access LDE values
        let materialized = lazy_batch.materialize(&mut timing, None);

        // Verify it works correctly
        let regular_batch = commit_values_with_split_mask::<F, C, D>(
            values,
            None,
            None,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        assert_eq!(regular_batch.raw.merkle_tree.cap, materialized.raw.merkle_tree.cap);
    }
}
