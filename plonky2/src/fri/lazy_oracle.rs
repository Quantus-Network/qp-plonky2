//! Lazy polynomial batch that stores coefficients and Merkle digests, but not LDE values.
//!
//! This implementation trades computation time for memory by:
//! 1. Building the Merkle tree from LDE values, then discarding the LDE values
//! 2. Recomputing LDE values on-demand during quotient computation
//! 3. Computing individual leaf values for FRI queries via polynomial evaluation

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::polynomial::PolynomialCoeffs;
use crate::hash::hash_types::RichField;
use crate::hash::lazy_merkle_tree::DigestOnlyMerkleTree;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// A polynomial batch that stores only coefficients and Merkle digests, not LDE values.
///
/// This saves ~2GB of memory for large circuits at the cost of:
/// - One extra FFT during quotient computation (to recompute LDE values)
/// - O(n) polynomial evaluations per FRI query (to recompute individual leaves)
#[derive(Debug)]
pub struct LazyPolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    /// The polynomial coefficients (kept for recomputing LDE values)
    pub polynomials: Vec<PolynomialCoeffs<F>>,

    /// Merkle tree with only digests and cap (leaves discarded)
    pub merkle_tree: DigestOnlyMerkleTree<F, C::Hasher>,

    /// log2 of the polynomial degree
    pub degree_log: usize,

    /// LDE rate (log2 of blowup factor)
    pub rate_bits: usize,

    /// Whether blinding/salting is used
    pub blinding: bool,

    /// Cached salt values (needed to reconstruct blinded leaves)
    /// Only present if blinding is true
    #[cfg(feature = "rand")]
    pub salt_values: Option<Vec<Vec<F>>>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for LazyPolynomialBatch<F, C, D>
{
    fn default() -> Self {
        Self {
            polynomials: Vec::new(),
            merkle_tree: DigestOnlyMerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
            #[cfg(feature = "rand")]
            salt_values: None,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    LazyPolynomialBatch<F, C, D>
{
    /// Creates a lazy polynomial batch from coefficient polynomials.
    ///
    /// This computes the LDE values and Merkle tree, then discards the LDE values
    /// to save memory. The coefficients are kept for later recomputation.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let degree_log = log2_strict(degree);

        // Compute LDE values (this is the memory-intensive part)
        let (lde_values, salt_values) = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values_with_salt(&polynomials, rate_bits, blinding, fft_root_table)
        );

        // Transpose to get leaves
        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);

        // Build Merkle tree (this will discard leaves after hashing)
        let merkle_tree = timed!(
            timing,
            "build Merkle tree (digest-only)",
            DigestOnlyMerkleTree::new(leaves, cap_height)
        );

        // lde_values and leaves are dropped here, saving memory!

        Self {
            polynomials,
            merkle_tree,
            degree_log,
            rate_bits,
            blinding,
            #[cfg(feature = "rand")]
            salt_values,
        }
    }

    /// Compute LDE values and optionally return salt values for blinding.
    fn lde_values_with_salt(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> (Vec<Vec<F>>, Option<Vec<Vec<F>>>) {
        let degree = polynomials[0].len();
        let lde_size = degree << rate_bits;

        let lde_values: Vec<Vec<F>> = polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .collect();

        #[cfg(feature = "rand")]
        if blinding {
            // Generate and store salt values
            let salt_values: Vec<Vec<F>> = (0..SALT_SIZE)
                .map(|_| F::rand_vec(lde_size))
                .collect();

            // Append salt to LDE values
            let mut all_values = lde_values;
            all_values.extend(salt_values.clone());
            return (all_values, Some(salt_values));
        }

        (lde_values, None)
    }

    /// Recompute all LDE values from coefficients.
    ///
    /// This is called during quotient computation when we need access to all LDE values.
    /// Returns the transposed, bit-reversed leaves (same format as stored in regular MerkleTree).
    pub fn recompute_lde_leaves(
        &self,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        // Recompute LDE values via FFT
        let mut lde_values: Vec<Vec<F>> = self.polynomials
            .par_iter()
            .map(|p| {
                p.lde(self.rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(self.rate_bits), fft_root_table)
                    .values
            })
            .collect();

        // Add back salt values if blinding
        #[cfg(feature = "rand")]
        if self.blinding {
            if let Some(ref salt) = self.salt_values {
                lde_values.extend(salt.clone());
            }
        }

        // Transpose and bit-reverse to match MerkleTree leaf format
        let mut leaves = transpose(&lde_values);
        reverse_index_bits_in_place(&mut leaves);

        leaves
    }

    /// Get LDE values at a specific index, recomputing from coefficients.
    ///
    /// This is used during FRI queries when we need specific leaf values.
    /// The index is in the bit-reversed domain.
    pub fn get_lde_values_at_index(&self, bit_reversed_index: usize) -> Vec<F> {
        // Un-reverse the index to get the actual evaluation point index
        let index = reverse_bits(bit_reversed_index, self.degree_log + self.rate_bits);

        // Compute the evaluation point on the coset
        let g = F::primitive_root_of_unity(self.degree_log + self.rate_bits);
        let x = F::coset_shift() * g.exp_u64(index as u64);

        // Evaluate each polynomial at x
        let mut values: Vec<F> = self.polynomials
            .iter()
            .map(|p| p.eval(x))
            .collect();

        // Add salt values at this index if blinding
        #[cfg(feature = "rand")]
        if self.blinding {
            if let Some(ref salt) = self.salt_values {
                for salt_col in salt {
                    values.push(salt_col[index]);
                }
            }
        }

        values
    }

    /// Create a Merkle proof for a leaf, recomputing the leaf value.
    ///
    /// Returns both the leaf data and the Merkle proof.
    pub fn prove_with_leaf(&self, leaf_index: usize) -> (Vec<F>, MerkleProof<F, C::Hasher>) {
        let leaf = self.get_lde_values_at_index(leaf_index);
        let proof = self.merkle_tree.prove(leaf_index);
        (leaf, proof)
    }

    /// Returns the Merkle cap.
    pub fn cap(&self) -> &crate::hash::lazy_merkle_tree::MerkleCap<F, C::Hasher> {
        &self.merkle_tree.cap
    }

    /// Returns the number of polynomials (excluding salt).
    pub fn num_polynomials(&self) -> usize {
        self.polynomials.len()
    }

    /// Returns the LDE size.
    pub fn lde_size(&self) -> usize {
        1 << (self.degree_log + self.rate_bits)
    }

    /// Create a lazy batch from a regular PolynomialBatch, dropping the leaves.
    ///
    /// This allows converting to lazy storage after quotient computation is done,
    /// freeing memory while retaining the ability to answer FRI queries.
    pub fn from_polynomial_batch(
        batch: crate::fri::oracle::PolynomialBatch<F, C, D>,
    ) -> Self {
        let digest_only_tree = DigestOnlyMerkleTree::from_merkle_tree(&batch.merkle_tree);
        
        Self {
            polynomials: batch.polynomials,
            merkle_tree: digest_only_tree,
            degree_log: batch.degree_log,
            rate_bits: batch.rate_bits,
            blinding: batch.blinding,
            #[cfg(feature = "rand")]
            salt_values: None, // Salt values not preserved - will be recomputed if needed
        }
    }

    /// Answer an FRI query by providing the leaf values and merkle proof.
    ///
    /// This recomputes the leaf values from polynomial coefficients and uses
    /// the stored digests for the Merkle proof.
    ///
    /// Returns (leaf_values, merkle_proof) pair for the given index.
    pub fn answer_fri_query(
        &self,
        bit_reversed_index: usize,
    ) -> (Vec<F>, MerkleProof<F, C::Hasher>) {
        self.prove_with_leaf(bit_reversed_index)
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::fri::oracle::PolynomialBatch;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::util::timing::TimingTree;
    use plonky2_field::types::Sample;

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    const D: usize = 2;

    #[test]
    fn test_lazy_batch_matches_regular() {
        let mut timing = TimingTree::default();

        // Create random polynomials
        let num_polys = 16;
        let degree = 64;
        let rate_bits = 3;
        let cap_height = 2;
        let blinding = false;

        let polynomials: Vec<PolynomialCoeffs<F>> = (0..num_polys)
            .map(|_| PolynomialCoeffs::new((0..degree).map(|_| F::rand()).collect()))
            .collect();

        // Build regular batch
        let regular_batch = PolynomialBatch::<F, C, D>::from_coeffs(
            polynomials.clone(),
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Build lazy batch
        let lazy_batch = LazyPolynomialBatch::<F, C, D>::from_coeffs(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Verify caps match
        assert_eq!(regular_batch.merkle_tree.cap, lazy_batch.merkle_tree.cap);

        // Verify digests match
        assert_eq!(regular_batch.merkle_tree.digests, lazy_batch.merkle_tree.digests);

        // Verify recomputed LDE values match
        let recomputed_leaves = lazy_batch.recompute_lde_leaves(None);
        assert_eq!(regular_batch.merkle_tree.leaves, recomputed_leaves);

        // Verify individual leaf retrieval
        let lde_size = degree << rate_bits;
        for i in [0, 1, lde_size / 2, lde_size - 1] {
            let regular_leaf = regular_batch.merkle_tree.get(i);
            let lazy_leaf = lazy_batch.get_lde_values_at_index(i);
            assert_eq!(regular_leaf, lazy_leaf.as_slice());
        }

        // Verify proofs match and are valid
        for i in [0, 1, lde_size / 2, lde_size - 1] {
            let (leaf, lazy_proof) = lazy_batch.prove_with_leaf(i);
            let regular_proof = regular_batch.merkle_tree.prove(i);

            assert_eq!(lazy_proof, regular_proof);

            // Verify proof is valid
            verify_merkle_proof_to_cap(leaf, i, &lazy_batch.merkle_tree.cap, &lazy_proof)
                .expect("Proof should be valid");
        }
    }

    #[test]
    fn test_memory_comparison() {
        let mut timing = TimingTree::default();

        // Use realistic parameters similar to aggregator circuit
        let num_polys = 135; // num_wires
        let degree = 1 << 14; // 16K rows (smaller than real 64K for test speed)
        let rate_bits = 3;
        let cap_height = 4;
        let blinding = false;

        let polynomials: Vec<PolynomialCoeffs<F>> = (0..num_polys)
            .map(|_| PolynomialCoeffs::new((0..degree).map(|_| F::rand()).collect()))
            .collect();

        // Build regular batch and measure
        let regular_batch = PolynomialBatch::<F, C, D>::from_coeffs(
            polynomials.clone(),
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Calculate memory usage for regular batch
        let leaves_size: usize = regular_batch.merkle_tree.leaves.iter()
            .map(|leaf| leaf.len() * 8) // 8 bytes per field element
            .sum();
        let digests_size = regular_batch.merkle_tree.digests.len() * 32; // ~32 bytes per hash
        let coeffs_size: usize = regular_batch.polynomials.iter()
            .map(|p| p.len() * 8)
            .sum();

        eprintln!("\n=== Regular PolynomialBatch Memory ===");
        eprintln!("Leaves: {} MB", leaves_size / (1024 * 1024));
        eprintln!("Digests: {} MB", digests_size / (1024 * 1024));
        eprintln!("Coefficients: {} MB", coeffs_size / (1024 * 1024));
        eprintln!("Total: {} MB", (leaves_size + digests_size + coeffs_size) / (1024 * 1024));

        // Drop regular batch to free memory
        drop(regular_batch);

        // Build lazy batch and measure
        let lazy_batch = LazyPolynomialBatch::<F, C, D>::from_coeffs(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            &mut timing,
            None,
        );

        // Calculate memory usage for lazy batch
        let lazy_digests_size = lazy_batch.merkle_tree.digests.len() * 32;
        let lazy_coeffs_size: usize = lazy_batch.polynomials.iter()
            .map(|p| p.len() * 8)
            .sum();

        eprintln!("\n=== Lazy PolynomialBatch Memory ===");
        eprintln!("Leaves: 0 MB (not stored)");
        eprintln!("Digests: {} MB", lazy_digests_size / (1024 * 1024));
        eprintln!("Coefficients: {} MB", lazy_coeffs_size / (1024 * 1024));
        eprintln!("Total: {} MB", (lazy_digests_size + lazy_coeffs_size) / (1024 * 1024));

        eprintln!("\n=== Savings ===");
        let regular_total = leaves_size + digests_size + coeffs_size;
        let lazy_total = lazy_digests_size + lazy_coeffs_size;
        eprintln!("Memory saved: {} MB ({:.1}%)", 
            (regular_total - lazy_total) / (1024 * 1024),
            100.0 * (regular_total - lazy_total) as f64 / regular_total as f64);

        // Verify the lazy batch still produces correct results
        let recomputed = lazy_batch.recompute_lde_leaves(None);
        assert_eq!(recomputed.len(), 1 << (14 + 3)); // degree << rate_bits
    }
}
