//! Lazy Merkle tree implementation that stores only digests and cap, not leaves.
//!
//! This is useful for memory-constrained environments where leaves can be recomputed
//! on-demand from polynomial coefficients.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub use qp_plonky2_core::MerkleCap;

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::hash::merkle_tree::{capacity_up_to_mut, fill_digests_buf, merkle_tree_prove};
use crate::plonk::config::Hasher;
use crate::util::log2_strict;

/// A Merkle tree that stores only digests and cap, discarding leaves after construction.
///
/// This saves significant memory (~2GB for large circuits) at the cost of requiring
/// leaves to be recomputed when needed for FRI queries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DigestOnlyMerkleTree<F: RichField, H: Hasher<F>> {
    /// Number of leaves (needed for proof generation)
    pub num_leaves: usize,

    /// The digests in the tree (same layout as MerkleTree)
    pub digests: Vec<H::Hash>,

    /// The Merkle cap
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for DigestOnlyMerkleTree<F, H> {
    fn default() -> Self {
        Self {
            num_leaves: 0,
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

impl<F: RichField, H: Hasher<F>> DigestOnlyMerkleTree<F, H> {
    /// Build a digest-only Merkle tree from leaves, then discard the leaves.
    ///
    /// This computes all the hashes needed for the Merkle tree but does not
    /// retain the leaf data, saving significant memory.
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let num_leaves = leaves.len();
        let log2_leaves_len = log2_strict(num_leaves);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (num_leaves - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

        unsafe {
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        // leaves are dropped here, saving memory!

        Self {
            num_leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    /// Create a Merkle proof from a leaf index.
    ///
    /// Note: This only returns the sibling hashes. The caller must provide
    /// the actual leaf data separately (by recomputing it from coefficients).
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.num_leaves, cap_height, &self.digests);

        MerkleProof { siblings }
    }

    /// Returns the number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.num_leaves
    }

    /// Returns true if the tree has no leaves.
    pub fn is_empty(&self) -> bool {
        self.num_leaves == 0
    }

    /// Create a digest-only tree from an existing MerkleTree, dropping the leaves.
    ///
    /// This allows converting a regular MerkleTree to a lazy version after
    /// the leaves are no longer needed for direct access.
    pub fn from_merkle_tree(
        tree: &crate::hash::merkle_tree::MerkleTree<F, H>,
    ) -> Self {
        Self {
            num_leaves: tree.leaves.len(),
            digests: tree.digests.clone(),
            cap: tree.cap.clone(),
        }
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::merkle_tree::MerkleTree;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2_field::types::Sample;

    type F = GoldilocksField;
    type C = PoseidonGoldilocksConfig;
    type H = <C as GenericConfig<2>>::Hasher;

    #[test]
    fn test_digest_only_merkle_tree_matches_regular() {
        // Create random leaves
        let num_leaves = 256;
        let leaf_size = 16;
        let cap_height = 4;

        let leaves: Vec<Vec<F>> = (0..num_leaves)
            .map(|_| (0..leaf_size).map(|_| F::rand()).collect())
            .collect();

        // Build regular Merkle tree
        let regular_tree = MerkleTree::<F, H>::new(leaves.clone(), cap_height);

        // Build digest-only Merkle tree
        let digest_only_tree = DigestOnlyMerkleTree::<F, H>::new(leaves.clone(), cap_height);

        // Verify caps match
        assert_eq!(regular_tree.cap, digest_only_tree.cap);

        // Verify digests match
        assert_eq!(regular_tree.digests, digest_only_tree.digests);

        // Verify proofs match and are valid
        for i in [0, 1, 127, 128, 255] {
            let regular_proof = regular_tree.prove(i);
            let digest_only_proof = digest_only_tree.prove(i);

            assert_eq!(regular_proof, digest_only_proof);

            // Verify the proof is valid
            verify_merkle_proof_to_cap(leaves[i].clone(), i, &digest_only_tree.cap, &digest_only_proof)
                .expect("Proof should be valid");
        }
    }
}
