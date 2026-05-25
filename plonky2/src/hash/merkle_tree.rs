#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use anyhow::{ensure, Result};
use plonky2_maybe_rayon::*;
// Re-export MerkleCap from core for unified type across crates
pub use qp_plonky2_core::{checked_merkle_cap_len, MerkleCap};

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::Hasher;
use crate::util::log2_strict;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: Vec<Vec<F>>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: Vec::new(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_merkle_leaf(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_merkle_leaf(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

#[allow(dead_code)]
pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    try_merkle_tree_prove::<F, H>(leaf_index, leaves_len, cap_height, digests)
        .expect("Merkle leaf index and tree metadata must be valid")
}

pub(crate) fn try_merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Result<Vec<H::Hash>> {
    ensure!(
        leaves_len.is_power_of_two(),
        "Merkle tree leaf count must be a power of two"
    );
    ensure!(leaf_index < leaves_len, "Merkle leaf index is out of range");
    let cap_len = checked_merkle_cap_len(cap_height)?;
    ensure!(
        cap_len <= leaves_len,
        "Merkle cap height exceeds leaf height"
    );

    let num_layers = log2_strict(leaves_len) - cap_height;
    let digest_len = leaves_len
        .checked_sub(cap_len)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("Merkle digest length overflow"))?;
    ensure!(
        digest_len == digests.len(),
        "Merkle digest length does not match leaf count and cap height"
    );

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        let tree_start = tree_len
            .checked_mul(tree_index)
            .ok_or_else(|| anyhow::anyhow!("Merkle digest tree index overflow"))?;
        let tree_end = tree_start
            .checked_add(tree_len)
            .ok_or_else(|| anyhow::anyhow!("Merkle digest tree range overflow"))?;
        ensure!(
            tree_end <= digests.len(),
            "Merkle digest tree range is out of bounds"
        );
        &digests[tree_start..tree_end]
    };

    // Mask out high bits to get the index within the sub-tree.
    let subtree_len = 1usize
        .checked_shl(num_layers.try_into().unwrap_or(usize::BITS))
        .ok_or_else(|| anyhow::anyhow!("Merkle subtree length overflow"))?;
    let mut pair_index = leaf_index & (subtree_len - 1);
    let siblings = (0..num_layers)
        .map(|i| -> Result<H::Hash> {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let shifted_pair_index = pair_index
                .checked_shl((i + 1).try_into().unwrap_or(usize::BITS))
                .ok_or_else(|| anyhow::anyhow!("Merkle sibling index overflow"))?;
            let layer_offset = 1usize
                .checked_shl(i.try_into().unwrap_or(usize::BITS))
                .and_then(|n| n.checked_sub(1))
                .ok_or_else(|| anyhow::anyhow!("Merkle sibling layer offset overflow"))?;
            let siblings_index = shifted_pair_index
                .checked_add(layer_offset)
                .ok_or_else(|| anyhow::anyhow!("Merkle sibling index overflow"))?;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = siblings_index
                .checked_mul(2)
                .and_then(|n| n.checked_add(1 - parity))
                .ok_or_else(|| anyhow::anyhow!("Merkle sibling index overflow"))?;
            digest_tree
                .get(sibling_index)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Merkle sibling index is out of bounds"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(siblings)
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn try_get(&self, i: usize) -> Result<&[F]> {
        self.leaves
            .get(i)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow::anyhow!("Merkle leaf index is out of range"))
    }

    /// Return a leaf by index.
    ///
    /// Panics if `i` is out of range. Use [`Self::try_get`] for untrusted indices.
    pub fn get(&self, i: usize) -> &[F] {
        self.try_get(i).expect("Merkle leaf index must be in range")
    }

    /// Create a Merkle proof from a leaf index.
    pub fn try_prove(&self, leaf_index: usize) -> Result<MerkleProof<F, H>> {
        ensure!(
            !self.cap.is_empty() && self.cap.len().is_power_of_two(),
            "Merkle cap must be non-empty and power-of-two sized"
        );
        let cap_height = log2_strict(self.cap.len());
        let siblings = try_merkle_tree_prove::<F, H>(
            leaf_index,
            self.leaves.len(),
            cap_height,
            &self.digests,
        )?;

        Ok(MerkleProof { siblings })
    }

    /// Create a Merkle proof from a leaf index.
    ///
    /// Panics if `leaf_index` is out of range. Use [`Self::try_prove`] for untrusted indices.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        self.try_prove(leaf_index)
            .expect("Merkle leaf index must be in range")
    }
}

#[cfg(test)]
#[cfg(feature = "rand")]
pub(crate) mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}
