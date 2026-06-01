//! Hashing configuration to be used when building a circuit.
//!
//! This module defines a [`Hasher`] trait as well as its recursive
//! counterpart [`AlgebraicHasher`] for in-circuit hashing. It also
//! provides concrete configurations, one fully recursive leveraging
//! the Poseidon hash function both internally and natively, and one
//! mixing Poseidon internally and truncated Keccak externally.

// Re-export core config types - these are the canonical definitions

pub use qp_plonky2_core::config::{
    merkle_node_hash_input, GenericConfig, GenericHashOut, Hasher, KeccakGoldilocksConfig,
    PoseidonGoldilocksConfig, MERKLE_LEAF_DOMAIN_TAG, MERKLE_NODE_DOMAIN_TAG,
};

use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOut, HashOutTarget, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;

/// Trait for algebraic hash functions, built from a permutation using the sponge construction.
/// This extends the base `Hasher` trait with circuit-building capabilities.
pub trait AlgebraicHasher<F: RichField>: Hasher<F, Hash = HashOut<F>> {
    type AlgebraicPermutation: PlonkyPermutation<Target>;

    /// Circuit to conditionally swap two chunks of the inputs (useful in verifying Merkle proofs),
    /// then apply the permutation.
    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>;

    fn hash_merkle_leaf_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        leaf_data: Vec<Target>,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        let mut encoded = Vec::with_capacity(leaf_data.len() + 2);
        encoded.push(builder.constant(F::from_canonical_u64(MERKLE_LEAF_DOMAIN_TAG)));
        encoded.push(builder.constant(F::from_canonical_usize(leaf_data.len())));
        encoded.extend_from_slice(&leaf_data);
        builder.hash_n_to_hash_no_pad::<Self>(encoded)
    }

    fn hash_merkle_node_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        left: HashOutTarget,
        right: HashOutTarget,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        let mut encoded = Vec::with_capacity(1 + 2 * NUM_HASH_OUT_ELTS);
        encoded.push(builder.constant(F::from_canonical_u64(MERKLE_NODE_DOMAIN_TAG)));
        encoded.extend_from_slice(&left.elements);
        encoded.extend_from_slice(&right.elements);
        builder.hash_n_to_hash_no_pad::<Self>(encoded)
    }

    fn hash_merkle_node_swapped_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        left_when_unswapped: HashOutTarget,
        right_when_unswapped: HashOutTarget,
        swap: BoolTarget,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        let left = HashOutTarget {
            elements: core::array::from_fn(|i| {
                builder.select(
                    swap,
                    right_when_unswapped.elements[i],
                    left_when_unswapped.elements[i],
                )
            }),
        };
        let right = HashOutTarget {
            elements: core::array::from_fn(|i| {
                builder.select(
                    swap,
                    left_when_unswapped.elements[i],
                    right_when_unswapped.elements[i],
                )
            }),
        };
        Self::hash_merkle_node_circuit(builder, left, right)
    }
}
