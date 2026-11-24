//! Hashing configuration to be used when building a circuit.
//!
//! This module defines a [`Hasher`] trait as well as its recursive
//! counterpart [`AlgebraicHasher`] for in-circuit hashing. It also
//! provides concrete configurations, one fully recursive leveraging
//! the Poseidon hash function both internally and natively, and one
//! mixing Poseidon internally and truncated Keccak externally.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Re-export core config types - these are the canonical definitions
pub use qp_plonky2_core::config::{
    GenericConfig, GenericHashOut, Hasher, KeccakGoldilocksConfig, PoseidonGoldilocksConfig,
};

use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOut, HashOutTarget, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;

/// Trait for algebraic hash functions, built from a permutation using the sponge construction.
/// This extends the base `Hasher` trait with circuit-building capabilities.
///
/// The provided `*_circuit` methods are the overwrite-mode sponge that `PoseidonHash` uses
/// natively. A hasher whose native [`Hasher`] methods use a different sponge (padding, additive
/// absorption, ...) must override them so that in-circuit digests match native ones; generic
/// gadgets such as Merkle verification and the public-input hash only ever call these.
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

    /// In-circuit counterpart of [`Hasher::hash_no_pad`].
    fn hash_no_pad_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        inputs: Vec<Target>,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        HashOutTarget::from_vec(builder.hash_n_to_m_no_pad::<Self>(inputs, NUM_HASH_OUT_ELTS))
    }

    /// In-circuit counterpart of [`Hasher::hash_leaf`]: `len + 1` in the capacity lane
    /// domain-separates leaves from internal nodes and makes the digest injective in length.
    fn hash_leaf_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        inputs: Vec<Target>,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        let zero = builder.zero();
        let len_tag = builder.constant(F::from_canonical_usize(inputs.len() + 1));
        let mut state = Self::AlgebraicPermutation::new(core::iter::repeat(zero));
        state.set_elt(len_tag, Self::AlgebraicPermutation::RATE);

        for input_chunk in inputs.chunks(Self::AlgebraicPermutation::RATE) {
            state.set_from_slice(input_chunk, 0);
            state = builder.permute::<Self>(state);
        }

        HashOutTarget::from_vec(state.squeeze()[..NUM_HASH_OUT_ELTS].to_vec())
    }

    /// In-circuit counterpart of [`Hasher::two_to_one`] on `(left, right)`, or on
    /// `(right, left)` when `swap` is set.
    fn two_to_one_swapped_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D>,
        left: HashOutTarget,
        right: HashOutTarget,
        swap: BoolTarget,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        let zero = builder.zero();
        let mut perm_inputs = Self::AlgebraicPermutation::default();
        perm_inputs.set_from_slice(&left.elements, 0);
        perm_inputs.set_from_slice(&right.elements, NUM_HASH_OUT_ELTS);
        perm_inputs.set_from_iter(core::iter::repeat(zero), 2 * NUM_HASH_OUT_ELTS);
        let perm_outs = Self::permute_swapped(perm_inputs, swap, builder);
        HashOutTarget::from_vec(perm_outs.squeeze()[..NUM_HASH_OUT_ELTS].to_vec())
    }
}
