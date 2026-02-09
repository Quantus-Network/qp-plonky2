//! Core traits and types shared between plonky2 prover and verifier.
//!
//! This crate provides the foundational types and traits that both the
//! full plonky2 prover and the lightweight verifier depend on.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
pub extern crate alloc;

/// Re-export of `plonky2_field` for field types.
#[doc(inline)]
pub use plonky2_field as field;

// Core modules - order matters for dependencies
mod arch;
pub mod challenger;
pub mod config;
pub mod hash_types;
pub mod hashing;
pub mod keccak;
pub mod merkle_proofs;
pub mod merkle_tree;
pub mod poseidon;
pub mod poseidon_crandall;
pub mod poseidon_goldilocks;
pub mod reducing;
pub mod strided_view;

// Re-export key types at crate root for convenience
pub use challenger::Challenger;
pub use config::{
    GenericConfig, GenericHashOut, Hasher, KeccakGoldilocksConfig, PoseidonGoldilocksConfig,
};
pub use hash_types::{BytesHash, HashOut, RichField, NUM_HASH_OUT_ELTS};
pub use hashing::PlonkyPermutation;
pub use keccak::{KeccakHash, KeccakPermutation};
pub use merkle_proofs::MerkleProof;
pub use merkle_tree::{
    capacity_up_to_mut, fill_digests_buf, fill_subtree, merkle_tree_prove, MerkleCap, MerkleTree,
};
pub use poseidon::{
    Poseidon, PoseidonHash, PoseidonPermutation, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS,
    N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS, N_ROUNDS, SPONGE_CAPACITY, SPONGE_RATE, SPONGE_WIDTH,
};

/// The extension degree for the field extension (D=2 provides 100-bits of security)
pub const D: usize = 2;

/// The standard Plonky2 configuration using Poseidon hash over Goldilocks field
pub type C = PoseidonGoldilocksConfig;

/// The Goldilocks prime field
pub type F = field::goldilocks_field::GoldilocksField;
