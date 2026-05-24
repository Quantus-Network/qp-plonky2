//! Hashing configuration for verification.
//!
//! This module re-exports the core configuration traits and types
//! for use in the verifier crate.

// Re-export all config types from core
pub use qp_plonky2_core::config::{
    merkle_node_hash_input, GenericConfig, GenericHashOut, Hasher, KeccakGoldilocksConfig,
    PoseidonGoldilocksConfig, MERKLE_LEAF_DOMAIN_TAG, MERKLE_NODE_DOMAIN_TAG,
};
