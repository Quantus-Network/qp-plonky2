//! FRI proof types.
//!
//! Re-exports types from core.

// Re-export all FRI proof types from core
pub use qp_plonky2_core::fri_proof::{
    CompressedFriProof, CompressedFriQueryRounds, FriInitialTreeProof, FriProof, FriQueryRound,
    FriQueryStep,
};

// Re-export FriChallenges from core
pub use qp_plonky2_core::FriChallenges;
