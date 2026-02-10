//! Fast Reed-Solomon IOP (FRI) protocol.
//!
//! This module provides the FRI verifier implementation.

pub mod challenges;
pub mod proof;
pub mod structure;
pub(crate) mod validate_shape;
pub mod verifier;

// Re-export FRI types from core
pub use qp_plonky2_core::{FriConfig, FriParams, FriReductionStrategy};
