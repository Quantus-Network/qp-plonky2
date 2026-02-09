//! Logic common to multiple IOPs.

pub mod ext_target;
pub mod target;
pub mod wire;

// Re-export Challenger from core
pub use qp_plonky2_core::challenger;
