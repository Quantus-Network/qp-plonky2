//! Information about the structure of a FRI instance, in terms of the oracles and polynomials
//! involved, and the points they are opened at.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::iop::ext_target::ExtensionTarget;

// Re-export base FRI structure types from core
pub use qp_plonky2_core::{
    FriBatchInfo, FriInstanceInfo, FriOpeningBatch, FriOpenings, FriOracleInfo, FriPolynomialInfo,
};

/// Describes an instance of a FRI-based batch opening (circuit target version).
#[derive(Debug)]
pub struct FriInstanceInfoTarget<const D: usize> {
    /// The oracles involved, not counting oracles created during the commit phase.
    pub oracles: Vec<FriOracleInfo>,
    /// Batches of openings, where each batch is associated with a particular point.
    pub batches: Vec<FriBatchInfoTarget<D>>,
}

/// A batch of openings at a particular point (circuit target version).
#[derive(Debug)]
pub struct FriBatchInfoTarget<const D: usize> {
    pub point: ExtensionTarget<D>,
    pub polynomials: Vec<FriPolynomialInfo>,
}

/// Opened values of each polynomial (circuit target version).
#[derive(Debug)]
pub struct FriOpeningsTarget<const D: usize> {
    pub batches: Vec<FriOpeningBatchTarget<D>>,
}

/// Opened values of each polynomial that's opened at a particular point (circuit target version).
#[derive(Debug)]
pub struct FriOpeningBatchTarget<const D: usize> {
    pub values: Vec<ExtensionTarget<D>>,
}
