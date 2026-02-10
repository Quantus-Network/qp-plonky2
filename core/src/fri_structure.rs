//! FRI structure types shared between prover and verifier.
//!
//! These types describe the structure of FRI openings and challenges.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ops::Range;

use crate::field::extension::Extendable;
use crate::hash_types::RichField;

/// Describes an instance of a FRI-based batch opening.
#[derive(Clone, Debug)]
pub struct FriInstanceInfo<F: RichField + Extendable<D>, const D: usize> {
    /// The oracles involved, not counting oracles created during the commit phase.
    pub oracles: Vec<FriOracleInfo>,
    /// Batches of openings, where each batch is associated with a particular point.
    pub batches: Vec<FriBatchInfo<F, D>>,
}

/// Information about a FRI oracle.
#[derive(Copy, Clone, Debug)]
pub struct FriOracleInfo {
    pub num_polys: usize,
    pub blinding: bool,
}

/// A batch of openings at a particular point.
#[derive(Clone, Debug)]
pub struct FriBatchInfo<F: RichField + Extendable<D>, const D: usize> {
    pub point: F::Extension,
    pub polynomials: Vec<FriPolynomialInfo>,
}

/// Information about a polynomial in a FRI oracle.
#[derive(Copy, Clone, Debug)]
pub struct FriPolynomialInfo {
    /// Index into `FriInstanceInfo`'s `oracles` list.
    pub oracle_index: usize,
    /// Index of the polynomial within the oracle.
    pub polynomial_index: usize,
}

impl FriPolynomialInfo {
    pub fn from_range(
        oracle_index: usize,
        polynomial_indices: Range<usize>,
    ) -> Vec<FriPolynomialInfo> {
        polynomial_indices
            .map(|polynomial_index| FriPolynomialInfo {
                oracle_index,
                polynomial_index,
            })
            .collect()
    }
}

/// Opened values of each polynomial.
#[derive(Debug)]
pub struct FriOpenings<F: RichField + Extendable<D>, const D: usize> {
    pub batches: Vec<FriOpeningBatch<F, D>>,
}

/// Opened values of each polynomial that's opened at a particular point.
#[derive(Debug)]
pub struct FriOpeningBatch<F: RichField + Extendable<D>, const D: usize> {
    pub values: Vec<F::Extension>,
}

/// Challenges generated during FRI verification.
#[derive(Debug)]
pub struct FriChallenges<F: RichField + Extendable<D>, const D: usize> {
    /// Scaling factor to combine polynomials.
    pub fri_alpha: F::Extension,

    /// Betas used in the FRI commit phase reductions.
    pub fri_betas: Vec<F::Extension>,

    /// Proof of work response.
    pub fri_pow_response: F,

    /// Indices at which the oracle is queried in FRI.
    pub fri_query_indices: Vec<usize>,
}
