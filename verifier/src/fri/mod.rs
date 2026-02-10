//! Fast Reed-Solomon IOP (FRI) protocol.
//!
//! This module provides the FRI verifier implementation.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::hash::hash_types::RichField;
use crate::plonk::config::Hasher;
use qp_plonky2_core::Challenger;

pub mod challenges;
pub mod proof;
pub mod structure;
pub(crate) mod validate_shape;
pub mod verifier;

// Re-export FRI types from core
pub use qp_plonky2_core::{FriChallenger, FriConfig, FriParams, FriReductionStrategy};

/// Trait for FriConfig to add observe method for verifier's Challenger.
pub trait FriConfigObserve {
    /// Observe the FRI configuration parameters.
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>);
}

impl FriConfigObserve for FriConfig {
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>) {
        challenger.observe_element(F::from_canonical_usize(self.rate_bits));
        challenger.observe_element(F::from_canonical_usize(self.cap_height));
        challenger.observe_element(F::from_canonical_u32(self.proof_of_work_bits));
        challenger.observe_elements(&self.reduction_strategy.serialize());
        challenger.observe_element(F::from_canonical_usize(self.num_query_rounds));
    }
}

/// Trait for FriParams to add observe method for verifier's Challenger.
pub trait FriParamsObserve {
    /// Observe the FRI parameters.
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>);
}

impl FriParamsObserve for FriParams {
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>) {
        self.config.observe(challenger);

        challenger.observe_element(F::from_bool(self.hiding));
        challenger.observe_element(F::from_canonical_usize(self.degree_bits));
        challenger.observe_elements(
            &self
                .reduction_arity_bits
                .iter()
                .map(|&e| F::from_canonical_usize(e))
                .collect::<Vec<_>>(),
        );
    }
}
