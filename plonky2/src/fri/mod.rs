//! Fast Reed-Solomon IOP (FRI) protocol.
//!
//! It provides both a native implementation and an in-circuit version
//! of the FRI verifier for recursive proof composition.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::extension::Extendable;

use crate::hash::hash_types::RichField;
use crate::iop::challenger::{Challenger, RecursiveChallenger};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

pub mod challenges;
pub mod oracle;
pub mod proof;
pub mod prover;
pub mod recursive_verifier;
pub mod structure;
pub(crate) mod validate_shape;
pub mod verifier;
pub mod witness_util;

// Re-export FRI types from core
pub use qp_plonky2_core::{FriChallenger, FriConfig, FriParams, FriReductionStrategy};

/// Trait for FriConfig to add prover-specific support.
pub trait FriConfigObserve {
    /// Observe the FRI configuration parameters (for prover's Challenger).
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>);

    /// Observe the FRI configuration parameters for the recursive verifier.
    fn observe_target<F, H, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        challenger: &mut RecursiveChallenger<F, H, D>,
    ) where
        F: RichField + Extendable<D>,
        H: AlgebraicHasher<F>;
}

impl FriConfigObserve for FriConfig {
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>) {
        challenger.observe_element(F::from_canonical_usize(self.rate_bits));
        challenger.observe_element(F::from_canonical_usize(self.cap_height));
        challenger.observe_element(F::from_canonical_u32(self.proof_of_work_bits));
        challenger.observe_elements(&self.reduction_strategy.serialize());
        challenger.observe_element(F::from_canonical_usize(self.num_query_rounds));
    }

    fn observe_target<F, H, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        challenger: &mut RecursiveChallenger<F, H, D>,
    ) where
        F: RichField + Extendable<D>,
        H: AlgebraicHasher<F>,
    {
        challenger.observe_element(builder.constant(F::from_canonical_usize(self.rate_bits)));
        challenger.observe_element(builder.constant(F::from_canonical_usize(self.cap_height)));
        challenger
            .observe_element(builder.constant(F::from_canonical_u32(self.proof_of_work_bits)));
        challenger.observe_elements(&builder.constants(&self.reduction_strategy.serialize()));
        challenger
            .observe_element(builder.constant(F::from_canonical_usize(self.num_query_rounds)));
    }
}

/// Trait for FriParams to add prover-specific support.
pub trait FriParamsObserve {
    /// Observe the FRI parameters (for prover's Challenger).
    fn observe<F: RichField, H: Hasher<F>>(&self, challenger: &mut Challenger<F, H>);

    /// Observe the FRI parameters for the recursive verifier.
    fn observe_target<F, H, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        challenger: &mut RecursiveChallenger<F, H, D>,
    ) where
        F: RichField + Extendable<D>,
        H: AlgebraicHasher<F>;
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

    fn observe_target<F, H, const D: usize>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        challenger: &mut RecursiveChallenger<F, H, D>,
    ) where
        F: RichField + Extendable<D>,
        H: AlgebraicHasher<F>,
    {
        self.config.observe_target(builder, challenger);

        challenger.observe_element(builder.constant(F::from_bool(self.hiding)));
        challenger.observe_element(builder.constant(F::from_canonical_usize(self.degree_bits)));
        challenger.observe_elements(
            &builder.constants(
                &self
                    .reduction_arity_bits
                    .iter()
                    .map(|&e| F::from_canonical_usize(e))
                    .collect::<Vec<_>>(),
            ),
        );
    }
}
