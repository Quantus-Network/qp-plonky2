//! FRI proof shape validation.
//!
//! Re-exports validation functions from core.

pub use qp_plonky2_core::fri_validate_shape::{
    check_batch_fri_degree_bits, check_fri_proof_counts, check_fri_verifier_input_counts,
    checked_leaf_lengths, per_batch_counts, validate_batch_fri_proof_shape,
    validate_fri_verifier_inputs,
};
