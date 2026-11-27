//! Utility methods and constants for Plonk.
//!
//! Re-exports types from core.

pub use qp_plonky2_core::plonk_common::{
    eval_l_0, eval_zero_poly, num_partial_products, reduce_with_powers, reduce_with_powers_multi,
    salt_size, PlonkOracle, SALT_SIZE,
};
