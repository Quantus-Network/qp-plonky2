//! Optimized Poseidon2 implementation with raw u64/u128 arithmetic.
//!
//! This module provides a `Poseidon2` trait similar to the `Poseidon` trait,
//! with highly optimized implementations for GoldilocksField using raw
//! integer arithmetic to minimize field reductions.

// Re-export constants from qp-poseidon-constants
pub use qp_poseidon_constants::{
    POSEIDON2_EXTERNAL_ROUNDS, POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
    POSEIDON2_INTERNAL_CONSTANTS_RAW, POSEIDON2_INTERNAL_ROUNDS, POSEIDON2_MATRIX_DIAG_12_RAW,
    POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW, SPONGE_CAPACITY, SPONGE_RATE, SPONGE_WIDTH,
};
use unroll::unroll_for_loops;

use crate::field::extension::FieldExtension;
use crate::field::types::PrimeField64;

/// Optimized Poseidon2 permutation trait.
///
/// Implementations should provide raw u64 arithmetic for maximum performance.
pub trait Poseidon2: PrimeField64 {
    // =========================================================================
    // S-box: x^7
    // =========================================================================

    /// Compute x^7 using raw u64 arithmetic.
    #[inline(always)]
    fn sbox_monomial(x: Self) -> Self {
        let x2 = x.square();
        let x4 = x2.square();
        let x3 = x * x2;
        x3 * x4
    }

    /// Apply S-box to all lanes (external rounds).
    #[inline(always)]
    #[unroll_for_loops]
    fn sbox_layer(state: &mut [Self; SPONGE_WIDTH]) {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                state[i] = Self::sbox_monomial(state[i]);
            }
        }
    }

    // =========================================================================
    // Light MDS (external diffusion): 4x4 circulant blocks + cross-block sums
    // =========================================================================

    /// Apply the 4x4 circulant matrix [2,3,1,1] to a 4-element block.
    /// Uses u128 accumulation to defer reductions.
    #[inline(always)]
    fn apply_mat4_u128(a: u64, b: u64, c: u64, d: u64) -> (u128, u128, u128, u128) {
        // t = a + b + c + d
        let t = (a as u128) + (b as u128) + (c as u128) + (d as u128);
        // y0 = 2a + 3b + c + d = t + a + 2b
        let y0 = t + (a as u128) + (b as u128) + (b as u128);
        // y1 = a + 2b + 3c + d = t + b + 2c
        let y1 = t + (b as u128) + (c as u128) + (c as u128);
        // y2 = a + b + 2c + 3d = t + c + 2d
        let y2 = t + (c as u128) + (d as u128) + (d as u128);
        // y3 = 3a + b + c + 2d = t + 2a + d
        let y3 = t + (a as u128) + (a as u128) + (d as u128);
        (y0, y1, y2, y3)
    }

    /// Apply the light MDS layer used in Poseidon2 external rounds.
    /// Operates on raw u64 values and returns field elements.
    #[inline(always)]
    #[unroll_for_loops]
    fn mds_light_layer(state: &[Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        // Convert to raw u64
        let mut s = [0u64; SPONGE_WIDTH];
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                s[i] = state[i].to_noncanonical_u64();
            }
        }

        // Apply 4x4 blocks
        let (y0, y1, y2, y3) = Self::apply_mat4_u128(s[0], s[1], s[2], s[3]);
        let (y4, y5, y6, y7) = Self::apply_mat4_u128(s[4], s[5], s[6], s[7]);
        let (y8, y9, y10, y11) = Self::apply_mat4_u128(s[8], s[9], s[10], s[11]);

        // Compute sums per residue class (still in u128)
        let sum0 = y0 + y4 + y8;
        let sum1 = y1 + y5 + y9;
        let sum2 = y2 + y6 + y10;
        let sum3 = y3 + y7 + y11;

        // Final values and reduce
        let mut result = [Self::ZERO; SPONGE_WIDTH];
        result[0] = Self::from_noncanonical_u128(y0 + sum0);
        result[1] = Self::from_noncanonical_u128(y1 + sum1);
        result[2] = Self::from_noncanonical_u128(y2 + sum2);
        result[3] = Self::from_noncanonical_u128(y3 + sum3);
        result[4] = Self::from_noncanonical_u128(y4 + sum0);
        result[5] = Self::from_noncanonical_u128(y5 + sum1);
        result[6] = Self::from_noncanonical_u128(y6 + sum2);
        result[7] = Self::from_noncanonical_u128(y7 + sum3);
        result[8] = Self::from_noncanonical_u128(y8 + sum0);
        result[9] = Self::from_noncanonical_u128(y9 + sum1);
        result[10] = Self::from_noncanonical_u128(y10 + sum2);
        result[11] = Self::from_noncanonical_u128(y11 + sum3);

        result
    }

    /// Light MDS for field extensions.
    fn mds_light_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; SPONGE_WIDTH],
    ) -> [F; SPONGE_WIDTH] {
        // Apply 4x4 blocks using field arithmetic
        #[inline(always)]
        fn apply_mat4_field<F: Copy + core::ops::Add<Output = F>>(
            a: F,
            b: F,
            c: F,
            d: F,
        ) -> [F; 4] {
            let t = a + b + c + d;
            [
                t + a + b + b, // 2a + 3b + c + d
                t + b + c + c, // a + 2b + 3c + d
                t + c + d + d, // a + b + 2c + 3d
                t + a + a + d, // 3a + b + c + 2d
            ]
        }

        let [y0, y1, y2, y3] = apply_mat4_field(state[0], state[1], state[2], state[3]);
        let [y4, y5, y6, y7] = apply_mat4_field(state[4], state[5], state[6], state[7]);
        let [y8, y9, y10, y11] = apply_mat4_field(state[8], state[9], state[10], state[11]);

        let sum0 = y0 + y4 + y8;
        let sum1 = y1 + y5 + y9;
        let sum2 = y2 + y6 + y10;
        let sum3 = y3 + y7 + y11;

        [
            y0 + sum0,
            y1 + sum1,
            y2 + sum2,
            y3 + sum3,
            y4 + sum0,
            y5 + sum1,
            y6 + sum2,
            y7 + sum3,
            y8 + sum0,
            y9 + sum1,
            y10 + sum2,
            y11 + sum3,
        ]
    }

    // =========================================================================
    // Internal diffusion: y[i] = diag[i] * x[i] + sum(x)
    // =========================================================================

    /// Apply internal mixing layer using raw arithmetic.
    #[inline(always)]
    #[unroll_for_loops]
    fn internal_mix_layer(state: &[Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        // Convert to raw u64
        let mut s = [0u64; SPONGE_WIDTH];
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                s[i] = state[i].to_noncanonical_u64();
            }
        }

        // Compute sum in u128
        let mut sum = 0u128;
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                sum += s[i] as u128;
            }
        }

        // Compute y[i] = diag[i] * s[i] + sum
        let mut result = [Self::ZERO; SPONGE_WIDTH];
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                let prod = (s[i] as u128) * (POSEIDON2_MATRIX_DIAG_12_RAW[i] as u128);
                result[i] = Self::from_noncanonical_u128(prod + sum);
            }
        }

        result
    }

    /// Internal mix for field extensions.
    fn internal_mix_layer_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &[F; SPONGE_WIDTH],
    ) -> [F; SPONGE_WIDTH] {
        let mut sum = F::ZERO;
        for i in 0..SPONGE_WIDTH {
            sum = sum + state[i];
        }

        let mut result = [F::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            let diag = F::from_canonical_u64(POSEIDON2_MATRIX_DIAG_12_RAW[i]);
            result[i] = state[i] * diag + sum;
        }
        result
    }

    // =========================================================================
    // Constant addition layers
    // =========================================================================

    /// Add external round constants (initial phase).
    #[inline(always)]
    #[unroll_for_loops]
    fn add_ext_initial_constants(state: &mut [Self; SPONGE_WIDTH], round: usize) {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                // SAFETY: Constants are canonical
                state[i] = unsafe {
                    state[i].add_canonical_u64(POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW[round][i])
                };
            }
        }
    }

    /// Add external round constants (terminal phase).
    #[inline(always)]
    #[unroll_for_loops]
    fn add_ext_terminal_constants(state: &mut [Self; SPONGE_WIDTH], round: usize) {
        for i in 0..12 {
            if i < SPONGE_WIDTH {
                // SAFETY: Constants are canonical
                state[i] = unsafe {
                    state[i].add_canonical_u64(POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW[round][i])
                };
            }
        }
    }

    /// Add internal round constant to lane 0.
    #[inline(always)]
    fn add_internal_constant(state: &mut [Self; SPONGE_WIDTH], round: usize) {
        // SAFETY: Constants are canonical
        state[0] = unsafe { state[0].add_canonical_u64(POSEIDON2_INTERNAL_CONSTANTS_RAW[round]) };
    }

    // =========================================================================
    // Full permutation
    // =========================================================================

    /// Execute the full Poseidon2 permutation.
    #[inline(always)]
    fn poseidon2_permutation(state: &mut [Self; SPONGE_WIDTH]) {
        // Initial preamble: light MDS
        *state = Self::mds_light_layer(state);

        // 4 initial external rounds
        for r in 0..4 {
            Self::add_ext_initial_constants(state, r);
            Self::sbox_layer(state);
            *state = Self::mds_light_layer(state);
        }

        // 22 internal rounds
        for r in 0..POSEIDON2_INTERNAL_ROUNDS {
            Self::add_internal_constant(state, r);
            state[0] = Self::sbox_monomial(state[0]);
            *state = Self::internal_mix_layer(state);
        }

        // 4 terminal external rounds
        for r in 0..4 {
            Self::add_ext_terminal_constants(state, r);
            Self::sbox_layer(state);
            *state = Self::mds_light_layer(state);
        }
    }
}

// Implement for GoldilocksField
use plonky2_field::goldilocks_field::GoldilocksField;

impl Poseidon2 for GoldilocksField {}

#[cfg(test)]
mod tests {
    use plonky2_field::types::Field;

    use super::*;

    #[test]
    fn test_poseidon2_permutation() {
        // Test that permutation produces non-trivial output
        let mut state = [GoldilocksField::ZERO; SPONGE_WIDTH];
        state[0] = GoldilocksField::ONE;

        GoldilocksField::poseidon2_permutation(&mut state);

        // Output should be non-zero and different from input
        assert_ne!(state[0], GoldilocksField::ZERO);
        assert_ne!(state[0], GoldilocksField::ONE);
    }

    #[test]
    fn test_mds_light_layer() {
        // Test MDS layer produces expected structure
        let mut state = [GoldilocksField::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = GoldilocksField::from_canonical_u64(i as u64 + 1);
        }

        let result = GoldilocksField::mds_light_layer(&state);

        // Result should be non-trivial
        for i in 0..SPONGE_WIDTH {
            assert_ne!(result[i], GoldilocksField::ZERO);
        }
    }

    #[test]
    fn test_internal_mix_layer() {
        // Test internal mix produces expected structure
        let mut state = [GoldilocksField::ZERO; SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = GoldilocksField::from_canonical_u64(i as u64 + 1);
        }

        let result = GoldilocksField::internal_mix_layer(&state);

        // Result should be non-trivial
        for i in 0..SPONGE_WIDTH {
            assert_ne!(result[i], GoldilocksField::ZERO);
        }
    }
}
