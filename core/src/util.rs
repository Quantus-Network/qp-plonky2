//! Utility functions shared between prover and verifier.
//!
//! Re-exports from plonky2_util for consistency, with additional helper functions.

pub use plonky2_util::{
    assume, branch_hint, log2_ceil, log2_strict, reverse_index_bits, reverse_index_bits_in_place,
};

/// Reverse the `num_bits` lowest bits of `n`.
pub const fn reverse_bits(n: usize, num_bits: usize) -> usize {
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    n.reverse_bits()
        .overflowing_shr(usize::BITS - num_bits as u32)
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b0000000000, 10), 0b0000000000);
        assert_eq!(reverse_bits(0b0000000001, 10), 0b1000000000);
        assert_eq!(reverse_bits(0b1000000000, 10), 0b0000000001);
        assert_eq!(reverse_bits(0b00000, 5), 0b00000);
        assert_eq!(reverse_bits(0b01011, 5), 0b11010);
    }

    #[test]
    fn test_log2_strict() {
        assert_eq!(log2_strict(1), 0);
        assert_eq!(log2_strict(2), 1);
        assert_eq!(log2_strict(1 << 18), 18);
        assert_eq!(log2_strict(1 << 31), 31);
    }

    #[test]
    fn test_log2_ceil() {
        assert_eq!(log2_ceil(0), 0);
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(1 << 18), 18);
    }
}
