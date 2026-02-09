//! Utility module for helper methods and plonky2 serialization logic.

#[doc(inline)]
pub use plonky2_util::*;

pub mod serialization;

// Re-export reducing and strided_view from core
pub use qp_plonky2_core::reducing;
pub use qp_plonky2_core::strided_view;

pub(crate) const fn reverse_bits(n: usize, num_bits: usize) -> usize {
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

    #[cfg(not(feature = "std"))]
    use alloc::vec;

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
    fn test_reverse_index_bits() {
        assert_eq!(reverse_index_bits(&[10, 20, 30, 40]), vec![10, 30, 20, 40]);
    }
}
