#![no_std]

// Minimal usage to ensure the crate links in no_std
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;

#[allow(dead_code)]
pub fn add_one(x: GoldilocksField) -> GoldilocksField {
    x + GoldilocksField::from_canonical_u64(1)
}
