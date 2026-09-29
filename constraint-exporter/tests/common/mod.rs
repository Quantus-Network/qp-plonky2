use constraint_exporter::gadget::{Call, Fact};
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::poseidon2::Poseidon2Hash;
use plonky2::iop::target::Target;
use plonky2::plonk::config::Hasher;

/// Every recorded fact holds numerically on the witness `val`.
pub fn assert_facts_hold(calls: &[Call], val: impl Fn(Target) -> F) {
    for c in calls {
        match c.fact {
            Fact::Select { b, x, y, out } => {
                assert_eq!(val(out), val(b) * (val(x) - val(y)) + val(y))
            }
            Fact::Not { b, out } => assert_eq!(val(out), F::ONE - val(b)),
            Fact::And { b1, b2, out } => assert_eq!(val(out), val(b1) * val(b2)),
            Fact::Or { b1, b2, out } => {
                assert_eq!(val(out), val(b1) + val(b2) - val(b1) * val(b2))
            }
            Fact::Add { x, y, out } => assert_eq!(val(out), val(x) + val(y)),
            Fact::Sub { x, y, out } => assert_eq!(val(out), val(x) - val(y)),
            Fact::Mul { x, y, out } => assert_eq!(val(out), val(x) * val(y)),
            Fact::AssertBool { b } => assert!(val(b) == F::ZERO || val(b) == F::ONE),
            Fact::IsEqual { x, y, equal, inv } => {
                let d = val(x) - val(y);
                assert_eq!(val(equal) * d, F::ZERO);
                assert_eq!(d * val(inv) - (F::ONE - val(equal)), F::ZERO);
            }
            Fact::RangeCheck { x, bits } => assert!(val(x).to_canonical_u64() < 1 << bits),
            Fact::Connect { x, y } => assert_eq!(val(x), val(y)),
            Fact::SplitLe { x, row, bits } => {
                let limbs: Vec<u64> = (1..=bits)
                    .map(|col| val(Target::wire(row, col)).to_canonical_u64())
                    .collect();
                assert!(limbs.iter().all(|&b| b < 2));
                let sum: u64 = limbs.iter().rev().fold(0, |acc, &b| 2 * acc + b);
                assert_eq!(val(x).to_canonical_u64(), sum);
            }
            Fact::Poseidon2 {
                ref rows,
                ref inputs,
            } => {
                let digest =
                    Poseidon2Hash::hash_no_pad(&inputs.iter().map(|&t| val(t)).collect::<Vec<_>>());
                let last = *rows.last().unwrap();
                for (i, d) in digest.elements.iter().enumerate() {
                    assert_eq!(val(Target::wire(last, 12 + i)), *d);
                }
            }
        }
    }
}
