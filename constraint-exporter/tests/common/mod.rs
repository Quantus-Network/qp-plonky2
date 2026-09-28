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
            Fact::Poseidon2 { row, inputs } => {
                let digest = Poseidon2Hash::hash_no_pad(&inputs.map(&val));
                for (i, d) in digest.elements.iter().enumerate() {
                    assert_eq!(val(Target::wire(row, 12 + i)), *d);
                }
            }
        }
    }
}
