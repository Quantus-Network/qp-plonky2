#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};
use core::marker::PhantomData;

use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::gate::VerificationGate;
use crate::gates::poseidon2::P2_WIDTH;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult};

/// Poseidon2 light-MDS Gate (width = P2_WIDTH).
///
/// This enforces one *light* MDS layer:
///   1. Apply the 4×4 matrix to each block of 4 elements (3 blocks total).
///   2. Add the column sums across the 3 blocks to each element in that column.
#[derive(Debug, Default)]
pub struct Poseidon2MdsGate<F: RichField + Extendable<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2MdsGate<F, D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    pub(crate) const fn wires_input(i: usize) -> core::ops::Range<usize> {
        assert!(i < P2_WIDTH);
        i * D..(i + 1) * D
    }

    pub(crate) const fn wires_output(i: usize) -> core::ops::Range<usize> {
        assert!(i < P2_WIDTH);
        (P2_WIDTH + i) * D..(P2_WIDTH + i + 1) * D
    }

    /// Light-MDS on extension field elements (for `eval_unfiltered_base_one`).
    fn mds_light_field<T: Field>(state: &[T; P2_WIDTH]) -> [T; P2_WIDTH] {
        let two = T::from_canonical_u64(2);
        let three = T::from_canonical_u64(3);

        let mut tmp = [T::ZERO; P2_WIDTH];

        // 3 blocks of size 4: [0..4), [4..8), [8..12)
        for k in (0..P2_WIDTH).step_by(4) {
            let a = state[k];
            let x = state[k + 1];
            let c = state[k + 2];
            let d = state[k + 3];

            // Standard 4×4 Poseidon2 light MDS matrix:
            //
            // [2 3 1 1]
            // [1 2 3 1]
            // [1 1 2 3]
            // [3 1 1 2]
            tmp[k] = a * two + x * three + c + d;
            tmp[k + 1] = a + x * two + c * three + d;
            tmp[k + 2] = a + x + c * two + d * three;
            tmp[k + 3] = a * three + x + c + d * two;
        }

        // Column sums across the three blocks.
        let mut sums = [T::ZERO; 4];
        for i in 0..4 {
            sums[i] = tmp[i] + tmp[4 + i] + tmp[8 + i];
        }

        // Add sums to each lane based on its column index mod 4.
        let mut out = [T::ZERO; P2_WIDTH];
        for i in 0..P2_WIDTH {
            out[i] = tmp[i] + sums[i % 4];
        }

        out
    }

    /// Light-MDS on extension algebra elements (for `eval_unfiltered`).
    fn mds_light_algebra(
        state: &[ExtensionAlgebra<F::Extension, D>; P2_WIDTH],
    ) -> [ExtensionAlgebra<F::Extension, D>; P2_WIDTH] {
        let two = F::Extension::from_canonical_u64(2);
        let three = F::Extension::from_canonical_u64(3);

        let mut tmp = [ExtensionAlgebra::ZERO; P2_WIDTH];

        for k in (0..P2_WIDTH).step_by(4) {
            let a = state[k];
            let x = state[k + 1];
            let c = state[k + 2];
            let d = state[k + 3];

            tmp[k] = a.scalar_mul(two) + x.scalar_mul(three) + c + d;
            tmp[k + 1] = a + x.scalar_mul(two) + c.scalar_mul(three) + d;
            tmp[k + 2] = a + x + c.scalar_mul(two) + d.scalar_mul(three);
            tmp[k + 3] = a.scalar_mul(three) + x + c + d.scalar_mul(two);
        }

        let mut sums = [ExtensionAlgebra::ZERO; 4];
        for i in 0..4 {
            sums[i] = tmp[i] + tmp[4 + i] + tmp[8 + i];
        }

        let mut out = [ExtensionAlgebra::ZERO; P2_WIDTH];
        for i in 0..P2_WIDTH {
            out[i] = tmp[i] + sums[i % 4];
        }

        out
    }
}

impl<F: RichField + Extendable<D>, const D: usize> VerificationGate<F, D>
    for Poseidon2MdsGate<F, D>
{
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={P2_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Poseidon2MdsGate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let inputs: [_; P2_WIDTH] = (0..P2_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_light_algebra(&inputs);

        (0..P2_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let inputs: [_; P2_WIDTH] = (0..P2_WIDTH)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_light_field(&inputs);

        yield_constr.many(
            (0..P2_WIDTH)
                .map(|i| vars.get_local_ext(Self::wires_output(i)))
                .zip(computed_outputs)
                .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array()),
        )
    }

    fn num_wires(&self) -> usize {
        2 * D * P2_WIDTH
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        P2_WIDTH * D
    }
}
