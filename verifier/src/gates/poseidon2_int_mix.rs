use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::gate::VerificationGate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult};
#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};
use core::marker::PhantomData;
use core::ops::Range;

// Re-use your existing constants
use crate::gates::poseidon2::Poseidon2Params;
use crate::gates::poseidon2::P2_WIDTH;
use qp_poseidon_constants::{
    POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW, POSEIDON2_INTERNAL_CONSTANTS_RAW,
    POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
};

/// Same formula as your existing helper:
/// y[i] = diag[i] * x[i] + sum_j x[j]
fn internal_mix_base<F: Field>(x: &[F; P2_WIDTH], diag: &[F; P2_WIDTH]) -> [F; P2_WIDTH] {
    let mut sum = x[0];
    for i in 1..P2_WIDTH {
        sum += x[i];
    }
    let mut y = [F::ZERO; P2_WIDTH];
    for i in 0..P2_WIDTH {
        y[i] = diag[i] * x[i] + sum;
    }
    y
}

#[derive(Clone, Debug, Default)]
pub struct Poseidon2IntMixGate<F: RichField + Extendable<D>, const D: usize> {
    /// diag in the base field F (same as Poseidon2Params.diag)
    diag: [F; P2_WIDTH],
    _pd: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2IntMixGate<F, D> {
    pub fn new() -> Self {
        // Reuse Poseidon2Params to get diag; we ignore other fields.
        let params = Poseidon2Params::<F, D>::from_p3_constants_u64(
            POSEIDON2_INITIAL_EXTERNAL_CONSTANTS_RAW,
            POSEIDON2_TERMINAL_EXTERNAL_CONSTANTS_RAW,
            POSEIDON2_INTERNAL_CONSTANTS_RAW,
        );
        Self {
            diag: params.diag,
            _pd: PhantomData,
        }
    }

    pub(crate) const fn wires_input(i: usize) -> Range<usize> {
        assert!(i < P2_WIDTH);
        i * D..(i + 1) * D
    }

    pub(crate) const fn wires_output(i: usize) -> Range<usize> {
        assert!(i < P2_WIDTH);
        (P2_WIDTH + i) * D..(P2_WIDTH + i + 1) * D
    }

    /// Internal mix over an extension algebra of F::Extension
    fn internal_mix_algebra(
        &self,
        state: &[ExtensionAlgebra<F::Extension, D>; P2_WIDTH],
    ) -> [ExtensionAlgebra<F::Extension, D>; P2_WIDTH] {
        // Lift diag to F::Extension
        let diag_ext: [F::Extension; P2_WIDTH] = core::array::from_fn(|i| {
            F::Extension::from_canonical_u64(self.diag[i].to_canonical_u64())
        });

        let mut sum = state[0];
        for i in 1..P2_WIDTH {
            sum += state[i];
        }

        let mut out = [ExtensionAlgebra::ZERO; P2_WIDTH];
        for i in 0..P2_WIDTH {
            let coeff = diag_ext[i];
            // coeff * state[i] + sum
            out[i] = state[i].scalar_mul(coeff) + sum;
        }
        out
    }
}

impl<F: RichField + Extendable<D>, const D: usize> VerificationGate<F, D>
    for Poseidon2IntMixGate<F, D>
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
        Ok(Self::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        // inputs as extension algebra elements
        let inputs: [_; P2_WIDTH] = (0..P2_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = self.internal_mix_algebra(&inputs);

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
        // Here we work in the base-one extension (F::Extension)
        let inputs: [_; P2_WIDTH] = (0..P2_WIDTH)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // diag lifted into F::Extension
        let diag_ext: [F::Extension; P2_WIDTH] = core::array::from_fn(|i| {
            F::Extension::from_canonical_u64(self.diag[i].to_canonical_u64())
        });

        let computed_outputs = internal_mix_base::<F::Extension>(&inputs, &diag_ext);

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
