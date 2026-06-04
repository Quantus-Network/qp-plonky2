//! An example of generating and verifying a STARK to highlight the use of the
//! permutation argument with logUp.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::StarkFrame;
use crate::lookup::{Column, Lookup};
use crate::stark::Stark;
use crate::util::trace_rows_to_poly_values;

/// Computes a sequence with state `[i, j]` using the state transition
/// i' <- i+1, j' <- j+1`.
/// Note: The `0, 1` columns are the columns used to test the permutation argument.
#[derive(Copy, Clone)]
struct PermutationStark<F: RichField + Extendable<D>, const D: usize> {
    num_rows: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> PermutationStark<F, D> {
    const fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            _phantom: PhantomData,
        }
    }

    /// Generate the trace using `x0, x0+1, 1` as initial state values.
    fn generate_trace(&self, x0: F) -> Vec<PolynomialValues<F>> {
        let mut trace_rows = (0..self.num_rows)
            .scan([x0, x0 + F::ONE, F::ONE], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[0] + F::ONE;
                acc[1] = tmp[1] + F::ONE;
                // acc[2] (i.e. frequency column) remains unchanged, as we're permuting a strictly monotonous sequence.
                Some(tmp)
            })
            .collect::<Vec<_>>();
        trace_rows[self.num_rows - 1][1] = x0; // So that column 0 and 1 are permutation of one another.
        trace_rows_to_poly_values(trace_rows)
    }
}

const PERM_COLUMNS: usize = 3;
const PERM_PUBLIC_INPUTS: usize = 1;

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PermutationStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, PERM_COLUMNS, PERM_PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, PERM_COLUMNS, PERM_PUBLIC_INPUTS>;

    fn constraint_degree(&self) -> usize {
        0
    }

    fn lookups(&self) -> Vec<Lookup<F>> {
        vec![Lookup {
            columns: vec![Column::single(0)],
            table_column: Column::single(1),
            frequencies_column: Column::single(2),
            filter_columns: vec![Default::default()],
        }]
    }

    // We don't constrain any register, for the sake of highlighting the permutation argument only.
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        _vars: &Self::EvaluationFrame<FE, P, D2>,
        _yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }

    // We don't constrain any register, for the sake of highlighting the permutation argument only.
    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: &Self::EvaluationFrameTarget,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2::field::extension::Extendable;
    use plonky2::field::types::Field;
    use plonky2::hash::hash_types::RichField;
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    use crate::config::StarkConfig;
    use crate::permutation_stark::PermutationStark;
    use crate::proof::StarkProofWithPublicInputs;
    use crate::prover::prove;
    use crate::recursive_verifier::{
        add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target,
        verify_stark_proof_circuit,
    };
    use crate::stark::Stark;
    use crate::stark_testing::{test_stark_circuit_constraints, test_stark_low_degree};
    use crate::verifier::verify_stark_proof;

    #[test]
    fn test_pemutations_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = PermutationStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let num_rows = 1 << 5;

        let public_input = F::ZERO;

        let stark = S::new(num_rows);
        let trace = stark.generate_trace(public_input);
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            &[public_input],
            None,
            &mut TimingTree::default(),
        )?;

        verify_stark_proof(stark, proof, &config, None)
    }

    #[test]
    fn test_permutation_stark_degree() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = PermutationStark<F, D>;

        let num_rows = 1 << 5;
        let stark = S::new(num_rows);
        test_stark_low_degree(stark)
    }

    #[test]
    fn test_permutation_stark_circuit() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = PermutationStark<F, D>;

        let num_rows = 1 << 5;
        let stark = S::new(num_rows);
        test_stark_circuit_constraints::<F, C, S, D>(stark)
    }

    #[test]
    fn test_recursive_stark_verifier() -> Result<()> {
        init_logger();
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = PermutationStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let num_rows = 1 << 5;
        let public_input = F::ZERO;

        let stark = S::new(num_rows);
        let trace = stark.generate_trace(public_input);
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            &[public_input],
            None,
            &mut TimingTree::default(),
        )?;
        verify_stark_proof(stark, proof.clone(), &config, None)?;

        recursive_proof::<F, C, S, C, D>(stark, proof, &config, true)
    }

    fn recursive_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        S: Stark<F, D> + Copy,
        InnerC: GenericConfig<D, F = F>,
        const D: usize,
    >(
        stark: S,
        inner_proof: StarkProofWithPublicInputs<F, InnerC, D>,
        inner_config: &StarkConfig,
        print_gate_counts: bool,
    ) -> Result<()>
    where
        InnerC::Hasher: AlgebraicHasher<F>,
        InnerC::InnerHasher: AlgebraicHasher<F>,
        C::InnerHasher: AlgebraicHasher<F>,
    {
        let circuit_config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(circuit_config);
        let mut pw = PartialWitness::new();
        let degree_bits = inner_proof.proof.recover_degree_bits(inner_config);
        let pt =
            add_virtual_stark_proof_with_pis(&mut builder, &stark, inner_config, degree_bits, 0, 0);
        set_stark_proof_with_pis_target(&mut pw, &pt, &inner_proof, degree_bits, builder.zero())?;

        verify_stark_proof_circuit::<F, InnerC, S, D>(&mut builder, stark, pt, inner_config, None);

        if print_gate_counts {
            builder.print_gate_counts(0);
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof)
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }

    /// Test that verifies the security fix for lookup-only STARKs.
    ///
    /// This test creates an invalid trace where the lookup column contains a value
    /// that doesn't exist in the table column, violating the lookup constraint.
    ///
    /// Before the fix (when `constraint_degree() == 0` caused `quotient_degree_factor() == 0`),
    /// no quotient polynomials were generated and the verifier would incorrectly accept
    /// this invalid proof. After the fix, the verifier correctly rejects it.
    #[test]
    fn test_invalid_lookup_rejected() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = PermutationStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let num_rows = 1 << 5;
        let public_input = F::ZERO;

        let stark = S::new(num_rows);

        // Generate a valid trace first
        let mut trace = stark.generate_trace(public_input);

        // Corrupt the trace: change a value in the looking column (column 0)
        // to a value that doesn't exist in the table column (column 1).
        // The valid trace has values 0, 1, 2, ..., 31 in both columns (as a permutation).
        // We'll change one value in column 0 to 999, which doesn't exist in column 1.
        trace[0].values[0] = F::from_canonical_u64(999);

        // Attempt to prove with the corrupted trace.
        // The prover may succeed in generating a proof (it doesn't always check constraints),
        // but verification MUST fail.
        let proof_result = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            &[public_input],
            None,
            &mut TimingTree::default(),
        );

        match proof_result {
            Err(_) => {
                // Proof generation failed - this is acceptable (constraint violation detected early)
            }
            Ok(proof) => {
                // Proof was generated, but verification MUST fail
                let verify_result = verify_stark_proof(stark, proof, &config, None);
                assert!(
                    verify_result.is_err(),
                    "Verification should fail for invalid lookup trace. \
                     Before the security fix, this would have incorrectly succeeded because \
                     quotient_degree_factor() was 0 for lookup-only STARKs, meaning no \
                     quotient polynomials were generated and the verifier had nothing to check."
                );
            }
        }
    }

    /// This test verifies that the fix correctly computes quotient_degree_factor for lookup-only STARKs.
    ///
    /// Before the fix:
    /// - constraint_degree() = 0 (no base constraints)
    /// - quotient_degree_factor() = 0 (derived only from constraint_degree)
    /// - num_quotient_polys() = 0 (no quotient polynomials generated)
    ///
    /// After the fix:
    /// - constraint_degree() = 0 (unchanged - no base constraints)
    /// - quotient_degree_factor() = 1 (accounts for degree-2 lookup constraints)
    /// - num_quotient_polys() > 0 (quotient polynomials are generated)
    #[test]
    fn test_lookup_only_stark_has_quotient_polys() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = PermutationStark<F, D>;

        let config = StarkConfig::standard_fast_config();
        let num_rows = 1 << 5;

        let stark = S::new(num_rows);

        // Verify the STARK configuration
        assert_eq!(
            stark.constraint_degree(),
            0,
            "PermutationStark should have constraint_degree() == 0 (no base constraints)"
        );

        assert!(
            stark.uses_lookups(),
            "PermutationStark should use lookups"
        );

        // This is the key assertion - with the fix, quotient_degree_factor should be > 0
        // even when constraint_degree() == 0, because lookup constraints have degree 2.
        assert!(
            stark.quotient_degree_factor() > 0,
            "quotient_degree_factor() should be > 0 for lookup-only STARKs. \
             Before the fix, this was 0, which meant lookup constraints were never enforced."
        );

        assert!(
            stark.num_quotient_polys(&config) > 0,
            "num_quotient_polys() should be > 0 for lookup-only STARKs"
        );

        // Generate a valid proof and verify the proof structure includes quotient polynomials
        let public_input = F::ZERO;
        let trace = stark.generate_trace(public_input);
        let proof = prove::<F, C, S, D>(
            stark,
            &config,
            trace,
            &[public_input],
            None,
            &mut TimingTree::default(),
        )
        .expect("Valid proof should succeed");

        assert!(
            proof.proof.quotient_polys_cap.is_some(),
            "Proof should contain quotient polynomial commitment"
        );

        assert!(
            proof.proof.openings.quotient_polys.is_some(),
            "Proof should contain quotient polynomial openings"
        );

        // Verify the proof is valid
        verify_stark_proof(stark, proof, &config, None)
            .expect("Valid proof should verify successfully");
    }
}
