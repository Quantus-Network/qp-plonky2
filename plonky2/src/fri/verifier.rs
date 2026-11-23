//! FRI verification logic.
//!
//! Re-exports FRI verification functions from core.

pub use qp_plonky2_core::fri_verifier::{
    compute_evaluation, eval_opening_expression, fri_combine_initial, fri_verify_proof_of_work,
    verify_fri_proof, PrecomputedReducedOpenings,
};

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::verify_fri_proof;
    use crate::fri::proof::{FriChallenges, FriProof};
    use crate::fri::structure::{FriInstanceInfo, FriOpenings};
    use crate::fri::{FriParams, FriReductionStrategy};
    use crate::hash::merkle_tree::MerkleCap;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type H = <C as GenericConfig<D>>::Hasher;

    /// An honest Plonk proof's FRI opening proof together with everything `verify_fri_proof`
    /// zips against it, so each list can be shortened or extended independently.
    struct FriVerifierInputs {
        instance: FriInstanceInfo<F, D>,
        openings: FriOpenings<F, D>,
        challenges: FriChallenges<F, D>,
        initial_merkle_caps: Vec<MerkleCap<F, H>>,
        proof: FriProof<F, H, D>,
        params: FriParams,
    }

    impl FriVerifierInputs {
        fn honest() -> anyhow::Result<Self> {
            // One reduction step, so the commit phase has a cap and a beta to tamper with.
            let mut config = CircuitConfig::standard_recursion_config();
            config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1]);
            let mut builder = CircuitBuilder::<F, D>::new(config);
            let one = builder.one();
            builder.register_public_input(one);
            let data = builder.build::<C>();

            let proof_with_pis = data.prove(PartialWitness::new())?;
            let challenges = proof_with_pis.get_challenges(
                proof_with_pis.get_public_inputs_hash(),
                &data.verifier_only.circuit_digest,
                &data.common,
            )?;
            let proof = proof_with_pis.proof;
            Ok(Self {
                instance: data.common.get_fri_instance(challenges.plonk_zeta),
                openings: proof.openings.to_fri_openings(),
                challenges: challenges.fri_challenges,
                initial_merkle_caps: vec![
                    data.verifier_only.constants_sigmas_cap.clone(),
                    proof.wires_cap,
                    proof.plonk_zs_partial_products_cap,
                    proof.quotient_polys_cap,
                ],
                proof: proof.opening_proof,
                params: data.common.fri_params.clone(),
            })
        }

        fn verify(&self) -> anyhow::Result<()> {
            verify_fri_proof::<F, C, D>(
                &self.instance,
                &self.openings,
                &self.challenges,
                &self.initial_merkle_caps,
                &self.proof,
                &self.params,
            )
        }

        fn rejects(&self) -> bool {
            catch_unwind(AssertUnwindSafe(|| self.verify())).map_or(true, |result| result.is_err())
        }
    }

    #[test]
    fn honest_inputs_verify() -> anyhow::Result<()> {
        FriVerifierInputs::honest()?.verify()
    }

    #[test]
    fn extra_commit_phase_cap_rejected() -> anyhow::Result<()> {
        let mut inputs = FriVerifierInputs::honest()?;
        let last_cap = inputs
            .proof
            .commit_phase_merkle_caps
            .last()
            .unwrap()
            .clone();
        inputs.proof.commit_phase_merkle_caps.push(last_cap);
        assert!(inputs.verify().is_err());
        Ok(())
    }

    #[test]
    fn short_fri_betas_rejected() -> anyhow::Result<()> {
        let mut inputs = FriVerifierInputs::honest()?;
        inputs.challenges.fri_betas.pop();
        assert!(inputs.rejects());
        Ok(())
    }

    #[test]
    fn short_query_indices_rejected() -> anyhow::Result<()> {
        let mut inputs = FriVerifierInputs::honest()?;
        inputs.challenges.fri_query_indices.pop();
        assert!(inputs.verify().is_err());
        Ok(())
    }

    #[test]
    fn out_of_domain_query_index_rejected() -> anyhow::Result<()> {
        let mut inputs = FriVerifierInputs::honest()?;
        inputs.challenges.fri_query_indices[0] = inputs.params.lde_size();
        assert!(inputs.rejects());
        Ok(())
    }

    #[test]
    fn missing_initial_cap_rejected() -> anyhow::Result<()> {
        let mut inputs = FriVerifierInputs::honest()?;
        inputs.initial_merkle_caps.pop();
        assert!(inputs.verify().is_err());
        Ok(())
    }

    #[test]
    fn short_openings_batch_rejected() -> anyhow::Result<()> {
        let mut inputs = FriVerifierInputs::honest()?;
        inputs.openings.batches[0].values.pop();
        assert!(inputs.rejects());
        Ok(())
    }
}
