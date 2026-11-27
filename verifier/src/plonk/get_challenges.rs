use qp_plonky2_core::field::polynomial::PolynomialCoeffs;
use qp_plonky2_core::fri_proof::FriProof;
use qp_plonky2_core::merkle_tree::MerkleCap;
use qp_plonky2_core::{Challenger, FriChallenger, FriParamsObserve};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::proof::{OpeningSet, Proof, ProofChallenges, ProofWithPublicInputs};

fn get_challenges<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    public_inputs_hash: <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_cap: &MerkleCap<F, C::Hasher>,
    plonk_zs_partial_products_cap: &MerkleCap<F, C::Hasher>,
    quotient_polys_cap: &MerkleCap<F, C::Hasher>,
    openings: &OpeningSet<F, D>,
    commit_phase_merkle_caps: &[MerkleCap<F, C::Hasher>],
    final_poly: &PolynomialCoeffs<F::Extension>,
    pow_witness: F,
    circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
    common_data: &CommonCircuitData<F, D>,
) -> anyhow::Result<ProofChallenges<F, D>> {
    let config = &common_data.config;
    let num_challenges = config.num_challenges;

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(*circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(wires_cap);
    let plonk_betas = challenger.get_n_challenges(num_challenges);
    let plonk_gammas = challenger.get_n_challenges(num_challenges);

    challenger.observe_cap::<C::Hasher>(plonk_zs_partial_products_cap);
    let plonk_alphas = challenger.get_n_challenges(num_challenges);

    challenger.observe_cap::<C::Hasher>(quotient_polys_cap);
    let plonk_zeta = challenger.get_extension_challenge::<D>();

    challenger.observe_openings(&openings.to_fri_openings());

    Ok(ProofChallenges {
        plonk_betas,
        plonk_gammas,
        plonk_alphas,
        plonk_zeta,
        fri_challenges: challenger.fri_challenges::<C, D>(
            commit_phase_merkle_caps,
            final_poly,
            pow_witness,
            common_data.degree_bits(),
            &config.fri_config,
        ),
    })
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProofWithPublicInputs<F, C, D>
{
    /// Computes all Fiat-Shamir challenges used in the Plonk proof.
    pub fn get_challenges(
        &self,
        public_inputs_hash: <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<ProofChallenges<F, D>> {
        let Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof:
                FriProof {
                    commit_phase_merkle_caps,
                    final_poly,
                    pow_witness,
                    ..
                },
        } = &self.proof;

        get_challenges::<F, C, D>(
            public_inputs_hash,
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            commit_phase_merkle_caps,
            final_poly,
            *pow_witness,
            circuit_digest,
            common_data,
        )
    }
}
