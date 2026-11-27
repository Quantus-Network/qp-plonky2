#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use super::circuit_builder::NUM_COINS_LOOKUP;
use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialCoeffs;
use crate::fri::proof::{FriFinalPolyTarget, FriProof, FriProofTarget};
use crate::fri::{FriChallenger, FriParamsObserve, FriParamsObserveTarget};
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::challenger::{Challenger, RecursiveChallenger};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher};
use crate::plonk::proof::{
    OpeningSet, OpeningSetTarget, Proof, ProofChallenges, ProofChallengesTarget, ProofTarget,
    ProofWithPublicInputs, ProofWithPublicInputsTarget,
};

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
    let has_lookup = common_data.num_lookup_polys != 0;

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(*circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(wires_cap);
    let plonk_betas = challenger.get_n_challenges(num_challenges);
    let plonk_gammas = challenger.get_n_challenges(num_challenges);

    // If there are lookups in the circuit, we should get delta challenges as well.
    // But we can use the already generated `plonk_betas` and `plonk_gammas` as the first `plonk_deltas` challenges.
    let plonk_deltas = if has_lookup {
        let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;
        let mut deltas = Vec::with_capacity(num_lookup_challenges);
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        let additional = challenger.get_n_challenges(num_additional_challenges);
        deltas.extend(&plonk_betas);
        deltas.extend(&plonk_gammas);
        deltas.extend(additional);
        deltas
    } else {
        vec![]
    };

    // `plonk_zs_partial_products_cap` also contains the commitment to lookup polynomials.
    challenger.observe_cap::<C::Hasher>(plonk_zs_partial_products_cap);
    let plonk_alphas = challenger.get_n_challenges(num_challenges);

    challenger.observe_cap::<C::Hasher>(quotient_polys_cap);
    let plonk_zeta = challenger.get_extension_challenge::<D>();

    challenger.observe_openings(&openings.to_fri_openings());

    Ok(ProofChallenges {
        plonk_betas,
        plonk_gammas,
        plonk_alphas,
        plonk_deltas,
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

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    fn get_challenges<C: GenericConfig<D, F = F>>(
        &mut self,
        public_inputs_hash: HashOutTarget,
        wires_cap: &MerkleCapTarget,
        plonk_zs_partial_products_cap: &MerkleCapTarget,
        quotient_polys_cap: &MerkleCapTarget,
        openings: &OpeningSetTarget<D>,
        commit_phase_merkle_caps: &[MerkleCapTarget],
        final_poly: &FriFinalPolyTarget<D>,
        pow_witness: Target,
        inner_circuit_digest: HashOutTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) -> ProofChallengesTarget<D>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        let config = &inner_common_data.config;
        let num_challenges = config.num_challenges;

        let mut challenger = RecursiveChallenger::<F, C::Hasher, D>::new(self);
        let has_lookup = inner_common_data.num_lookup_polys != 0;

        // Observe the FRI config
        inner_common_data
            .fri_params
            .observe_target(self, &mut challenger);

        // Observe the instance.
        challenger.observe_hash(&inner_circuit_digest);
        challenger.observe_hash(&public_inputs_hash);

        challenger.observe_cap(wires_cap);

        let plonk_betas = challenger.get_n_challenges(self, num_challenges);
        let plonk_gammas = challenger.get_n_challenges(self, num_challenges);

        // If there are lookups in the circuit, we should get delta challenges as well.
        // But we can use the already generated `plonk_betas` and `plonk_gammas` as the first `plonk_deltas` challenges.
        let plonk_deltas = if has_lookup {
            let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;
            let mut deltas = Vec::with_capacity(num_lookup_challenges);
            let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
            let additional = challenger.get_n_challenges(self, num_additional_challenges);
            deltas.extend(&plonk_betas);
            deltas.extend(&plonk_gammas);
            deltas.extend(additional);
            deltas
        } else {
            vec![]
        };

        challenger.observe_cap(plonk_zs_partial_products_cap);
        let plonk_alphas = challenger.get_n_challenges(self, num_challenges);

        challenger.observe_cap(quotient_polys_cap);
        let plonk_zeta = challenger.get_extension_challenge(self);

        challenger.observe_openings(&openings.to_fri_openings());

        ProofChallengesTarget {
            plonk_betas,
            plonk_gammas,
            plonk_alphas,
            plonk_deltas,
            plonk_zeta,
            fri_challenges: challenger.fri_challenges(
                self,
                commit_phase_merkle_caps,
                final_poly,
                pow_witness,
                &inner_common_data.config.fri_config,
            ),
        }
    }
}

impl<const D: usize> ProofWithPublicInputsTarget<D> {
    pub(crate) fn get_challenges<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>>(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        public_inputs_hash: HashOutTarget,
        inner_circuit_digest: HashOutTarget,
        inner_common_data: &CommonCircuitData<F, D>,
    ) -> ProofChallengesTarget<D>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        let ProofTarget {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof:
                FriProofTarget {
                    commit_phase_merkle_caps,
                    final_poly,
                    pow_witness,
                    ..
                },
        } = &self.proof;

        builder.get_challenges::<C>(
            public_inputs_hash,
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            commit_phase_merkle_caps,
            final_poly,
            *pow_witness,
            inner_circuit_digest,
            inner_common_data,
        )
    }
}
