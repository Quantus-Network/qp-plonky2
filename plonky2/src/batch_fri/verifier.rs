#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::ensure;
use itertools::Itertools;
use plonky2_field::extension::{flatten, Extendable, FieldExtension};
use plonky2_field::types::Field;

use crate::batch_fri::{openings_at_leaf_positions, BatchLeafLayout};
use crate::fri::proof::{FriChallenges, FriInitialTreeProof, FriProof, FriQueryRound};
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo, FriOpenings};
use crate::fri::validate_shape::{
    check_batch_fri_degree_bits, validate_batch_fri_proof_shape, validate_fri_verifier_inputs,
};
use crate::fri::verifier::{
    compute_evaluation, eval_opening_expression, fri_verify_proof_of_work,
    PrecomputedReducedOpenings,
};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::{verify_batch_merkle_proof_to_cap, verify_merkle_proof_to_cap};
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::util::reducing::ReducingFactor;
use crate::util::reverse_bits;

pub fn verify_batch_fri_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    degree_bits: &[usize],
    instances: &[FriInstanceInfo<F, D>],
    openings: &[FriOpenings<F, D>],
    challenges: &FriChallenges<F, D>,
    initial_merkle_cap: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()> {
    check_batch_fri_degree_bits(degree_bits, instances.len(), params)
        .map_err(anyhow::Error::msg)?;
    validate_batch_fri_proof_shape::<F, C, D>(proof, instances, params)?;
    validate_fri_verifier_inputs(instances, openings, challenges, initial_merkle_cap, params)?;

    // Check PoW.
    fri_verify_proof_of_work(challenges.fri_pow_response, &params.config)?;

    let oracles_per_instance = instances
        .iter()
        .map(|inst| inst.oracles.as_slice())
        .collect_vec();
    let layouts = BatchLeafLayout::per_oracle(&oracles_per_instance, params.leaf_hiding);
    let instances = instances
        .iter()
        .map(|inst| FriInstanceInfo {
            oracles: inst.oracles.clone(),
            batches: inst
                .batches
                .iter()
                .map(|batch| FriBatchInfo {
                    point: batch.point,
                    openings: openings_at_leaf_positions(&batch.openings, &layouts),
                })
                .collect(),
        })
        .collect_vec();

    let mut precomputed_reduced_evals = Vec::with_capacity(openings.len());
    for opn in openings {
        let pre = PrecomputedReducedOpenings::from_os_and_alpha(opn, challenges.fri_alpha);
        precomputed_reduced_evals.push(pre);
    }
    let degree_bits = degree_bits
        .iter()
        .map(|d| d + params.config.rate_bits)
        .collect_vec();
    for (&x_index, round_proof) in challenges
        .fri_query_indices
        .iter()
        .zip(&proof.query_round_proofs)
    {
        batch_fri_verifier_query_round::<F, C, D>(
            &degree_bits,
            &instances,
            &layouts,
            challenges,
            &precomputed_reduced_evals,
            initial_merkle_cap,
            proof,
            x_index,
            round_proof,
            params,
        )?;
    }

    Ok(())
}

fn batch_fri_verify_initial_proof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
    degree_bits: &[usize],
    layouts: &[BatchLeafLayout],
    x_index: usize,
    proof: &FriInitialTreeProof<F, H>,
    initial_merkle_caps: &[MerkleCap<F, H>],
) -> anyhow::Result<()> {
    for (((evals, merkle_proof), cap), layout) in proof
        .evals_proofs
        .iter()
        .zip(initial_merkle_caps)
        .zip(layouts)
    {
        let leaves = layout
            .group_ranges()
            .map(|range| evals[range].to_vec())
            .collect::<Vec<_>>();

        verify_batch_merkle_proof_to_cap::<F, H>(&leaves, degree_bits, x_index, cap, merkle_proof)?;
    }

    Ok(())
}

fn batch_fri_combine_initial<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instances: &[FriInstanceInfo<F, D>],
    index: usize,
    proof: &FriInitialTreeProof<F, C::Hasher>,
    alpha: F::Extension,
    subgroup_x: F,
    precomputed_reduced_evals: &PrecomputedReducedOpenings<F, D>,
    params: &FriParams,
) -> F::Extension {
    assert!(D > 1, "Not implemented for D=1.");
    let subgroup_x = F::Extension::from_basefield(subgroup_x);
    let mut alpha = ReducingFactor::new(alpha);
    let mut sum = F::Extension::ZERO;

    for (batch, reduced_openings) in instances[index]
        .batches
        .iter()
        .zip(&precomputed_reduced_evals.reduced_openings_at_point)
    {
        let FriBatchInfo { point, openings } = batch;
        let evals = openings.iter().map(|expression| {
            eval_opening_expression(&instances[index], expression, proof, *point, params)
        });
        let reduced_evals = alpha.reduce(evals);
        let numerator = reduced_evals - *reduced_openings;
        let denominator = subgroup_x - *point;
        sum = alpha.shift(sum);
        sum += numerator / denominator;
    }

    sum
}

fn batch_fri_verifier_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    degree_bits: &[usize],
    instances: &[FriInstanceInfo<F, D>],
    layouts: &[BatchLeafLayout],
    challenges: &FriChallenges<F, D>,
    precomputed_reduced_evals: &[PrecomputedReducedOpenings<F, D>],
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    mut x_index: usize,
    round_proof: &FriQueryRound<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()> {
    batch_fri_verify_initial_proof::<F, C::Hasher, D>(
        degree_bits,
        layouts,
        x_index,
        &round_proof.initial_trees_proof,
        initial_merkle_caps,
    )?;
    let mut n = degree_bits[0];
    // `subgroup_x` is `subgroup[x_index]`, i.e., the actual field element in the domain.
    let mut subgroup_x = F::MULTIPLICATIVE_GROUP_GENERATOR
        * F::primitive_root_of_unity(n).exp_u64(reverse_bits(x_index, n) as u64);

    let mut batch_index = 0;
    // old_eval is the last derived evaluation; it will be checked for consistency with its
    // committed "parent" value in the next iteration.
    let mut old_eval = batch_fri_combine_initial::<F, C, D>(
        instances,
        batch_index,
        &round_proof.initial_trees_proof,
        challenges.fri_alpha,
        subgroup_x,
        &precomputed_reduced_evals[batch_index],
        params,
    );
    batch_index += 1;

    for (i, &arity_bits) in params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;
        let evals = &round_proof.steps[i].evals;

        // Split x_index into the index of the coset x is in, and the index of x within that coset.
        let coset_index = x_index >> arity_bits;
        let x_index_within_coset = x_index & (arity - 1);

        // Check consistency with our old evaluation from the previous round.
        ensure!(evals[x_index_within_coset] == old_eval);

        old_eval = compute_evaluation(
            subgroup_x,
            x_index_within_coset,
            arity_bits,
            evals,
            challenges.fri_betas[i],
        );
        verify_merkle_proof_to_cap::<F, C::Hasher>(
            flatten(evals),
            coset_index,
            &proof.commit_phase_merkle_caps[i],
            &round_proof.steps[i].merkle_proof,
        )?;

        // Update the point x to x^arity.
        subgroup_x = subgroup_x.exp_power_of_2(arity_bits);
        x_index = coset_index;
        n -= arity_bits;

        if batch_index < degree_bits.len() && n == degree_bits[batch_index] {
            let subgroup_x_init = F::MULTIPLICATIVE_GROUP_GENERATOR
                * F::primitive_root_of_unity(n).exp_u64(reverse_bits(x_index, n) as u64);
            let eval = batch_fri_combine_initial::<F, C, D>(
                instances,
                batch_index,
                &round_proof.initial_trees_proof,
                challenges.fri_alpha,
                subgroup_x_init,
                &precomputed_reduced_evals[batch_index],
                params,
            );
            old_eval = old_eval * challenges.fri_betas[i] + eval;
            batch_index += 1;
        }
    }
    // Final check of FRI. After all the reductions, we check that the final polynomial is equal
    // to the one sent by the prover.
    ensure!(
        proof.final_poly.eval(subgroup_x.into()) == old_eval,
        "Final polynomial evaluation is invalid."
    );

    Ok(())
}
