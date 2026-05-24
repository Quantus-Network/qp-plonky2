//! FRI proof shape validation.

#[cfg(not(feature = "std"))]
use alloc::vec;

use anyhow::ensure;

use crate::config::GenericConfig;
use crate::field::extension::Extendable;
use crate::fri::FriParams;
use crate::fri_proof::{
    FriBatchMaskProof, FriBatchMaskQuery, FriProof, FriQueryRound, FriQueryStep,
};
use crate::fri_structure::{FriChallenges, FriInstanceInfo, FriOpenings};
use crate::hash_types::RichField;
use crate::merkle_tree::MerkleCap;
use crate::plonk_common::salt_size;

pub fn validate_fri_proof_shape<F, C, const D: usize>(
    proof: &FriProof<F, C::Hasher, D>,
    instance: &FriInstanceInfo<F, D>,
    params: &FriParams,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    validate_batch_fri_proof_shape::<F, C, D>(proof, &[instance.clone()], params)
}

pub fn validate_batch_fri_proof_shape<F, C, const D: usize>(
    proof: &FriProof<F, C::Hasher, D>,
    instances: &[FriInstanceInfo<F, D>],
    params: &FriParams,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let FriProof {
        commit_phase_merkle_caps,
        batch_mask_proof,
        query_round_proofs,
        final_polys,
        pow_witness: _pow_witness,
    } = proof;

    check_batch_instance_references(instances)?;

    let cap_height = params.config.cap_height;
    ensure!(
        commit_phase_merkle_caps.len() == params.reduction_arity_bits.len(),
        "FRI commit-phase cap count does not match reduction arity count"
    );
    ensure!(
        query_round_proofs.len() == params.config.num_query_rounds,
        "FRI query round proof count does not match config"
    );
    for cap in commit_phase_merkle_caps {
        ensure!(cap.height() == cap_height);
    }

    match (&params.batch_masking, batch_mask_proof) {
        (
            Some(_),
            Some(FriBatchMaskProof {
                cap,
                query_openings,
            }),
        ) => {
            ensure!(cap.height() == cap_height);
            ensure!(query_openings.len() == params.config.num_query_rounds);
            for query_opening in query_openings {
                let FriBatchMaskQuery {
                    values,
                    merkle_proof,
                } = query_opening;
                ensure!(values.len() == params.final_poly_chunks());
                ensure!(merkle_proof.len() + cap_height == params.lde_bits());
            }
        }
        (None, None) => {}
        (Some(_), None) => ensure!(false, "Missing explicit batch-mask proof for masked FRI."),
        (None, Some(_)) => ensure!(
            false,
            "Unexpected batch-mask proof for FRI params without batch masking."
        ),
    }

    for query_round in query_round_proofs {
        let FriQueryRound {
            initial_trees_proof,
            steps,
        } = query_round;

        let oracle_count = initial_trees_proof.evals_proofs.len();
        let mut leaf_len = vec![0; oracle_count];
        for inst in instances {
            ensure!(oracle_count == inst.oracles.len());
            for (i, oracle) in inst.oracles.iter().enumerate() {
                leaf_len[i] += oracle.num_polys + salt_size(oracle.blinding && params.leaf_hiding);
            }
        }
        for (i, (leaf, merkle_proof)) in initial_trees_proof.evals_proofs.iter().enumerate() {
            ensure!(leaf.len() == leaf_len[i]);
            ensure!(merkle_proof.len() + cap_height == params.lde_bits());
        }

        ensure!(steps.len() == params.reduction_arity_bits.len());
        let mut codeword_len_bits = params.lde_bits();
        for (step, arity_bits) in steps.iter().zip(&params.reduction_arity_bits) {
            let FriQueryStep {
                evals,
                merkle_proof,
            } = step;

            let arity = 1 << arity_bits;
            codeword_len_bits -= arity_bits;

            ensure!(evals.len() == arity);
            ensure!(merkle_proof.len() + cap_height == codeword_len_bits);
        }
    }

    ensure!(final_polys.layout == params.final_poly_layout);
    ensure!(final_polys.chunks.len() == params.final_poly_chunks());
    for chunk in &final_polys.chunks {
        ensure!(chunk.len() == params.final_poly_len());
    }

    Ok(())
}

fn check_batch_instance_references<F: RichField + Extendable<D>, const D: usize>(
    instances: &[FriInstanceInfo<F, D>],
) -> anyhow::Result<()> {
    ensure!(!instances.is_empty(), "FRI instance list cannot be empty");
    let oracle_count = instances[0].oracles.len();
    let mut total_polys_by_oracle = vec![0; oracle_count];
    for instance in instances {
        ensure!(
            instance.oracles.len() == oracle_count,
            "FRI instances must share the same oracle count"
        );
        for (oracle_index, oracle) in instance.oracles.iter().enumerate() {
            total_polys_by_oracle[oracle_index] += oracle.num_polys;
        }
    }

    for instance in instances {
        for batch in &instance.batches {
            for expression in &batch.openings {
                for term in &expression.terms {
                    let total_polys = total_polys_by_oracle
                        .get(term.polynomial.oracle_index)
                        .ok_or_else(|| anyhow::anyhow!("FRI oracle index out of range"))?;
                    ensure!(
                        term.polynomial.polynomial_index < *total_polys,
                        "FRI polynomial index out of range"
                    );
                }
            }
        }
    }

    Ok(())
}

pub fn validate_fri_auxiliary_shape<F, C, const D: usize>(
    instance: &FriInstanceInfo<F, D>,
    openings: &FriOpenings<F, D>,
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    validate_batch_fri_auxiliary_shape::<F, C, D>(
        &[params.degree_bits],
        core::slice::from_ref(instance),
        core::slice::from_ref(openings),
        challenges,
        initial_merkle_caps,
        proof,
        params,
    )
}

pub fn validate_batch_fri_auxiliary_shape<F, C, const D: usize>(
    degree_bits: &[usize],
    instances: &[FriInstanceInfo<F, D>],
    openings: &[FriOpenings<F, D>],
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    validate_batch_fri_proof_shape::<F, C, D>(proof, instances, params)?;

    ensure!(!instances.is_empty(), "FRI instance list cannot be empty");
    ensure!(
        degree_bits.len() == instances.len(),
        "FRI degree-bit count does not match instance count"
    );
    ensure!(
        openings.len() == instances.len(),
        "FRI openings count does not match instance count"
    );
    ensure!(
        challenges.fri_query_indices.len() == params.config.num_query_rounds,
        "FRI query index count does not match config"
    );
    ensure!(
        challenges.fri_betas.len() == params.reduction_arity_bits.len(),
        "FRI beta count does not match reduction arity count"
    );

    let oracle_count = instances[0].oracles.len();
    ensure!(
        initial_merkle_caps.len() == oracle_count,
        "FRI initial cap count does not match oracle count"
    );
    for instance in instances {
        ensure!(
            instance.oracles.len() == oracle_count,
            "FRI instances must share the same oracle count"
        );
    }

    for (instance, openings) in instances.iter().zip(openings) {
        ensure!(
            openings.batches.len() == instance.batches.len(),
            "FRI opening batch count does not match instance batch count"
        );
        for (opening_batch, instance_batch) in openings.batches.iter().zip(&instance.batches) {
            ensure!(
                opening_batch.values.len() == instance_batch.openings.len(),
                "FRI opening value count does not match expression count"
            );
        }
    }

    let mut current_degree_bits = degree_bits[0]
        .checked_add(params.config.rate_bits)
        .ok_or_else(|| anyhow::anyhow!("FRI degree-bit overflow"))?;
    let mut next_instance = 1;
    for &arity_bits in &params.reduction_arity_bits {
        ensure!(
            current_degree_bits >= arity_bits,
            "FRI reduction arity exceeds current degree bits"
        );
        current_degree_bits -= arity_bits;
        if next_instance < degree_bits.len() {
            let next_degree_bits = degree_bits[next_instance]
                .checked_add(params.config.rate_bits)
                .ok_or_else(|| anyhow::anyhow!("FRI degree-bit overflow"))?;
            if current_degree_bits == next_degree_bits {
                next_instance += 1;
            }
        }
    }
    ensure!(
        next_instance == instances.len(),
        "FRI batch reductions did not consume every instance"
    );

    Ok(())
}
