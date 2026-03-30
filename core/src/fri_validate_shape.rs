//! FRI proof shape validation.

#[cfg(not(feature = "std"))]
use alloc::vec;

use anyhow::ensure;

use crate::config::GenericConfig;
use crate::field::extension::Extendable;
use crate::fri::FriParams;
use crate::fri_proof::{FriBatchMaskProof, FriBatchMaskQuery, FriProof, FriQueryRound, FriQueryStep};
use crate::fri_structure::FriInstanceInfo;
use crate::hash_types::RichField;
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

    let cap_height = params.config.cap_height;
    for cap in commit_phase_merkle_caps {
        ensure!(cap.height() == cap_height);
    }

    match (&params.batch_masking, batch_mask_proof) {
        (Some(_), Some(FriBatchMaskProof { cap, query_openings })) => {
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
