//! FRI proof shape validation.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::ensure;

use crate::config::GenericConfig;
use crate::field::extension::Extendable;
use crate::fri::FriParams;
use crate::fri_proof::{FriProof, FriQueryRound, FriQueryStep};
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
    validate_batch_fri_proof_shape::<F, C, D>(proof, core::slice::from_ref(instance), params)
}

/// Validates that every opening term references an in-range oracle and polynomial index, and
/// returns the expected per-oracle leaf length each initial-tree-proof leaf must have.
///
/// In batch FRI a single oracle's leaf concatenates polynomials from every instance, so both the
/// polynomial-index bound and the leaf length sum `num_polys` (plus salt) across instances; this
/// reduces to a single instance for plain FRI. Checking the bound here keeps the verifier from
/// indexing an oracle leaf out of bounds on malformed metadata (#64696).
fn checked_leaf_lengths<F, const D: usize>(
    instances: &[FriInstanceInfo<F, D>],
    leaf_hiding: bool,
) -> anyhow::Result<Vec<usize>>
where
    F: RichField + Extendable<D>,
{
    let Some(first) = instances.first() else {
        return Ok(Vec::new());
    };
    let oracle_count = first.oracles.len();
    let mut total_num_polys = vec![0usize; oracle_count];
    let mut leaf_len = vec![0usize; oracle_count];
    for inst in instances {
        ensure!(
            inst.oracles.len() == oracle_count,
            "FRI instances disagree on oracle count"
        );
        for (i, oracle) in inst.oracles.iter().enumerate() {
            total_num_polys[i] += oracle.num_polys;
            leaf_len[i] += oracle.num_polys + salt_size(oracle.blinding && leaf_hiding);
        }
    }
    for inst in instances {
        for batch in &inst.batches {
            for expression in &batch.openings {
                for term in &expression.terms {
                    let oracle_index = term.polynomial.oracle_index;
                    ensure!(oracle_index < oracle_count, "FRI oracle index out of range");
                    ensure!(
                        term.polynomial.polynomial_index < total_num_polys[oracle_index],
                        "FRI polynomial index out of range"
                    );
                }
            }
        }
    }
    Ok(leaf_len)
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
        query_round_proofs,
        final_poly,
        pow_witness: _pow_witness,
    } = proof;

    let leaf_len = checked_leaf_lengths(instances, params.leaf_hiding)?;

    let cap_height = params.config.cap_height;
    // Compare lengths rather than `height()`, which panics on a non-power-of-two cap.
    for cap in commit_phase_merkle_caps {
        ensure!(cap.len() == 1 << cap_height);
    }

    for query_round in query_round_proofs {
        let FriQueryRound {
            initial_trees_proof,
            steps,
        } = query_round;

        ensure!(initial_trees_proof.evals_proofs.len() == leaf_len.len());
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

    ensure!(final_poly.coeffs.len() == params.final_poly_len());

    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::field::types::Field;
    use crate::fri_structure::{
        FriBatchInfo, FriOpeningExpression, FriOracleInfo, FriPolynomialInfo,
    };

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    fn single_oracle_instance(num_polys: usize, polynomial_index: usize) -> FriInstanceInfo<F, D> {
        FriInstanceInfo {
            oracles: vec![FriOracleInfo {
                num_polys,
                blinding: false,
            }],
            batches: vec![FriBatchInfo {
                point: <F as Extendable<D>>::Extension::ZERO,
                openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                    oracle_index: 0,
                    polynomial_index,
                })],
            }],
        }
    }

    /// #64696: an opening term whose `polynomial_index` exceeds the oracle's `num_polys` must be
    /// rejected before `unsalted_eval` indexes the leaf out of bounds.
    #[test]
    fn rejects_out_of_range_polynomial_index() {
        let inst = single_oracle_instance(1, 5);
        assert!(checked_leaf_lengths(core::slice::from_ref(&inst), false).is_err());
    }

    #[test]
    fn leaf_length_includes_salt_only_for_blinded_oracles() {
        let mut inst = single_oracle_instance(3, 2);
        assert_eq!(
            checked_leaf_lengths(core::slice::from_ref(&inst), true).unwrap(),
            vec![3]
        );
        inst.oracles[0].blinding = true;
        assert_eq!(
            checked_leaf_lengths(core::slice::from_ref(&inst), true).unwrap(),
            vec![3 + salt_size(true)]
        );
    }
}
