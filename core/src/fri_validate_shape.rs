//! FRI proof shape validation.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::ensure;

use crate::config::{GenericConfig, Hasher};
use crate::field::extension::Extendable;
use crate::fri::FriParams;
use crate::fri_proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri_structure::{
    FriChallenges, FriInstanceInfo, FriOpenings, FriOracleInfo, FriPolynomialInfo,
};
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
    validate_batch_fri_proof_shape::<F, C, D>(proof, core::slice::from_ref(instance), params)
}

/// Validates that every opened polynomial references an in-range oracle and polynomial index, and
/// returns the expected per-oracle leaf length each initial-tree-proof leaf must have.
///
/// In batch FRI a single oracle's leaf concatenates polynomials from every instance, so both the
/// polynomial-index bound and the leaf length sum `num_polys` (plus salt) across instances; this
/// reduces to a single instance for plain FRI. Taking oracles and opened polynomials rather than
/// instances lets the native and the in-circuit verifiers share it, so neither path can index an
/// oracle leaf out of bounds and panic on malformed metadata (#64696).
pub fn checked_leaf_lengths<'a>(
    oracles_per_instance: &[&[FriOracleInfo]],
    opened_polynomials: impl IntoIterator<Item = &'a FriPolynomialInfo>,
    leaf_hiding: bool,
) -> anyhow::Result<Vec<usize>> {
    let Some(first) = oracles_per_instance.first() else {
        return Ok(Vec::new());
    };
    let oracle_count = first.len();
    let mut total_num_polys = vec![0usize; oracle_count];
    let mut leaf_len = vec![0usize; oracle_count];
    for oracles in oracles_per_instance {
        ensure!(
            oracles.len() == oracle_count,
            "FRI instances disagree on oracle count"
        );
        for (i, oracle) in oracles.iter().enumerate() {
            // The salt is trimmed per oracle, so every instance must agree on whether it exists.
            let salt = salt_size(oracle.blinding && leaf_hiding);
            ensure!(
                salt == salt_size(first[i].blinding && leaf_hiding),
                "FRI instances disagree on oracle blinding"
            );
            total_num_polys[i] += oracle.num_polys;
            leaf_len[i] += oracle.num_polys + salt;
        }
    }
    for polynomial in opened_polynomials {
        let oracle_index = polynomial.oracle_index;
        ensure!(oracle_index < oracle_count, "FRI oracle index out of range");
        ensure!(
            polynomial.polynomial_index < total_num_polys[oracle_index],
            "FRI polynomial index out of range"
        );
    }
    Ok(leaf_len)
}

fn instance_leaf_lengths<F, const D: usize>(
    instances: &[FriInstanceInfo<F, D>],
    leaf_hiding: bool,
) -> anyhow::Result<Vec<usize>>
where
    F: RichField + Extendable<D>,
{
    let oracles_per_instance: Vec<&[FriOracleInfo]> = instances
        .iter()
        .map(|inst| inst.oracles.as_slice())
        .collect();
    let opened_polynomials = instances
        .iter()
        .flat_map(|inst| &inst.batches)
        .flat_map(|batch| &batch.openings)
        .flat_map(|expression| &expression.terms)
        .map(|term| &term.polynomial);
    checked_leaf_lengths(&oracles_per_instance, opened_polynomials, leaf_hiding)
}

/// Validates a single FRI initial-tree proof's leaf shapes against `instances`, so that later
/// `unsalted_eval` indexing cannot go out of bounds (#64696).
///
/// The compressed verifier evaluates opening expressions during inference (`get_inferred_elements`)
/// *before* full proof-shape validation runs, so it must call this first. Only leaf contents are
/// checked (not Merkle-proof depth), which is correct for the compressed representation where
/// sibling paths are pruned.
pub fn validate_fri_initial_proof_shape<F, H, const D: usize>(
    initial_trees_proof: &FriInitialTreeProof<F, H>,
    instances: &[FriInstanceInfo<F, D>],
    leaf_hiding: bool,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    H: Hasher<F>,
{
    let leaf_len = instance_leaf_lengths(instances, leaf_hiding)?;
    ensure!(
        initial_trees_proof.evals_proofs.len() == leaf_len.len(),
        "FRI oracle count mismatch"
    );
    for ((leaf, _merkle_proof), expected) in initial_trees_proof.evals_proofs.iter().zip(&leaf_len)
    {
        ensure!(leaf.len() == *expected, "FRI leaf length mismatch");
    }
    Ok(())
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

    let leaf_len = instance_leaf_lengths(instances, params.leaf_hiding)?;

    check_fri_proof_counts(
        commit_phase_merkle_caps.len(),
        query_round_proofs.len(),
        params,
    )
    .map_err(anyhow::Error::msg)?;
    let cap_height = params.config.cap_height;
    for cap in commit_phase_merkle_caps {
        ensure!(cap.has_height(cap_height));
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

/// Checks the proof's own list lengths against the parameters. Every commit-phase cap is absorbed
/// into the transcript, so a cap without a matching reduction step would be unbound by any query,
/// handing the prover free Fiat–Shamir grinding.
pub fn check_fri_proof_counts(
    commit_cap_count: usize,
    query_round_count: usize,
    params: &FriParams,
) -> Result<(), &'static str> {
    if commit_cap_count != params.reduction_arity_bits.len() {
        return Err("Number of commit-phase caps does not match the reduction schedule.");
    }
    if query_round_count != params.config.num_query_rounds {
        return Err("Number of query rounds does not match config.");
    }
    Ok(())
}

/// Batch FRI folds each instance in once the codeword has shrunk to that instance's degree, so
/// the degrees must be heights the reduction schedule reaches, in fold order, starting at the FRI
/// degree. Anything else would leave an instance unfolded or index a Merkle layer that is never
/// opened.
pub fn check_batch_fri_degree_bits(
    degree_bits: &[usize],
    instance_count: usize,
    params: &FriParams,
) -> Result<(), &'static str> {
    if degree_bits.len() != instance_count {
        return Err("Number of degree bits does not match the instance count.");
    }
    let mut reached_degrees =
        core::iter::once(params.degree_bits).chain(params.reduction_arity_bits.iter().scan(
            Some(params.degree_bits),
            |degree, &arity_bits| {
                *degree = degree.and_then(|degree| degree.checked_sub(arity_bits));
                *degree
            },
        ));
    let is_fold_subsequence = degree_bits.first() == Some(&params.degree_bits)
        && degree_bits
            .iter()
            .all(|degree| reached_degrees.by_ref().any(|reached| reached == *degree));
    if !is_fold_subsequence {
        return Err(
            "Batch FRI degree bits must be reached by the reduction schedule in fold order.",
        );
    }
    Ok(())
}

/// The number of entries in each batch of each item: `openings.len()` per instance batch on the
/// expected side and `values.len()` per opening batch on the claimed side, for either crate's
/// native or target types.
pub fn per_batch_counts<Item, Batch>(
    items: &[Item],
    batches: impl Fn(&Item) -> &[Batch],
    count: impl Fn(&Batch) -> usize,
) -> Vec<Vec<usize>> {
    items
        .iter()
        .map(|item| batches(item).iter().map(&count).collect())
        .collect()
}

/// Checks the list lengths that both the native and the in-circuit FRI verifiers zip against each
/// other: one initial cap per oracle, one beta per reduction step, one query index per round, and
/// one claimed opening per opening expression (`expected_opening_counts` and
/// `claimed_opening_counts` are per instance, per batch). A short list would otherwise silently
/// leave part of the proof unchecked.
pub fn check_fri_verifier_input_counts(
    oracle_count: usize,
    initial_cap_count: usize,
    beta_count: usize,
    query_index_count: usize,
    expected_opening_counts: &[Vec<usize>],
    claimed_opening_counts: &[Vec<usize>],
    params: &FriParams,
) -> Result<(), &'static str> {
    if initial_cap_count != oracle_count {
        return Err("Number of initial Merkle caps does not match the oracle count.");
    }
    if beta_count != params.reduction_arity_bits.len() {
        return Err("Number of FRI betas does not match the reduction schedule.");
    }
    if query_index_count != params.config.num_query_rounds {
        return Err("Number of FRI query indices does not match config.");
    }
    if claimed_opening_counts.len() != expected_opening_counts.len() {
        return Err("Number of FRI openings does not match the instance count.");
    }
    for (expected, claimed) in expected_opening_counts.iter().zip(claimed_opening_counts) {
        if expected.len() != claimed.len() {
            return Err("Number of opening batches does not match the instance.");
        }
        if expected != claimed {
            return Err("Number of claimed openings does not match the batch.");
        }
    }
    Ok(())
}

/// Validates the caller-supplied verifier inputs that `verify_fri_proof` zips against the proof,
/// plus the value-level checks only the native verifier can make: cap heights and in-domain
/// query indices.
pub fn validate_fri_verifier_inputs<F, H, const D: usize>(
    instances: &[FriInstanceInfo<F, D>],
    openings: &[FriOpenings<F, D>],
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, H>],
    params: &FriParams,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    H: Hasher<F>,
{
    check_fri_verifier_input_counts(
        instances.first().map_or(0, |inst| inst.oracles.len()),
        initial_merkle_caps.len(),
        challenges.fri_betas.len(),
        challenges.fri_query_indices.len(),
        &per_batch_counts(
            instances,
            |inst| &inst.batches,
            |batch| batch.openings.len(),
        ),
        &per_batch_counts(
            openings,
            |opening| &opening.batches,
            |batch| batch.values.len(),
        ),
        params,
    )
    .map_err(anyhow::Error::msg)?;

    for cap in initial_merkle_caps {
        ensure!(cap.has_height(params.config.cap_height));
    }
    ensure!(
        challenges
            .fri_query_indices
            .iter()
            .all(|&index| index < params.lde_size()),
        "FRI query index outside the LDE domain."
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::field::types::Field;
    use crate::fri::{FriConfig, FriReductionStrategy};
    use crate::fri_structure::{
        FriBatchInfo, FriOpeningExpression, FriOracleInfo, FriPolynomialInfo,
    };
    use crate::merkle_proofs::MerkleProof;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type H = <C as GenericConfig<D>>::Hasher;

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

    fn single_oracle_proof(leaf_len: usize) -> FriInitialTreeProof<F, H> {
        FriInitialTreeProof {
            evals_proofs: vec![(vec![F::ZERO; leaf_len], MerkleProof { siblings: vec![] })],
        }
    }

    /// #64696: an opening term whose `polynomial_index` exceeds the oracle's `num_polys` must be
    /// rejected before `unsalted_eval` indexes the leaf out of bounds (the compressed-verifier
    /// inference path that the audit PoC exercised).
    #[test]
    fn rejects_out_of_range_polynomial_index() {
        let inst = single_oracle_instance(1, 5);
        let proof = single_oracle_proof(1);
        assert!(validate_fri_initial_proof_shape::<F, H, D>(
            &proof,
            core::slice::from_ref(&inst),
            false
        )
        .is_err());
    }

    /// #64696: an in-range index but a proof leaf shorter than the declared `num_polys` must also be
    /// rejected (a short attacker-supplied leaf for a large declared oracle).
    #[test]
    fn rejects_short_leaf() {
        let inst = single_oracle_instance(3, 2);
        let proof = single_oracle_proof(1);
        assert!(validate_fri_initial_proof_shape::<F, H, D>(
            &proof,
            core::slice::from_ref(&inst),
            false
        )
        .is_err());
    }

    #[test]
    fn accepts_consistent_shape() {
        let inst = single_oracle_instance(3, 2);
        let proof = single_oracle_proof(3);
        validate_fri_initial_proof_shape::<F, H, D>(&proof, core::slice::from_ref(&inst), false)
            .unwrap();
    }

    /// The salt is trimmed per oracle by the current instance's `blinding`, so instances that
    /// disagree on it would make the leaf layout and the opening reads disagree too.
    #[test]
    fn rejects_mixed_blinding_per_oracle() {
        let unsalted = single_oracle_instance(1, 0);
        let mut salted = single_oracle_instance(1, 1);
        salted.oracles[0].blinding = true;
        let mixed = [salted.clone(), unsalted];
        // A leaf the mixed layout would size exactly, so only the blinding check can reject it.
        let mixed_leaf = single_oracle_proof(1 + 1 + salt_size(true));
        let error =
            validate_fri_initial_proof_shape::<F, H, D>(&mixed_leaf, &mixed, true).unwrap_err();
        assert_eq!(
            error.downcast_ref::<&str>(),
            Some(&"FRI instances disagree on oracle blinding")
        );
        // Without leaf hiding no oracle is salted, so blinding does not shape the leaf.
        assert!(validate_fri_initial_proof_shape::<F, H, D>(
            &single_oracle_proof(2),
            &mixed,
            false
        )
        .is_ok());
        assert!(validate_fri_initial_proof_shape::<F, H, D>(
            &single_oracle_proof(2 * (1 + salt_size(true))),
            &[salted.clone(), salted],
            true
        )
        .is_ok());
    }

    #[test]
    fn batch_degree_bits_must_follow_the_fold_schedule() {
        let params = FriParams {
            config: FriConfig {
                rate_bits: 1,
                cap_height: 0,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::Fixed(vec![1, 2, 1]),
                num_query_rounds: 2,
            },
            leaf_hiding: false,
            degree_bits: 6,
            reduction_arity_bits: vec![1, 2, 1],
        };
        let check = |degree_bits: &[usize]| {
            check_batch_fri_degree_bits(degree_bits, degree_bits.len(), &params)
        };
        let not_reached =
            Err("Batch FRI degree bits must be reached by the reduction schedule in fold order.");
        assert_eq!(check(&[6]), Ok(()));
        assert_eq!(check(&[6, 5]), Ok(()));
        assert_eq!(check(&[6, 3, 2]), Ok(()));
        assert_eq!(check(&[6, 2]), Ok(()));
        assert_eq!(check(&[5]), not_reached);
        assert_eq!(check(&[6, 6]), not_reached);
        assert_eq!(check(&[6, 4]), not_reached);
        assert_eq!(check(&[6, 2, 3]), not_reached);
        assert_eq!(check(&[6, 5, 5]), not_reached);
        assert_eq!(
            check_batch_fri_degree_bits(&[6], 2, &params),
            Err("Number of degree bits does not match the instance count.")
        );
    }

    #[test]
    fn proof_counts_must_match_params() {
        let params = FriParams {
            config: FriConfig {
                rate_bits: 1,
                cap_height: 0,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::Fixed(vec![1, 1]),
                num_query_rounds: 3,
            },
            leaf_hiding: false,
            degree_bits: 4,
            reduction_arity_bits: vec![1, 1],
        };
        assert_eq!(check_fri_proof_counts(2, 3, &params), Ok(()));
        assert_eq!(
            check_fri_proof_counts(3, 3, &params),
            Err("Number of commit-phase caps does not match the reduction schedule.")
        );
        assert_eq!(
            check_fri_proof_counts(2, 2, &params),
            Err("Number of query rounds does not match config.")
        );
    }

    #[test]
    fn verifier_input_counts_must_all_match() {
        let params = FriParams {
            config: FriConfig {
                rate_bits: 1,
                cap_height: 0,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::Fixed(vec![1, 1]),
                num_query_rounds: 3,
            },
            leaf_hiding: false,
            degree_bits: 4,
            reduction_arity_bits: vec![1, 1],
        };
        let expected = vec![vec![2, 1], vec![1]];
        let check = |caps: usize, betas: usize, queries: usize, claimed: &[Vec<usize>]| {
            check_fri_verifier_input_counts(2, caps, betas, queries, &expected, claimed, &params)
        };
        assert_eq!(check(2, 2, 3, &expected), Ok(()));
        assert_eq!(
            check(1, 2, 3, &expected),
            Err("Number of initial Merkle caps does not match the oracle count.")
        );
        assert_eq!(
            check(2, 1, 3, &expected),
            Err("Number of FRI betas does not match the reduction schedule.")
        );
        assert_eq!(
            check(2, 2, 2, &expected),
            Err("Number of FRI query indices does not match config.")
        );
        assert_eq!(
            check(2, 2, 3, &expected[..1]),
            Err("Number of FRI openings does not match the instance count.")
        );
        assert_eq!(
            check(2, 2, 3, &[vec![2], vec![1]]),
            Err("Number of opening batches does not match the instance.")
        );
        assert_eq!(
            check(2, 2, 3, &[vec![2, 1], vec![2]]),
            Err("Number of claimed openings does not match the batch.")
        );
    }
}
