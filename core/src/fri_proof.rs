//! FRI proof types shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::config::Hasher;
use crate::field::extension::{flatten, unflatten, Extendable};
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::fri::{FriFinalPolyLayout, FriParams};
use crate::fri_structure::FriChallenges;
use crate::hash::path_compression::{compress_merkle_proofs, decompress_merkle_proofs};
use crate::hash_types::RichField;
use crate::merkle_proofs::MerkleProof;
use crate::merkle_tree::MerkleCap;
use crate::plonk_common::salt_size;
use crate::proof::{FriInferredElements, ProofChallenges};

/// Evaluations and Merkle proof produced by the prover in a FRI query step.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriQueryStep<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub evals: Vec<F::Extension>,
    pub merkle_proof: MerkleProof<F, H>,
}

/// Evaluations and Merkle proofs of the original set of polynomials,
/// before they are combined into a composition polynomial.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriInitialTreeProof<F: RichField, H: Hasher<F>> {
    pub evals_proofs: Vec<(Vec<F>, MerkleProof<F, H>)>,
}

impl<F: RichField, H: Hasher<F>> FriInitialTreeProof<F, H> {
    pub fn unsalted_eval(&self, oracle_index: usize, poly_index: usize, salted: bool) -> F {
        self.unsalted_evals(oracle_index, salted)[poly_index]
    }

    pub fn unsalted_evals(&self, oracle_index: usize, salted: bool) -> &[F] {
        let evals = &self.evals_proofs[oracle_index].0;
        &evals[..evals.len() - salt_size(salted)]
    }
}

/// Proof for a FRI query round.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriQueryRound<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub initial_trees_proof: FriInitialTreeProof<F, H>,
    pub steps: Vec<FriQueryStep<F, H, D>>,
}

/// Compressed proof of the FRI query rounds.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedFriQueryRounds<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// Query indices.
    pub indices: Vec<usize>,
    /// Map from initial indices `i` to the `FriInitialProof` for the `i`th leaf.
    pub initial_trees_proofs: HashMap<usize, FriInitialTreeProof<F, H>>,
    /// For each FRI query step, a map from indices `i` to the `FriQueryStep` for the `i`th leaf.
    pub steps: Vec<HashMap<usize, FriQueryStep<F, H, D>>>,
}

/// Authenticated opening of the explicit batch-mask oracle at one queried FRI point.
///
/// The values are chunked according to `FriFinalPolyLayout` so the verifier can combine them into
/// the logical `R(x)` using the same layout math as the final split polynomial path.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriBatchMaskQuery<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub values: Vec<F::Extension>,
    pub merkle_proof: MerkleProof<F, H>,
}

/// Transcript-visible commitment and query openings for the explicit FRI batch-mask oracle.
///
/// This oracle must be observed before sampling `fri_alpha`; its authenticated query values are
/// later added to the verifier's unmasked opening reduction at each FRI query point.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriBatchMaskProof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub cap: MerkleCap<F, H>,
    pub query_openings: Vec<FriBatchMaskQuery<F, H, D>>,
}

fn compress_batch_mask_proof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
    batch_mask_proof: Option<FriBatchMaskProof<F, H, D>>,
    indices: &[usize],
    cap_height: usize,
) -> Option<FriBatchMaskProof<F, H, D>> {
    batch_mask_proof.map(|mut batch_mask_proof| {
        debug_assert_eq!(batch_mask_proof.query_openings.len(), indices.len());
        let proofs = batch_mask_proof
            .query_openings
            .iter()
            .map(|query_opening| query_opening.merkle_proof.clone())
            .collect::<Vec<_>>();
        let compressed_proofs = compress_merkle_proofs(cap_height, indices, &proofs);
        for (query_opening, compressed_proof) in batch_mask_proof
            .query_openings
            .iter_mut()
            .zip(compressed_proofs)
        {
            query_opening.merkle_proof = compressed_proof;
        }
        batch_mask_proof
    })
}

fn decompress_batch_mask_proof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
    batch_mask_proof: Option<FriBatchMaskProof<F, H, D>>,
    indices: &[usize],
    height: usize,
    cap_height: usize,
) -> Option<FriBatchMaskProof<F, H, D>> {
    batch_mask_proof.map(|mut batch_mask_proof| {
        debug_assert_eq!(batch_mask_proof.query_openings.len(), indices.len());
        let leaves = batch_mask_proof
            .query_openings
            .iter()
            .map(|query_opening| flatten(&query_opening.values))
            .collect::<Vec<_>>();
        let compressed_proofs = batch_mask_proof
            .query_openings
            .iter()
            .map(|query_opening| query_opening.merkle_proof.clone())
            .collect::<Vec<_>>();
        let decompressed_proofs =
            decompress_merkle_proofs(&leaves, indices, &compressed_proofs, height, cap_height);
        for (query_opening, decompressed_proof) in batch_mask_proof
            .query_openings
            .iter_mut()
            .zip(decompressed_proofs)
        {
            query_opening.merkle_proof = decompressed_proof;
        }
        batch_mask_proof
    })
}

/// The reduced polynomial disclosed after the FRI commit phase.
///
/// `Split` keeps each coefficient chunk within the original degree cap while preserving a single
/// logical polynomial via powers of `X^{chunk_degree}`.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriFinalPolys<F: RichField + Extendable<D>, const D: usize> {
    pub layout: FriFinalPolyLayout,
    pub chunks: Vec<PolynomialCoeffs<F::Extension>>,
}

impl<F: RichField + Extendable<D>, const D: usize> FriFinalPolys<F, D> {
    pub fn from_single(chunk: PolynomialCoeffs<F::Extension>) -> Self {
        Self {
            layout: FriFinalPolyLayout::Single,
            chunks: vec![chunk],
        }
    }
}

pub fn combine_final_poly_chunks<F: RichField + Extendable<D>, const D: usize>(
    layout: &FriFinalPolyLayout,
    values: &[F::Extension],
    point: F::Extension,
) -> F::Extension {
    match layout {
        FriFinalPolyLayout::Single => values[0],
        FriFinalPolyLayout::Split {
            chunk_degree_bits,
            chunks,
        } => {
            debug_assert_eq!(*chunks, values.len());
            let point_stride = point.exp_power_of_2(*chunk_degree_bits);
            let mut weight = F::Extension::ONE;
            let mut sum = F::Extension::ZERO;
            for value in values {
                sum += weight * *value;
                weight *= point_stride;
            }
            sum
        }
    }
}

pub fn eval_final_polys_at_point<F: RichField + Extendable<D>, const D: usize>(
    final_polys: &FriFinalPolys<F, D>,
    point: F::Extension,
) -> F::Extension {
    let values = final_polys
        .chunks
        .iter()
        .map(|chunk| chunk.eval(point))
        .collect::<Vec<_>>();
    combine_final_poly_chunks::<F, D>(&final_polys.layout, &values, point)
}

/// The full FRI proof.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriProof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// A Merkle cap for each reduced polynomial in the commit phase.
    pub commit_phase_merkle_caps: Vec<MerkleCap<F, H>>,
    /// Explicit batch-mask oracle commitment and per-query openings, if enabled by the FRI params.
    pub batch_mask_proof: Option<FriBatchMaskProof<F, H, D>>,
    /// Query rounds proofs
    pub query_round_proofs: Vec<FriQueryRound<F, H, D>>,
    /// The final reduced polynomial, kept either raw or as degree-bounded chunks.
    pub final_polys: FriFinalPolys<F, D>,
    /// Witness showing that the prover did PoW.
    pub pow_witness: F,
}

/// Compressed FRI proof with reduced Merkle proof sizes.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedFriProof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// A Merkle cap for each reduced polynomial in the commit phase.
    pub commit_phase_merkle_caps: Vec<MerkleCap<F, H>>,
    /// Explicit batch-mask oracle commitment and per-query openings, if enabled by the FRI params.
    pub batch_mask_proof: Option<FriBatchMaskProof<F, H, D>>,
    /// Compressed query rounds proof.
    pub query_round_proofs: CompressedFriQueryRounds<F, H, D>,
    /// The final reduced polynomial, kept either raw or as degree-bounded chunks.
    pub final_polys: FriFinalPolys<F, D>,
    /// Witness showing that the prover did PoW.
    pub pow_witness: F,
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> FriProof<F, H, D> {
    /// Compress all the Merkle paths in the FRI proof and remove duplicate indices.
    pub fn compress(self, indices: &[usize], params: &FriParams) -> CompressedFriProof<F, H, D> {
        let FriProof {
            commit_phase_merkle_caps,
            batch_mask_proof,
            query_round_proofs,
            final_polys,
            pow_witness,
            ..
        } = self;
        let cap_height = params.config.cap_height;
        let reduction_arity_bits = &params.reduction_arity_bits;
        let num_reductions = reduction_arity_bits.len();
        let num_initial_trees = query_round_proofs[0].initial_trees_proof.evals_proofs.len();

        // "Transpose" the query round proofs, so that information for each Merkle tree is collected together.
        let mut initial_trees_indices = vec![vec![]; num_initial_trees];
        let mut initial_trees_leaves = vec![vec![]; num_initial_trees];
        let mut initial_trees_proofs = vec![vec![]; num_initial_trees];
        let mut steps_indices = vec![vec![]; num_reductions];
        let mut steps_evals = vec![vec![]; num_reductions];
        let mut steps_proofs = vec![vec![]; num_reductions];

        for (mut index, qrp) in indices.iter().cloned().zip(&query_round_proofs) {
            let FriQueryRound {
                initial_trees_proof,
                steps,
            } = qrp.clone();
            for (i, (leaves_data, proof)) in
                initial_trees_proof.evals_proofs.into_iter().enumerate()
            {
                initial_trees_indices[i].push(index);
                initial_trees_leaves[i].push(leaves_data);
                initial_trees_proofs[i].push(proof);
            }
            for (i, query_step) in steps.into_iter().enumerate() {
                let index_within_coset = index & ((1 << reduction_arity_bits[i]) - 1);
                index >>= reduction_arity_bits[i];
                steps_indices[i].push(index);
                let mut evals = query_step.evals;
                // Remove the element that can be inferred.
                evals.remove(index_within_coset);
                steps_evals[i].push(evals);
                steps_proofs[i].push(query_step.merkle_proof);
            }
        }

        // Compress all Merkle proofs.
        let initial_trees_proofs = initial_trees_indices
            .iter()
            .zip(initial_trees_proofs)
            .map(|(is, ps)| compress_merkle_proofs(cap_height, is, &ps))
            .collect::<Vec<_>>();
        let steps_proofs = steps_indices
            .iter()
            .zip(steps_proofs)
            .map(|(is, ps)| compress_merkle_proofs(cap_height, is, &ps))
            .collect::<Vec<_>>();

        let mut compressed_query_proofs = CompressedFriQueryRounds {
            indices: indices.to_vec(),
            initial_trees_proofs: HashMap::with_capacity(indices.len()),
            steps: vec![HashMap::new(); num_reductions],
        };

        // Replace the query round proofs with the compressed versions.
        for (i, mut index) in indices.iter().copied().enumerate() {
            let initial_proof = FriInitialTreeProof {
                evals_proofs: (0..num_initial_trees)
                    .map(|j| {
                        (
                            initial_trees_leaves[j][i].clone(),
                            initial_trees_proofs[j][i].clone(),
                        )
                    })
                    .collect(),
            };
            compressed_query_proofs
                .initial_trees_proofs
                .entry(index)
                .or_insert(initial_proof);
            for j in 0..num_reductions {
                index >>= reduction_arity_bits[j];
                let query_step = FriQueryStep {
                    evals: steps_evals[j][i].clone(),
                    merkle_proof: steps_proofs[j][i].clone(),
                };
                compressed_query_proofs.steps[j]
                    .entry(index)
                    .or_insert(query_step);
            }
        }

        CompressedFriProof {
            commit_phase_merkle_caps,
            batch_mask_proof: compress_batch_mask_proof(batch_mask_proof, indices, cap_height),
            query_round_proofs: compressed_query_proofs,
            final_polys,
            pow_witness,
        }
    }
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> CompressedFriProof<F, H, D> {
    /// Decompress all the Merkle paths in the FRI proof and reinsert duplicate indices.
    pub fn decompress(
        self,
        challenges: &ProofChallenges<F, D>,
        fri_inferred_elements: FriInferredElements<F, D>,
        params: &FriParams,
    ) -> FriProof<F, H, D> {
        let CompressedFriProof {
            commit_phase_merkle_caps,
            batch_mask_proof,
            query_round_proofs,
            final_polys,
            pow_witness,
            ..
        } = self;
        let FriChallenges {
            fri_query_indices: indices,
            ..
        } = &challenges.fri_challenges;
        let mut fri_inferred_elements = fri_inferred_elements.0.into_iter();
        let cap_height = params.config.cap_height;
        let reduction_arity_bits = &params.reduction_arity_bits;
        let num_reductions = reduction_arity_bits.len();
        let num_initial_trees = query_round_proofs
            .initial_trees_proofs
            .values()
            .next()
            .unwrap()
            .evals_proofs
            .len();

        // "Transpose" the query round proofs, so that information for each Merkle tree is collected together.
        let mut initial_trees_indices = vec![vec![]; num_initial_trees];
        let mut initial_trees_leaves = vec![vec![]; num_initial_trees];
        let mut initial_trees_proofs = vec![vec![]; num_initial_trees];
        let mut steps_indices = vec![vec![]; num_reductions];
        let mut steps_evals = vec![vec![]; num_reductions];
        let mut steps_proofs = vec![vec![]; num_reductions];
        let height = params.degree_bits + params.config.rate_bits;
        let heights = reduction_arity_bits
            .iter()
            .scan(height, |acc, &bits| {
                *acc -= bits;
                Some(*acc)
            })
            .collect::<Vec<_>>();

        // Holds the `evals` vectors that have already been reconstructed at each reduction depth.
        let mut evals_by_depth =
            vec![HashMap::<usize, Vec<_>>::new(); params.reduction_arity_bits.len()];
        for &(mut index) in indices {
            let initial_trees_proof = query_round_proofs.initial_trees_proofs[&index].clone();
            for (i, (leaves_data, proof)) in
                initial_trees_proof.evals_proofs.into_iter().enumerate()
            {
                initial_trees_indices[i].push(index);
                initial_trees_leaves[i].push(leaves_data);
                initial_trees_proofs[i].push(proof);
            }
            for i in 0..num_reductions {
                let index_within_coset = index & ((1 << reduction_arity_bits[i]) - 1);
                index >>= reduction_arity_bits[i];
                let FriQueryStep {
                    mut evals,
                    merkle_proof,
                } = query_round_proofs.steps[i][&index].clone();
                steps_indices[i].push(index);
                if let Some(v) = evals_by_depth[i].get(&index) {
                    // If this index has already been seen, get `evals` from the `HashMap`.
                    evals = v.to_vec();
                } else {
                    // Otherwise insert the next inferred element.
                    evals.insert(index_within_coset, fri_inferred_elements.next().unwrap());
                    evals_by_depth[i].insert(index, evals.clone());
                }
                steps_evals[i].push(flatten(&evals));
                steps_proofs[i].push(merkle_proof);
            }
        }

        // Decompress all Merkle proofs.
        let initial_trees_proofs = initial_trees_indices
            .iter()
            .zip(initial_trees_leaves.iter())
            .zip(initial_trees_proofs)
            .map(|((is, ls), ps)| decompress_merkle_proofs(ls, is, &ps, height, cap_height))
            .collect::<Vec<_>>();
        let steps_proofs = steps_indices
            .iter()
            .zip(steps_evals.iter())
            .zip(steps_proofs)
            .zip(heights)
            .map(|(((is, ls), ps), h)| decompress_merkle_proofs(ls, is, &ps, h, cap_height))
            .collect::<Vec<_>>();

        let mut decompressed_query_proofs = Vec::with_capacity(num_reductions);
        for i in 0..indices.len() {
            let initial_trees_proof = FriInitialTreeProof {
                evals_proofs: (0..num_initial_trees)
                    .map(|j| {
                        (
                            initial_trees_leaves[j][i].clone(),
                            initial_trees_proofs[j][i].clone(),
                        )
                    })
                    .collect(),
            };
            let steps = (0..num_reductions)
                .map(|j| FriQueryStep {
                    evals: unflatten(&steps_evals[j][i]),
                    merkle_proof: steps_proofs[j][i].clone(),
                })
                .collect();
            decompressed_query_proofs.push(FriQueryRound {
                initial_trees_proof,
                steps,
            })
        }

        FriProof {
            commit_phase_merkle_caps,
            batch_mask_proof: decompress_batch_mask_proof(
                batch_mask_proof,
                indices,
                params.lde_bits(),
                cap_height,
            ),
            query_round_proofs: decompressed_query_proofs,
            final_polys,
            pow_witness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{eval_final_polys_at_point, FriFinalPolys};
    use crate::field::extension::{Extendable, FieldExtension};
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::polynomial::PolynomialCoeffs;
    use crate::field::types::Field;
    use crate::fri::FriFinalPolyLayout;

    const D: usize = 2;
    type F = GoldilocksField;
    type FE = <F as Extendable<D>>::Extension;

    #[test]
    fn eval_split_final_polys_reconstructs_logical_polynomial() {
        let chunk0 = PolynomialCoeffs::new(vec![
            <FE as FieldExtension<D>>::from_basefield(F::ONE),
            <FE as FieldExtension<D>>::from_basefield(F::from_canonical_u64(2)),
        ]);
        let chunk1 = PolynomialCoeffs::new(vec![
            <FE as FieldExtension<D>>::from_basefield(F::from_canonical_u64(3)),
            <FE as FieldExtension<D>>::from_basefield(F::from_canonical_u64(4)),
        ]);
        let final_polys = FriFinalPolys::<F, D> {
            layout: FriFinalPolyLayout::Split {
                chunk_degree_bits: 2,
                chunks: 2,
            },
            chunks: vec![chunk0.clone(), chunk1.clone()],
        };
        let point = <FE as FieldExtension<D>>::from_basefield(F::from_canonical_u64(7));

        let expected = chunk0.eval(point) + point.exp_u64(4) * chunk1.eval(point);
        assert_eq!(eval_final_polys_at_point(&final_polys, point), expected);
    }
}
