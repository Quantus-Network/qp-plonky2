#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::types::{Field, Sample};
use plonky2_maybe_rayon::*;

use crate::field::extension::{flatten, unflatten, Extendable};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::{
    FriBatchMaskProof, FriBatchMaskQuery, FriFinalPolys, FriInitialTreeProof, FriProof,
    FriQueryRound, FriQueryStep,
};
use crate::fri::{FriConfig, FriFinalPolyLayout, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::{MerkleCap, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{reverse_index_bits_in_place, transpose};

/// The LDE input for the final FRI composition polynomial.
///
/// Phase 2 still folds a single masked codeword through the commit phase, but the disclosed
/// reduced polynomial may be chunked afterward to keep each transmitted chunk within the original
/// degree cap.
#[derive(Debug)]
pub struct FriLdeInput<F: RichField + Extendable<D>, const D: usize> {
    pub layout: FriFinalPolyLayout,
    pub coeffs: Vec<PolynomialCoeffs<F::Extension>>,
    pub values: Vec<PolynomialValues<F::Extension>>,
}

/// Explicit oracle committed before `fri_alpha` is sampled in the PolyFri native path.
///
/// Each chunk is evaluated over the same FRI query domain as the final masked codeword so the
/// verifier can authenticate `R(x)` with a separate Merkle opening at every query point.
#[derive(Debug)]
pub struct FriBatchMaskOracle<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub coeffs: Vec<PolynomialCoeffs<F::Extension>>,
    pub values: Vec<PolynomialValues<F::Extension>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub cap: MerkleCap<F, C::Hasher>,
    pub layout: FriFinalPolyLayout,
}

#[cfg(feature = "rand")]
fn sample_batch_mask_chunk<F: RichField + Extendable<D>, const D: usize>(
    mask_degree: usize,
    force_zero: bool,
) -> PolynomialCoeffs<F::Extension> {
    if force_zero {
        PolynomialCoeffs::new(vec![F::Extension::ZERO; mask_degree + 1])
    } else {
        PolynomialCoeffs::new(F::Extension::rand_vec(mask_degree + 1))
    }
}

#[cfg(not(feature = "rand"))]
fn sample_batch_mask_chunk<F: RichField + Extendable<D>, const D: usize>(
    _mask_degree: usize,
    _force_zero: bool,
) -> PolynomialCoeffs<F::Extension> {
    panic!("Cannot enable FRI batch masking without the rand feature");
}

pub(crate) fn build_batch_mask_oracle<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    fri_params: &FriParams,
    cap_height: usize,
    force_zero_mask: bool,
    timing: &mut TimingTree,
) -> FriBatchMaskOracle<F, C, D> {
    let batch_masking = fri_params
        .batch_masking
        .as_ref()
        .expect("Batch-mask oracle requested without FRI batch masking parameters");
    let chunk_count = fri_params.final_poly_chunks();
    let lde_size = fri_params.lde_size();
    let coeffs = timed!(
        timing,
        "sample batch mask oracle",
        (0..chunk_count)
            .map(|_| sample_batch_mask_chunk::<F, D>(batch_masking.mask_degree, force_zero_mask))
            .collect::<Vec<_>>()
    );
    let values = timed!(
        timing,
        "evaluate batch mask oracle",
        coeffs
            .iter()
            .map(|coeffs| coeffs.padded(lde_size).coset_fft(F::coset_shift().into()))
            .collect::<Vec<_>>()
    );
    let value_rows = values
        .iter()
        .map(|poly_values| poly_values.values.clone())
        .collect::<Vec<_>>();
    let mut leaves = timed!(
        timing,
        "transpose batch mask oracle",
        transpose(&value_rows)
            .into_iter()
            .map(|chunk_values| flatten(&chunk_values))
            .collect::<Vec<_>>()
    );
    reverse_index_bits_in_place(&mut leaves);
    let merkle_tree = timed!(
        timing,
        "build batch mask tree",
        MerkleTree::new(leaves, cap_height)
    );
    let cap = merkle_tree.cap.clone();

    FriBatchMaskOracle {
        coeffs,
        values,
        merkle_tree,
        cap,
        layout: fri_params.batch_mask_layout(),
    }
}

pub(crate) fn batch_mask_query_openings<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    oracle: &FriBatchMaskOracle<F, C, D>,
    indices: &[usize],
) -> Vec<FriBatchMaskQuery<F, C::Hasher, D>> {
    indices
        .iter()
        .map(|&x_index| FriBatchMaskQuery {
            values: unflatten(oracle.merkle_tree.get(x_index)),
            merkle_proof: oracle.merkle_tree.prove(x_index),
        })
        .collect()
}

/// Builds a FRI proof.
pub fn fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    // Coefficients of the polynomial on which the LDT is performed. Only the first `1/rate` coefficients are non-zero.
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    // Evaluation of the polynomial on the large domain.
    lde_polynomial_values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    fri_proof_from_chunks::<F, C, D>(
        initial_merkle_trees,
        FriLdeInput {
            layout: FriFinalPolyLayout::Single,
            coeffs: vec![lde_polynomial_coeffs],
            values: vec![lde_polynomial_values],
        },
        challenger,
        fri_params,
        final_poly_coeff_len,
        max_num_query_steps,
        None,
        timing,
    )
}

pub(crate) type FriCommitedTrees<F, C, const D: usize> = (
    Vec<MerkleTree<F, <C as GenericConfig<D>>::Hasher>>,
    PolynomialCoeffs<<F as Extendable<D>>::Extension>,
);

pub fn final_poly_coeff_len(mut degree_bits: usize, reduction_arity_bits: &Vec<usize>) -> usize {
    for arity_bits in reduction_arity_bits {
        degree_bits -= *arity_bits;
    }
    1 << degree_bits
}

pub fn fri_proof_from_chunks<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    final_input: FriLdeInput<F, D>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    batch_mask_oracle: Option<&FriBatchMaskOracle<F, C, D>>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    assert_eq!(
        final_input.coeffs.len(),
        1,
        "Phase 2 keeps one masked FRI codeword and only chunks the disclosed final polynomial",
    );
    assert_eq!(
        final_input.values.len(),
        1,
        "Phase 2 keeps one masked FRI codeword and only chunks the disclosed final polynomial",
    );

    let FriLdeInput {
        layout,
        mut coeffs,
        mut values,
    } = final_input;
    let lde_polynomial_coeffs = coeffs.pop().unwrap();
    let lde_polynomial_values = values.pop().unwrap();
    let n = lde_polynomial_values.len();
    assert_eq!(lde_polynomial_coeffs.len(), n);

    // Commit phase
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        fri_committed_trees::<F, C, D>(
            lde_polynomial_coeffs,
            lde_polynomial_values,
            challenger,
            fri_params,
            max_num_query_steps,
        )
    );
    let final_polys =
        decompose_final_polynomial::<F, D>(final_coeffs, layout, final_poly_coeff_len);
    observe_final_polys::<F, C, D>(challenger, &final_polys, final_poly_coeff_len);

    // PoW phase
    let pow_witness = timed!(
        timing,
        "find proof-of-work witness",
        fri_proof_of_work::<F, C, D>(challenger, &fri_params.config)
    );

    // Query phase
    let (query_indices, query_round_proofs) =
        fri_prover_query_rounds::<F, C, D>(initial_merkle_trees, &trees, challenger, n, fri_params);
    let batch_mask_proof = batch_mask_oracle.map(|batch_mask_oracle| FriBatchMaskProof {
        cap: batch_mask_oracle.cap.clone(),
        query_openings: batch_mask_query_openings(batch_mask_oracle, &query_indices),
    });

    FriProof {
        commit_phase_merkle_caps: trees.iter().map(|t| t.cap.clone()).collect(),
        batch_mask_proof,
        query_round_proofs,
        final_polys,
        pow_witness,
    }
}

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    mut values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    for arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1 << arity_bits;

        reverse_index_bits_in_place(&mut values.values);
        let chunked_values = values
            .values
            .par_chunks(arity)
            .map(|chunk: &[F::Extension]| flatten(chunk))
            .collect();
        let tree = MerkleTree::<F, C::Hasher>::new(chunked_values, fri_params.config.cap_height);

        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        coeffs = PolynomialCoeffs::new(
            coeffs
                .coeffs
                .par_chunks_exact(arity)
                .map(|chunk| reduce_with_powers(chunk, beta))
                .collect::<Vec<_>>(),
        );
        shift = shift.exp_u64(arity as u64);
        values = coeffs.coset_fft(shift.into())
    }

    // When verifying this proof in a circuit with a different number of query steps,
    // we need the challenger to stay in sync with the verifier. Therefore, the challenger
    // must observe the additional hash caps and generate dummy challenges.
    if let Some(step_count) = max_num_query_steps {
        let cap_len = (1 << fri_params.config.cap_height) * NUM_HASH_OUT_ELTS;
        let zero_cap = vec![F::ZERO; cap_len];
        for _ in fri_params.reduction_arity_bits.len()..step_count {
            challenger.observe_elements(&zero_cap);
            challenger.get_extension_challenge::<D>();
        }
    }

    // The coefficients being removed here should always be zero.
    coeffs
        .coeffs
        .truncate(coeffs.len() >> fri_params.config.rate_bits);

    (trees, coeffs)
}

fn decompose_final_polynomial<F: RichField + Extendable<D>, const D: usize>(
    final_coeffs: PolynomialCoeffs<F::Extension>,
    layout: FriFinalPolyLayout,
    final_poly_coeff_len: Option<usize>,
) -> FriFinalPolys<F, D> {
    match layout {
        FriFinalPolyLayout::Single => FriFinalPolys::from_single(final_coeffs),
        FriFinalPolyLayout::Split {
            chunk_degree_bits,
            chunks,
        } => {
            let chunk_len = final_poly_coeff_len.unwrap_or(1 << chunk_degree_bits);
            let mut padded = final_coeffs;
            padded
                .pad(chunk_len * chunks)
                .expect("Chunked final polynomial exceeds the configured layout");
            FriFinalPolys {
                layout: FriFinalPolyLayout::Split {
                    chunk_degree_bits,
                    chunks,
                },
                chunks: padded.chunks(chunk_len),
            }
        }
    }
}

fn observe_final_polys<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    challenger: &mut Challenger<F, C::Hasher>,
    final_polys: &FriFinalPolys<F, D>,
    final_poly_coeff_len: Option<usize>,
) {
    for chunk in &final_polys.chunks {
        challenger.observe_extension_elements(&chunk.coeffs);
        if let Some(len) = final_poly_coeff_len {
            for _ in chunk.coeffs.len()..len {
                challenger.observe_extension_element(&F::Extension::ZERO);
            }
        }
    }
}

/// Performs the proof-of-work (a.k.a. grinding) step of the FRI protocol. Returns the PoW witness.
pub(crate) fn fri_proof_of_work<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    challenger: &mut Challenger<F, C::Hasher>,
    config: &FriConfig,
) -> F {
    let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;

    // The easiest implementation would be repeatedly clone our Challenger. With each clone, we'd
    // observe an incrementing PoW witness, then get the PoW response. If it contained sufficient
    // leading zeros, we'd end the search, and store this clone as our new challenger.
    //
    // However, performance is critical here. We want to avoid cloning Challenger, particularly
    // since it stores vectors, which means allocations. We'd like a more compact state to clone.
    //
    // We know that a duplex will be performed right after we send the PoW witness, so we can ignore
    // any output_buffer, which will be invalidated. We also know
    // input_buffer.len() < H::Permutation::WIDTH, an invariant of Challenger.
    //
    // We separate the duplex operation into two steps, one which can be performed now, and the
    // other which depends on the PoW witness candidate. The first step is the overwrite our sponge
    // state with any inputs (excluding the PoW witness candidate). The second step is to overwrite
    // one more element of our sponge state with the candidate, then apply the permutation,
    // obtaining our duplex's post-state which contains the PoW response.
    let mut duplex_intermediate_state = challenger.sponge_state();
    let witness_input_pos = challenger.input_buffer().len();
    duplex_intermediate_state.set_from_iter(challenger.input_buffer().iter().copied(), 0);

    let pow_witness = (0..=F::NEG_ONE.to_canonical_u64())
        .into_par_iter()
        .find_any(|&candidate| {
            let mut duplex_state = duplex_intermediate_state;
            duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
            duplex_state.permute();
            let pow_response = duplex_state.squeeze().iter().last().unwrap();
            let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
            leading_zeros >= min_leading_zeros
        })
        .map(F::from_canonical_u64)
        .expect("Proof of work failed. This is highly unlikely!");

    // Recompute pow_response using our normal Challenger code, and make sure it matches.
    challenger.observe_element(pow_witness);
    let pow_response = challenger.get_challenge();
    let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
    assert!(leading_zeros >= min_leading_zeros);
    pow_witness
}

fn fri_prover_query_rounds<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    challenger: &mut Challenger<F, C::Hasher>,
    n: usize,
    fri_params: &FriParams,
) -> (Vec<usize>, Vec<FriQueryRound<F, C::Hasher, D>>) {
    let query_indices = challenger
        .get_n_challenges(fri_params.config.num_query_rounds)
        .into_par_iter()
        .map(|rand| rand.to_canonical_u64() as usize % n)
        .collect::<Vec<_>>();
    let query_round_proofs = query_indices
        .par_iter()
        .map(|&x_index| {
            fri_prover_query_round::<F, C, D>(initial_merkle_trees, trees, x_index, fri_params)
        })
        .collect();
    (query_indices, query_round_proofs)
}

fn fri_prover_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    mut x_index: usize,
    fri_params: &FriParams,
) -> FriQueryRound<F, C::Hasher, D> {
    let mut query_steps = Vec::new();
    let initial_proof = initial_merkle_trees
        .iter()
        .map(|t| (t.get(x_index).to_vec(), t.prove(x_index)))
        .collect::<Vec<_>>();
    for (i, tree) in trees.iter().enumerate() {
        let arity_bits = fri_params.reduction_arity_bits[i];
        let evals = unflatten(tree.get(x_index >> arity_bits));
        let merkle_proof = tree.prove(x_index >> arity_bits);

        query_steps.push(FriQueryStep {
            evals,
            merkle_proof,
        });

        x_index >>= arity_bits;
    }
    FriQueryRound {
        initial_trees_proof: FriInitialTreeProof {
            evals_proofs: initial_proof,
        },
        steps: query_steps,
    }
}
