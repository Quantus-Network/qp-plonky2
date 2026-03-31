#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::{
    build_batch_mask_oracle, fri_proof_from_chunks, FriBatchMaskOracle, FriLdeInput,
};
use crate::fri::structure::{FriBatchInfo, FriCoefficient, FriInstanceInfo, FriOpeningExpression};
use crate::fri::{FriFinalPolyLayout, FriParams};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

/// Final batched polynomial kept in the same logical layout used by the disclosed FRI final
/// polynomial and the explicit batch-mask oracle.
#[derive(Clone, Debug)]
struct DecomposedPolynomial<F: Field> {
    layout: FriFinalPolyLayout,
    coeffs: Vec<PolynomialCoeffs<F>>,
}

/// Native reduction result for the PolyFri opening expression path.
///
/// The unmasked polynomial comes only from opening expressions; the masked polynomial adds the
/// explicit pre-alpha batch-mask oracle chunk-wise in the same logical final-poly layout.
#[derive(Debug)]
struct BatchReductionResult<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    unmasked_final: DecomposedPolynomial<F::Extension>,
    masked_final: DecomposedPolynomial<F::Extension>,
    batch_mask_oracle: Option<FriBatchMaskOracle<F, C, D>>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    fn eval_coefficient(coefficient: &FriCoefficient<F, D>, point: F::Extension) -> F::Extension {
        match coefficient {
            FriCoefficient::One => F::Extension::ONE,
            FriCoefficient::PointPower(power) => point.exp_u64(*power as u64),
            FriCoefficient::Constant(constant) => *constant,
        }
    }

    fn opening_expression_poly(
        expression: &FriOpeningExpression<F, D>,
        oracles: &[&Self],
        point: F::Extension,
    ) -> PolynomialCoeffs<F::Extension> {
        expression
            .terms
            .iter()
            .map(|term| {
                let coefficient = Self::eval_coefficient(&term.coefficient, point);
                let poly = &oracles[term.polynomial.oracle_index].polynomials
                    [term.polynomial.polynomial_index];
                let mut scaled = poly.to_extension::<D>();
                scaled *= coefficient;
                scaled
            })
            .sum()
    }

    fn reduce_openings_to_unmasked_final_poly(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        timing: &mut TimingTree,
    ) -> PolynomialCoeffs<F::Extension> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        let mut final_poly = PolynomialCoeffs::empty();
        for FriBatchInfo { point, openings } in &instance.batches {
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} opening expressions", openings.len()),
                alpha.reduce_polys(
                    openings
                        .iter()
                        .map(|expr| Self::opening_expression_poly(expr, oracles, *point))
                )
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        final_poly
    }

    fn decompose_final_poly(
        final_poly: PolynomialCoeffs<F::Extension>,
        layout: FriFinalPolyLayout,
    ) -> DecomposedPolynomial<F::Extension> {
        let coeffs = match layout {
            FriFinalPolyLayout::Single => vec![final_poly],
            FriFinalPolyLayout::Split {
                chunk_degree_bits,
                chunks,
            } => {
                let chunk_len = 1 << chunk_degree_bits;
                let mut padded = final_poly;
                padded
                    .pad(chunk_len * chunks)
                    .expect("Final polynomial exceeds the configured split layout");
                padded.chunks(chunk_len)
            }
        };

        DecomposedPolynomial { layout, coeffs }
    }

    fn add_poly_coeffs(
        lhs: &PolynomialCoeffs<F::Extension>,
        rhs: &PolynomialCoeffs<F::Extension>,
    ) -> PolynomialCoeffs<F::Extension> {
        let len = lhs.len().max(rhs.len());
        let mut coeffs = vec![F::Extension::ZERO; len];
        for (i, &coeff) in lhs.coeffs.iter().enumerate() {
            coeffs[i] += coeff;
        }
        for (i, &coeff) in rhs.coeffs.iter().enumerate() {
            coeffs[i] += coeff;
        }
        PolynomialCoeffs::new(coeffs)
    }

    fn add_batch_mask_to_decomposed_final(
        unmasked_final: &DecomposedPolynomial<F::Extension>,
        batch_mask_oracle: &FriBatchMaskOracle<F, C, D>,
    ) -> DecomposedPolynomial<F::Extension> {
        debug_assert_eq!(unmasked_final.layout, batch_mask_oracle.layout);
        let coeffs = unmasked_final
            .coeffs
            .iter()
            .zip_eq(batch_mask_oracle.coeffs.iter())
            .map(|(chunk, mask)| Self::add_poly_coeffs(chunk, mask))
            .collect();

        DecomposedPolynomial {
            layout: unmasked_final.layout.clone(),
            coeffs,
        }
    }

    fn compose_decomposed_final(
        final_poly: &DecomposedPolynomial<F::Extension>,
    ) -> PolynomialCoeffs<F::Extension> {
        match final_poly.layout {
            FriFinalPolyLayout::Single => final_poly.coeffs[0].clone(),
            FriFinalPolyLayout::Split {
                chunk_degree_bits,
                chunks,
            } => {
                let chunk_len = 1 << chunk_degree_bits;
                let mut coeffs = vec![F::Extension::ZERO; chunk_len * chunks];
                for (chunk_index, chunk) in final_poly.coeffs.iter().enumerate() {
                    let offset = chunk_index * chunk_len;
                    for (coeff_index, &coeff) in chunk.coeffs.iter().enumerate() {
                        coeffs[offset + coeff_index] += coeff;
                    }
                }
                PolynomialCoeffs::new(coeffs)
            }
        }
    }

    fn reduce_openings_with_batch_mask(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
        force_zero_batch_mask: bool,
    ) -> BatchReductionResult<F, C, D> {
        let batch_mask_oracle = fri_params.batch_masking.as_ref().map(|_| {
            let batch_mask_oracle = build_batch_mask_oracle::<F, C, D>(
                fri_params,
                fri_params.config.cap_height,
                force_zero_batch_mask,
                timing,
            );
            challenger.observe_cap(&batch_mask_oracle.cap);
            batch_mask_oracle
        });

        let unmasked_final = Self::decompose_final_poly(
            Self::reduce_openings_to_unmasked_final_poly(instance, oracles, challenger, timing),
            fri_params.batch_mask_layout(),
        );
        let masked_final = batch_mask_oracle
            .as_ref()
            .map(|batch_mask_oracle| {
                Self::add_batch_mask_to_decomposed_final(&unmasked_final, batch_mask_oracle)
            })
            .unwrap_or_else(|| unmasked_final.clone());

        BatchReductionResult {
            unmasked_final,
            masked_final,
            batch_mask_oracle,
        }
    }

    fn final_poly_to_lde_input(
        final_poly: PolynomialCoeffs<F::Extension>,
        fri_params: &FriParams,
    ) -> FriLdeInput<F, D> {
        // Phase 2 keeps the FRI domain tied to the chunk degree. A small batch mask may add a few
        // high coefficients, so we pad directly to the fixed LDE size instead of using
        // `PolynomialCoeffs::lde`, which would otherwise round the polynomial length itself.
        let lde_size = fri_params.lde_size();
        assert!(
            final_poly.len() <= lde_size,
            "FRI batch mask exceeded the configured LDE size"
        );
        let lde_coeffs = final_poly.padded(lde_size);
        let lde_values = lde_coeffs.coset_fft(F::coset_shift().into());
        FriLdeInput {
            layout: fri_params.final_poly_layout.clone(),
            coeffs: vec![lde_coeffs],
            values: vec![lde_values],
        }
    }

    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        if blinding {
            #[cfg(feature = "rand")]
            return Self::lde_blinded_values(polynomials, rate_bits, fft_root_table);
            #[cfg(not(feature = "rand"))]
            {
                assert!(false, "Cannot set blinding without rand feature");
                [].into()
            }
        } else {
            Self::lde_unblinded_values(polynomials, rate_bits, fft_root_table)
        }
    }

    #[cfg(feature = "rand")]
    fn lde_blinded_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..SALT_SIZE)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    fn lde_unblinded_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        let final_poly =
            Self::reduce_openings_to_unmasked_final_poly(instance, oracles, challenger, timing);
        let final_input = timed!(
            timing,
            &format!("perform final FFT {}", fri_params.lde_size()),
            Self::final_poly_to_lde_input(final_poly, fri_params)
        );

        fri_proof_from_chunks::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            final_input,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            None,
            timing,
        )
    }

    /// Expression-aware opening proof entrypoint used by the PolyFri prover path.
    ///
    /// The instance metadata decides whether each logical opening is raw or reconstructed from
    /// split-mask pieces; the oracle inputs stay as raw committed batches.
    pub(crate) fn prove_openings_masked_with_options(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        force_zero_batch_mask: bool,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        let batch_reduction = Self::reduce_openings_with_batch_mask(
            instance,
            oracles,
            challenger,
            fri_params,
            timing,
            force_zero_batch_mask,
        );
        let final_input = timed!(
            timing,
            &format!("perform final FFT {}", fri_params.lde_size()),
            Self::final_poly_to_lde_input(
                Self::compose_decomposed_final(&batch_reduction.masked_final),
                fri_params,
            )
        );
        debug_assert_eq!(
            batch_reduction.unmasked_final.layout,
            batch_reduction.masked_final.layout,
        );

        fri_proof_from_chunks::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            final_input,
            challenger,
            fri_params,
            None,
            None,
            batch_reduction.batch_mask_oracle.as_ref(),
            timing,
        )
    }

    pub fn prove_openings_masked(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        Self::prove_openings_masked_with_options(
            instance, oracles, challenger, fri_params, false, timing,
        )
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::PolynomialBatch;
    use crate::field::extension::Extendable;
    use crate::field::polynomial::PolynomialValues;
    use crate::field::types::Field;
    use crate::fri::proof::{FriChallenges, FriProof};
    use crate::fri::structure::{
        FriBatchInfo, FriOpeningBatch, FriOpeningExpression, FriOpenings, FriOracleInfo,
        FriPolynomialInfo,
    };
    use crate::fri::verifier::{
        eval_batch_mask_at_query_point, eval_masked_final_at_query_point, fri_combine_initial,
        verify_fri_proof, PrecomputedReducedOpenings,
    };
    use crate::fri::{
        FriBatchMaskingParams, FriChallenger, FriConfig, FriFinalPolyLayout, FriParams,
        FriReductionStrategy,
    };
    use crate::iop::challenger::Challenger;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::util::reverse_bits;
    use crate::util::timing::TimingTree;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type FE = <F as Extendable<D>>::Extension;
    type H = <C as GenericConfig<D>>::Hasher;

    fn test_fri_params() -> FriParams {
        FriParams {
            config: FriConfig {
                rate_bits: 1,
                cap_height: 0,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::Fixed(vec![1, 1]),
                num_query_rounds: 4,
            },
            leaf_hiding: false,
            batch_masking: Some(FriBatchMaskingParams { mask_degree: 1 }),
            degree_bits: 4,
            reduction_arity_bits: vec![1, 1],
            final_poly_layout: FriFinalPolyLayout::Split {
                chunk_degree_bits: 2,
                chunks: 2,
            },
        }
    }

    fn test_oracle_and_instance(
        fri_params: &FriParams,
    ) -> (
        PolynomialBatch<F, C, D>,
        crate::fri::structure::FriInstanceInfo<F, D>,
        FriOpenings<F, D>,
        FE,
    ) {
        let mut timing = TimingTree::default();
        let trace = PolynomialValues::new(
            (0..(1 << fri_params.degree_bits))
                .map(F::from_canonical_usize)
                .collect(),
        );
        let oracle = PolynomialBatch::<F, C, D>::from_values(
            vec![trace],
            fri_params.config.rate_bits,
            false,
            fri_params.config.cap_height,
            &mut timing,
            None,
        );

        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_cap(&oracle.merkle_tree.cap);
        let zeta = challenger.get_extension_challenge::<D>();
        let opening = oracle.polynomials[0].to_extension::<D>().eval(zeta);
        let instance = crate::fri::structure::FriInstanceInfo {
            oracles: vec![FriOracleInfo {
                num_polys: 1,
                blinding: false,
            }],
            batches: vec![FriBatchInfo {
                point: zeta,
                openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                    oracle_index: 0,
                    polynomial_index: 0,
                })],
            }],
        };
        let openings = FriOpenings {
            batches: vec![FriOpeningBatch {
                values: vec![opening],
            }],
        };

        (oracle, instance, openings, zeta)
    }

    fn compute_fri_challenges(
        oracle: &PolynomialBatch<F, C, D>,
        openings: &FriOpenings<F, D>,
        proof: &FriProof<F, H, D>,
        fri_params: &FriParams,
    ) -> FriChallenges<F, D> {
        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_cap(&oracle.merkle_tree.cap);
        let _zeta = challenger.get_extension_challenge::<D>();
        challenger.observe_openings(openings);
        if let Some(batch_mask_proof) = &proof.batch_mask_proof {
            challenger.observe_cap(&batch_mask_proof.cap);
        }
        challenger.fri_challenges::<C, D>(
            &proof.commit_phase_merkle_caps,
            &proof.final_polys,
            proof.pow_witness,
            fri_params.degree_bits,
            &fri_params.config,
            None,
            None,
        )
    }

    #[test]
    fn zero_batch_mask_matches_unmasked_final() {
        let fri_params = test_fri_params();
        let (oracle, instance, openings, _zeta) = test_oracle_and_instance(&fri_params);
        let mut timing = TimingTree::default();

        let mut baseline = Challenger::<F, H>::new();
        baseline.observe_cap(&oracle.merkle_tree.cap);
        let _ = baseline.get_extension_challenge::<D>();
        baseline.observe_openings(&openings);

        let mut with_zero_cap = baseline.clone();
        let zero_mask_oracle = crate::fri::prover::build_batch_mask_oracle::<F, C, D>(
            &fri_params,
            0,
            true,
            &mut timing,
        );
        with_zero_cap.observe_cap(&zero_mask_oracle.cap);
        let expected_unmasked = PolynomialBatch::<F, C, D>::reduce_openings_to_unmasked_final_poly(
            &instance,
            &[&oracle],
            &mut with_zero_cap,
            &mut timing,
        );

        let reduction = PolynomialBatch::<F, C, D>::reduce_openings_with_batch_mask(
            &instance,
            &[&oracle],
            &mut baseline,
            &fri_params,
            &mut timing,
            true,
        );
        assert_eq!(
            zero_mask_oracle.cap,
            reduction.batch_mask_oracle.unwrap().cap,
            "zero-mask helper must be transcript-equivalent to the live mask builder",
        );
        assert_eq!(
            expected_unmasked,
            PolynomialBatch::<F, C, D>::compose_decomposed_final(&reduction.masked_final),
        );
    }

    #[test]
    fn observing_batch_mask_cap_changes_alpha() {
        let fri_params = test_fri_params();
        let (oracle, instance, openings, _zeta) = test_oracle_and_instance(&fri_params);
        let mut timing = TimingTree::default();
        let mut prover_challenger = Challenger::<F, H>::new();
        prover_challenger.observe_cap(&oracle.merkle_tree.cap);
        let _ = prover_challenger.get_extension_challenge::<D>();
        prover_challenger.observe_openings(&openings);

        let proof = PolynomialBatch::<F, C, D>::prove_openings_masked_with_options(
            &instance,
            &[&oracle],
            &mut prover_challenger,
            &fri_params,
            true,
            &mut timing,
        );
        let alpha = compute_fri_challenges(&oracle, &openings, &proof, &fri_params).fri_alpha;

        let mut mutated_proof = proof.clone();
        mutated_proof.batch_mask_proof.as_mut().unwrap().cap.0[0].elements[0] += F::ONE;
        let mutated_alpha =
            compute_fri_challenges(&oracle, &openings, &mutated_proof, &fri_params).fri_alpha;

        assert_ne!(alpha, mutated_alpha);
    }

    #[test]
    fn queried_batch_mask_values_reconstruct_masked_first_layer() {
        let fri_params = test_fri_params();
        let (oracle, instance, openings, _zeta) = test_oracle_and_instance(&fri_params);
        let mut timing = TimingTree::default();
        let mut prover_challenger = Challenger::<F, H>::new();
        prover_challenger.observe_cap(&oracle.merkle_tree.cap);
        let _ = prover_challenger.get_extension_challenge::<D>();
        prover_challenger.observe_openings(&openings);

        let proof = PolynomialBatch::<F, C, D>::prove_openings_masked(
            &instance,
            &[&oracle],
            &mut prover_challenger,
            &fri_params,
            &mut timing,
        );
        let fri_challenges = compute_fri_challenges(&oracle, &openings, &proof, &fri_params);
        let precomputed =
            PrecomputedReducedOpenings::from_os_and_alpha(&openings, fri_challenges.fri_alpha);
        let query_round_index = 0;
        let x_index = fri_challenges.fri_query_indices[query_round_index];
        let subgroup_x = F::MULTIPLICATIVE_GROUP_GENERATOR
            * F::primitive_root_of_unity(fri_params.lde_bits())
                .exp_u64(reverse_bits(x_index, fri_params.lde_bits()) as u64);
        let round_proof = &proof.query_round_proofs[query_round_index];
        let expected_unmasked_final = fri_combine_initial::<F, C, D>(
            &instance,
            &round_proof.initial_trees_proof,
            fri_challenges.fri_alpha,
            subgroup_x,
            &precomputed,
            &fri_params,
        );
        let batch_mask_query =
            &proof.batch_mask_proof.as_ref().unwrap().query_openings[query_round_index];
        let batch_mask_eval =
            eval_batch_mask_at_query_point(batch_mask_query, subgroup_x.into(), &fri_params);
        let expected_masked_final = eval_masked_final_at_query_point::<F, D>(
            expected_unmasked_final,
            Some(batch_mask_eval),
        );
        let arity = 1 << fri_params.reduction_arity_bits[0];
        let x_index_within_coset = x_index & (arity - 1);

        assert_eq!(
            round_proof.steps[0].evals[x_index_within_coset],
            expected_masked_final,
        );
    }

    #[test]
    fn corrupt_batch_mask_query_value_fails_verification() -> Result<()> {
        let fri_params = test_fri_params();
        let (oracle, instance, openings, _zeta) = test_oracle_and_instance(&fri_params);
        let mut timing = TimingTree::default();
        let mut prover_challenger = Challenger::<F, H>::new();
        prover_challenger.observe_cap(&oracle.merkle_tree.cap);
        let _ = prover_challenger.get_extension_challenge::<D>();
        prover_challenger.observe_openings(&openings);

        let proof = PolynomialBatch::<F, C, D>::prove_openings_masked(
            &instance,
            &[&oracle],
            &mut prover_challenger,
            &fri_params,
            &mut timing,
        );
        let fri_challenges = compute_fri_challenges(&oracle, &openings, &proof, &fri_params);
        verify_fri_proof::<F, C, D>(
            &instance,
            &openings,
            &fri_challenges,
            &[oracle.merkle_tree.cap.clone()],
            &proof,
            &fri_params,
        )?;

        let mut corrupted_proof = proof.clone();
        corrupted_proof
            .batch_mask_proof
            .as_mut()
            .unwrap()
            .query_openings[0]
            .values[0] += FE::ONE;
        assert!(verify_fri_proof::<F, C, D>(
            &instance,
            &openings,
            &fri_challenges,
            &[oracle.merkle_tree.cap.clone()],
            &corrupted_proof,
            &fri_params,
        )
        .is_err());

        Ok(())
    }

    #[test]
    fn corrupt_batch_mask_merkle_path_fails_verification() -> Result<()> {
        let fri_params = test_fri_params();
        let (oracle, instance, openings, _zeta) = test_oracle_and_instance(&fri_params);
        let mut timing = TimingTree::default();
        let mut prover_challenger = Challenger::<F, H>::new();
        prover_challenger.observe_cap(&oracle.merkle_tree.cap);
        let _ = prover_challenger.get_extension_challenge::<D>();
        prover_challenger.observe_openings(&openings);

        let proof = PolynomialBatch::<F, C, D>::prove_openings_masked(
            &instance,
            &[&oracle],
            &mut prover_challenger,
            &fri_params,
            &mut timing,
        );
        let fri_challenges = compute_fri_challenges(&oracle, &openings, &proof, &fri_params);

        let mut corrupted_proof = proof.clone();
        corrupted_proof
            .batch_mask_proof
            .as_mut()
            .unwrap()
            .query_openings[0]
            .merkle_proof
            .siblings[0]
            .elements[0] += F::ONE;
        assert!(verify_fri_proof::<F, C, D>(
            &instance,
            &openings,
            &fri_challenges,
            &[oracle.merkle_tree.cap.clone()],
            &corrupted_proof,
            &fri_params,
        )
        .is_err());

        Ok(())
    }
}
