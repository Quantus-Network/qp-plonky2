use std::panic::{catch_unwind, AssertUnwindSafe};

use plonky2::batch_fri::oracle::BatchFriOracle;
use plonky2::batch_fri::prover::batch_fri_proof;
use plonky2::batch_fri::verifier::verify_batch_fri_proof;
use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::proof::{
    FriChallenges, FriFinalPolys, FriInitialTreeProof, FriProof, FriQueryRound,
};
use plonky2::fri::{
    structure::{
        FriBatchInfo, FriInstanceInfo, FriOpeningBatch, FriOpeningExpression, FriOpenings,
        FriOracleInfo, FriPolynomialInfo,
    },
    validate_batch_fri_auxiliary_shape, validate_fri_auxiliary_shape, FriBatchMaskingParams,
    FriChallenger, FriConfig, FriFinalPolyLayout, FriParams, FriReductionStrategy,
};
use plonky2::gates::coset_interpolation::CosetInterpolationGate;
use plonky2::gates::exponentiation::ExponentiationGate;
use plonky2::gates::gate::{Gate, GateRef};
use plonky2::gates::noop::NoopGate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::merkle_proofs::{verify_merkle_proof_to_cap, MerkleProof, MerkleProofTarget};
use plonky2::hash::merkle_tree::{MerkleCap, MerkleTree};
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::hash::poseidon2::Poseidon2Hash;
use plonky2::iop::challenger::Challenger;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::WitnessGeneratorRef;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData, ProverOnlyCircuitData};
use plonky2::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
use plonky2::plonk::prover::prove;
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars};
use plonky2::util::serialization::{
    Buffer, DefaultGateSerializer, DefaultGeneratorSerializer, IoResult, Write,
};
use plonky2::util::strided_view::PackedStridedView;
use plonky2::util::timing::TimingTree;
use qp_plonky2_core::ZkMode;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type FF = <C as GenericConfig<D>>::FE;
type H = <C as GenericConfig<D>>::Hasher;

#[derive(Clone, Debug)]
struct GateIdCollisionWeakGate;

#[derive(Clone, Debug)]
struct GateIdCollisionStrongGate;

impl Gate<F, D> for GateIdCollisionWeakGate {
    fn id(&self) -> String {
        "GateIdCollision".to_string()
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self)
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<FF> {
        vec![vars.local_wires[0]]
    }

    fn eval_unfiltered_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        vec![vars.local_wires[0]]
    }

    fn generators(&self, _row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        Vec::new()
    }

    fn num_wires(&self) -> usize {
        1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        1
    }
}

impl Gate<F, D> for GateIdCollisionStrongGate {
    fn id(&self) -> String {
        "GateIdCollision".to_string()
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self)
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<FF> {
        vec![vars.local_wires[0] - FF::ONE]
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let one = builder.constant_extension(FF::ONE);
        vec![builder.sub_extension(vars.local_wires[0], one)]
    }

    fn generators(&self, _row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        Vec::new()
    }

    fn num_wires(&self) -> usize {
        1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        1
    }
}

#[test]
fn malformed_low_high_generator_rejects_unsupported_widths() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        let _ = builder.split_low_high(x, 64, 64);
    }));

    assert!(result.is_err());
}

#[test]
fn malformed_low_high_generator_valid_width_generates_witness() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let x = builder.add_virtual_target();
    let (low, high) = builder.split_low_high(x, 3, 6);

    let data = builder.mock_build::<C>();
    let mut pw = PartialWitness::new();
    pw.set_target(x, F::from_canonical_u64(45))?;

    let witness = data.generate_witness(pw);
    assert_eq!(witness.get_target(low), F::from_canonical_u64(5));
    assert_eq!(witness.get_target(high), F::from_canonical_u64(5));

    Ok(())
}

#[test]
fn zero_denominator_extension_division_returns_err() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let denominator = builder.add_virtual_extension_target();
    let _inverse = builder.inverse_extension(denominator);

    let data = builder.build::<C>();
    let mut pw = PartialWitness::new();
    pw.set_extension_target(denominator, FF::ZERO)?;

    let result = catch_unwind(AssertUnwindSafe(|| data.prove(pw)));
    assert!(result.is_ok(), "zero denominator must not panic");
    assert!(result.unwrap().is_err());

    Ok(())
}

#[test]
fn nonzero_denominator_extension_division_still_proves() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let denominator = builder.add_virtual_extension_target();
    let inverse = builder.inverse_extension(denominator);
    let expected = builder.constant_extension(FF::ONE);
    let product = builder.mul_extension(denominator, inverse);
    builder.connect_extension(product, expected);

    let data = builder.build::<C>();
    let mut pw = PartialWitness::new();
    pw.set_extension_target(denominator, FF::from_canonical_u64(7))?;

    let proof = data.prove(pw)?;
    data.verify(proof)
}

#[test]
fn oversized_exp_from_bits_rejected() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let config = CircuitConfig::standard_recursion_config();
        let capacity = ExponentiationGate::<F, D>::new_from_config(&config).num_power_bits;
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let base = builder.constant(F::TWO);
        let mut bits = Vec::new();
        for _ in 0..=capacity {
            bits.push(builder.constant_bool(false));
        }
        let _ = builder.exp_from_bits(base, bits.iter());
    }));

    assert!(result.is_err());
}

fn exp_from_bits_with_len_proves(bit_len: usize) -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let base = builder.constant(F::TWO);
    let mut bits = Vec::new();
    for i in 0..bit_len {
        bits.push(builder.constant_bool(i == 0));
    }
    let result = builder.exp_from_bits(base, bits.iter());
    let expected = builder.constant(F::TWO);
    builder.connect(result, expected);

    let data = builder.build::<C>();
    let proof = data.prove(PartialWitness::new())?;
    data.verify(proof)
}

#[test]
fn max_capacity_exp_from_bits_still_proves() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let capacity = ExponentiationGate::<F, D>::new_from_config(&config).num_power_bits;
    exp_from_bits_with_len_proves(capacity)
}

#[test]
fn below_capacity_exp_from_bits_still_proves() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let capacity = ExponentiationGate::<F, D>::new_from_config(&config).num_power_bits;
    exp_from_bits_with_len_proves(capacity - 1)
}

fn add_zero_value_coset_interpolation_circuit(
    shift_value: F,
) -> plonky2::plonk::circuit_data::CircuitData<F, C, D> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let gate = CosetInterpolationGate::<F, D>::new(2);
    let row = builder.num_gates();

    let shift = builder.constant(shift_value);
    builder.connect(shift, Target::wire(row, 0));

    let zero_ext = builder.zero_extension();
    for i in 0..4 {
        let start = 1 + i * D;
        builder.connect_extension(zero_ext, ExtensionTarget::from_range(row, start..start + D));
    }

    let evaluation_point = builder.constant_extension(FF::ONE);
    builder.connect_extension(evaluation_point, ExtensionTarget::from_range(row, 9..11));

    let evaluation_value = ExtensionTarget::from_range(row, 11..13);
    builder.connect_extension(evaluation_value, zero_ext);

    builder.add_gate(gate, vec![]);
    builder.build::<C>()
}

#[test]
fn interpolation_generator_zero_shift_returns_err() {
    let data = add_zero_value_coset_interpolation_circuit(F::ZERO);
    let result = catch_unwind(AssertUnwindSafe(|| data.prove(PartialWitness::new())));

    assert!(result.is_ok(), "zero coset shift must not panic");
    assert!(result.unwrap().is_err());
}

#[test]
fn interpolation_generator_nonzero_shift_still_proves() -> anyhow::Result<()> {
    let data = add_zero_value_coset_interpolation_circuit(F::ONE);
    let proof = data.prove(PartialWitness::new())?;
    data.verify(proof)
}

#[test]
fn zero_coeff_reducing_gate_rejected() {
    let base_result = catch_unwind(AssertUnwindSafe(|| ReducingGate::<D>::new(0)));
    assert!(base_result.is_err());

    let extension_result = catch_unwind(AssertUnwindSafe(|| ReducingExtensionGate::<D>::new(0)));
    assert!(extension_result.is_err());
}

#[test]
fn zero_coeff_reducing_gate_deserialization_rejected() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let builder = CircuitBuilder::<F, D>::new(config);
    let data = builder.build::<C>();

    let mut bytes = Vec::new();
    bytes.write_usize(0).unwrap();

    let mut base_buffer = Buffer::new(&bytes);
    assert!(<ReducingGate<D> as Gate<F, D>>::deserialize(&mut base_buffer, &data.common).is_err());

    let mut extension_buffer = Buffer::new(&bytes);
    assert!(<ReducingExtensionGate<D> as Gate<F, D>>::deserialize(
        &mut extension_buffer,
        &data.common
    )
    .is_err());

    Ok(())
}

fn simple_circuit_data() -> plonky2::plonk::circuit_data::CircuitData<F, C, D> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let one = builder.one();
    builder.register_public_input(one);
    builder.build::<C>()
}

#[test]
fn verifier_cap_height_mismatch_returns_err() -> anyhow::Result<()> {
    let data = simple_circuit_data();
    let proof = data.prove(PartialWitness::new())?;
    let mut verifier_data = data.verifier_data();

    assert!(verifier_data.verifier_only.constants_sigmas_cap.len() > 1);
    verifier_data
        .verifier_only
        .constants_sigmas_cap
        .0
        .truncate(1);

    let result = catch_unwind(AssertUnwindSafe(|| verifier_data.verify(proof)));
    assert!(result.is_ok(), "malformed verifier cap must not panic");
    assert!(result.unwrap().is_err());

    Ok(())
}

#[test]
fn malformed_fft_root_table_rejected() {
    let mut data = simple_circuit_data();
    data.prover_only.fft_root_table = Some(vec![vec![F::ONE]]);
    let serializer = DefaultGeneratorSerializer::<C, D>::default();

    let bytes = data
        .prover_only
        .to_bytes(&serializer, &data.common)
        .unwrap();

    assert!(
        ProverOnlyCircuitData::<F, C, D>::from_bytes(&bytes, &serializer, &data.common).is_err()
    );
}

#[test]
fn valid_fft_root_table_roundtrip_recomputes() {
    let data = simple_circuit_data();
    let serializer = DefaultGeneratorSerializer::<C, D>::default();

    let bytes = data
        .prover_only
        .to_bytes(&serializer, &data.common)
        .unwrap();
    let decoded =
        ProverOnlyCircuitData::<F, C, D>::from_bytes(&bytes, &serializer, &data.common).unwrap();

    assert!(decoded.fft_root_table.is_some());
}

#[test]
fn zero_poly_on_coset_rejects_unsupported_domains() {
    assert!(ZeroPolyOnCoset::<F>::try_new(1, F::TWO_ADICITY + 1).is_err());
    assert!(ZeroPolyOnCoset::<F>::try_new(F::TWO_ADICITY, 1).is_err());
}

#[test]
fn zero_poly_on_coset_rejects_resource_exhausting_rate() {
    assert!(ZeroPolyOnCoset::<F>::try_new(1, 21).is_err());
}

#[test]
fn zero_poly_on_coset_valid_domain_works() {
    let z_h = ZeroPolyOnCoset::<F>::try_new(2, 1).unwrap();
    let eval = z_h.eval(0);
    let eval_inverse = z_h.eval_inverse(0);
    assert_eq!(eval * eval_inverse, F::ONE);
}

#[test]
fn malformed_fri_oracle_metadata_rejected() {
    let instance = FriInstanceInfo::<F, D> {
        oracles: vec![FriOracleInfo {
            num_polys: 1,
            blinding: false,
        }],
        batches: vec![FriBatchInfo {
            point: FF::ZERO,
            openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                oracle_index: 0,
                polynomial_index: 1,
            })],
        }],
    };

    assert!(instance.check_references().is_err());
}

#[test]
fn valid_fri_oracle_metadata_references_pass() {
    let instance = FriInstanceInfo::<F, D> {
        oracles: vec![FriOracleInfo {
            num_polys: 1,
            blinding: false,
        }],
        batches: vec![FriBatchInfo {
            point: FF::ZERO,
            openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                oracle_index: 0,
                polynomial_index: 0,
            })],
        }],
    };

    assert!(instance.check_references().is_ok());
}

fn small_batch_fri_oracle() -> BatchFriOracle<F, C, D> {
    let mut timing = TimingTree::default();
    BatchFriOracle::from_values(
        vec![PolynomialValues::new(vec![
            F::ZERO,
            F::ONE,
            F::TWO,
            F::from_canonical_u64(3),
        ])],
        1,
        false,
        0,
        &mut timing,
        &[None],
    )
}

fn small_batch_fri_params() -> FriParams {
    FriParams {
        config: FriConfig {
            rate_bits: 1,
            cap_height: 0,
            proof_of_work_bits: 0,
            reduction_strategy: FriReductionStrategy::Fixed(vec![1]),
            num_query_rounds: 1,
        },
        leaf_hiding: false,
        batch_masking: None,
        degree_bits: 2,
        reduction_arity_bits: vec![1],
        final_poly_layout: FriFinalPolyLayout::Single,
    }
}

#[test]
fn malformed_batch_fri_metadata_returns_err() {
    let oracle = small_batch_fri_oracle();
    let params = small_batch_fri_params();
    let mut challenger = Challenger::<F, H>::new();
    let mut timing = TimingTree::default();

    assert!(BatchFriOracle::prove_openings(
        &[2],
        &[],
        &[&oracle],
        &mut challenger,
        &params,
        &mut timing,
    )
    .is_err());

    let missing_oracle = FriInstanceInfo {
        oracles: vec![FriOracleInfo {
            num_polys: 1,
            blinding: false,
        }],
        batches: vec![FriBatchInfo {
            point: FF::ONE,
            openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                oracle_index: 1,
                polynomial_index: 0,
            })],
        }],
    };
    assert!(BatchFriOracle::prove_openings(
        &[2],
        &[missing_oracle],
        &[&oracle],
        &mut challenger,
        &params,
        &mut timing,
    )
    .is_err());

    let missing_polynomial = FriInstanceInfo {
        oracles: vec![FriOracleInfo {
            num_polys: 1,
            blinding: false,
        }],
        batches: vec![FriBatchInfo {
            point: FF::ONE,
            openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                oracle_index: 0,
                polynomial_index: 1,
            })],
        }],
    };
    assert!(BatchFriOracle::prove_openings(
        &[2],
        &[missing_polynomial],
        &[&oracle],
        &mut challenger,
        &params,
        &mut timing,
    )
    .is_err());

    assert!(oracle.get_lde_values(0, usize::MAX, 2, 0, 1).is_err());
    assert!(oracle.get_lde_values(0, 0, 1, usize::MAX, 1).is_err());
    assert!(oracle
        .get_lde_values_packed::<F>(0, usize::MAX, 2, 0, 1)
        .is_err());
}

#[test]
fn empty_batch_fri_inputs_return_err() {
    let params = small_batch_fri_params();
    let mut challenger = Challenger::<F, H>::new();
    let mut timing = TimingTree::default();

    let empty_all = catch_unwind(AssertUnwindSafe(|| {
        batch_fri_proof::<F, C, D>(
            &[],
            PolynomialCoeffs::empty(),
            &[],
            &mut challenger,
            &params,
            &mut timing,
        )
    }));
    assert!(empty_all.is_ok(), "empty batch FRI inputs must not panic");
    assert!(empty_all.unwrap().is_err());

    let coeffs = PolynomialCoeffs::new(vec![FF::ZERO; 4]);
    let values = vec![PolynomialValues::new(vec![FF::ZERO; 4])];
    let empty_trees = catch_unwind(AssertUnwindSafe(|| {
        batch_fri_proof::<F, C, D>(&[], coeffs, &values, &mut challenger, &params, &mut timing)
    }));
    assert!(empty_trees.is_ok(), "empty committed trees must not panic");
    assert!(empty_trees.unwrap().is_err());
}

#[test]
fn polyfri_parameters_split_final_poly_not_hardcoded_single() -> anyhow::Result<()> {
    let oracle = small_batch_fri_oracle();
    let mut params = small_batch_fri_params();
    params.final_poly_layout = FriFinalPolyLayout::Split {
        chunk_degree_bits: 0,
        chunks: 2,
    };

    let mut challenger = Challenger::<F, H>::new();
    challenger.observe_cap(&oracle.batch_merkle_tree.cap);
    let zeta = challenger.get_extension_challenge::<D>();
    let opening = oracle.polynomials[0].to_extension::<D>().eval(zeta);
    challenger.observe_extension_element::<D>(&opening);
    let mut verifier_challenger = challenger.clone();
    let mut timing = TimingTree::default();

    let instance = FriInstanceInfo {
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
    let proof = BatchFriOracle::prove_openings(
        &[2],
        std::slice::from_ref(&instance),
        &[&oracle],
        &mut challenger,
        &params,
        &mut timing,
    )?;

    assert_eq!(proof.final_polys.layout, params.final_poly_layout);
    assert_eq!(proof.final_polys.chunks.len(), 2);
    assert!(proof
        .final_polys
        .chunks
        .iter()
        .all(|chunk| chunk.len() == 1));
    assert!(proof.batch_mask_proof.is_none());

    let fri_challenges = verifier_challenger.fri_challenges::<C, D>(
        &proof.commit_phase_merkle_caps,
        &proof.final_polys,
        proof.pow_witness,
        params.degree_bits,
        &params.config,
        None,
        None,
    );
    verify_batch_fri_proof::<F, C, D>(
        &[2],
        std::slice::from_ref(&instance),
        &[FriOpenings {
            batches: vec![FriOpeningBatch {
                values: vec![opening],
            }],
        }],
        &fri_challenges,
        &[oracle.batch_merkle_tree.cap.clone()],
        &proof,
        &params,
    )?;

    params.batch_masking = Some(FriBatchMaskingParams { mask_degree: 0 });
    let unsupported = BatchFriOracle::prove_openings(
        &[2],
        std::slice::from_ref(&instance),
        &[&oracle],
        &mut challenger,
        &params,
        &mut timing,
    );
    assert!(unsupported.is_err());

    Ok(())
}

fn fri_params_with_final_layout(final_poly_layout: FriFinalPolyLayout) -> FriParams {
    FriParams {
        config: CircuitConfig::standard_recursion_config().fri_config,
        leaf_hiding: false,
        batch_masking: None,
        degree_bits: 8,
        reduction_arity_bits: vec![2],
        final_poly_layout,
    }
}

#[test]
fn malformed_split_final_poly_layout_rejected() {
    assert!(fri_params_with_final_layout(FriFinalPolyLayout::Split {
        chunk_degree_bits: 4,
        chunks: 3,
    })
    .check_valid()
    .is_err());

    assert!(fri_params_with_final_layout(FriFinalPolyLayout::Split {
        chunk_degree_bits: 4,
        chunks: 0,
    })
    .check_valid()
    .is_err());

    assert!(fri_params_with_final_layout(FriFinalPolyLayout::Split {
        chunk_degree_bits: 5,
        chunks: 4,
    })
    .check_valid()
    .is_err());
}

#[test]
fn valid_split_final_poly_layout_passes() {
    assert!(fri_params_with_final_layout(FriFinalPolyLayout::Split {
        chunk_degree_bits: 4,
        chunks: 4,
    })
    .check_valid()
    .is_ok());
}

#[test]
fn mismatched_fri_degree_bits_rejected() {
    let data = simple_circuit_data();
    let mut common = data.common.clone();
    common.fri_params.degree_bits += 1;
    let serializer = DefaultGateSerializer;

    assert!(common.check_valid().is_err());
    let bytes = common.to_bytes(&serializer).unwrap();
    assert!(CommonCircuitData::<F, D>::from_bytes(bytes, &serializer).is_err());
}

#[test]
fn matched_fri_degree_bits_deserialize() {
    let data = simple_circuit_data();
    let serializer = DefaultGateSerializer;

    data.common.check_valid().unwrap();
    let bytes = data.common.to_bytes(&serializer).unwrap();
    assert!(CommonCircuitData::<F, D>::from_bytes(bytes, &serializer).is_ok());
}

#[test]
fn forged_proving_key_crashes_rejected() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_polyfri_zk_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let one = builder.one();
    builder.register_public_input(one);
    let data = builder.build::<C>();

    let mut forged_common = data.common.clone();
    forged_common.public_initial_degree_bits = forged_common.trace_degree_bits;
    forged_common.fri_params.degree_bits = forged_common.public_initial_degree_bits;
    assert!(forged_common.check_valid().is_err());

    let prove_result = catch_unwind(AssertUnwindSafe(|| {
        prove::<F, C, D>(
            &data.prover_only,
            &forged_common,
            PartialWitness::new(),
            &mut TimingTree::default(),
        )
    }));
    assert!(prove_result.is_ok(), "forged proving key must not panic");
    assert!(prove_result.unwrap().is_err());

    let mut extreme_common = data.common.clone();
    if let ZkMode::PolyFri(poly_fri) = &mut extreme_common.config.zk_config.mode {
        poly_fri.wire_mask_degree = usize::MAX;
    }
    assert!(extreme_common.check_valid().is_err());

    Ok(())
}

#[test]
fn merkle_poseidon_zero_suffix_leaf_collision_rejected() -> anyhow::Result<()> {
    let leaf = vec![
        F::from_canonical_u64(1),
        F::from_canonical_u64(2),
        F::from_canonical_u64(3),
        F::from_canonical_u64(4),
        F::from_canonical_u64(5),
    ];
    let leaves = vec![leaf.clone(), vec![F::from_canonical_u64(9); 5]];
    let tree = MerkleTree::<F, PoseidonHash>::new(leaves, 0);
    let proof = tree.prove(0);

    verify_merkle_proof_to_cap(leaf.clone(), 0, &tree.cap, &proof)?;

    let mut zero_extended_leaf = leaf;
    zero_extended_leaf.push(F::ZERO);
    assert!(verify_merkle_proof_to_cap(zero_extended_leaf, 0, &tree.cap, &proof).is_err());

    Ok(())
}

#[test]
fn merkle_leaf_proof_reopens_as_digest_shaped_alias_rejected() {
    let leaves = vec![
        vec![F::from_canonical_u64(11), F::from_canonical_u64(12)],
        vec![F::from_canonical_u64(21), F::from_canonical_u64(22)],
    ];
    let tree = MerkleTree::<F, PoseidonHash>::new(leaves.clone(), 0);
    let left = PoseidonHash::hash_merkle_leaf(&leaves[0]);
    let right = PoseidonHash::hash_merkle_leaf(&leaves[1]);
    let mut node_preimage_as_leaf = left.elements.to_vec();
    node_preimage_as_leaf.extend_from_slice(&right.elements);

    let root_cap = MerkleCap::<F, PoseidonHash>(vec![tree.cap.0[0]]);
    let empty_proof = MerkleProof::<F, PoseidonHash> { siblings: vec![] };

    assert!(verify_merkle_proof_to_cap(node_preimage_as_leaf, 0, &root_cap, &empty_proof).is_err());
}

#[test]
fn merkle_recursive_verifier_matches_native_leaf_hash() -> anyhow::Result<()> {
    let leaves = vec![
        vec![F::from_canonical_u64(1), F::from_canonical_u64(2)],
        vec![F::from_canonical_u64(3), F::from_canonical_u64(4)],
        vec![F::from_canonical_u64(5), F::from_canonical_u64(6)],
        vec![F::from_canonical_u64(7), F::from_canonical_u64(8)],
    ];
    let tree = MerkleTree::<F, PoseidonHash>::new(leaves.clone(), 1);
    let leaf_index = 2;
    let proof = tree.prove(leaf_index);
    verify_merkle_proof_to_cap(leaves[leaf_index].clone(), leaf_index, &tree.cap, &proof)?;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let mut pw = PartialWitness::new();

    let proof_t = MerkleProofTarget {
        siblings: builder.add_virtual_hashes(proof.siblings.len()),
    };
    for (&target, &sibling) in proof_t.siblings.iter().zip(&proof.siblings) {
        pw.set_hash_target(target, sibling)?;
    }

    let cap_t = builder.add_virtual_cap(1);
    pw.set_cap_target::<PoseidonHash>(&cap_t, &tree.cap)?;

    let leaf_targets = builder.add_virtual_targets(leaves[leaf_index].len());
    for (&target, &value) in leaf_targets.iter().zip(&leaves[leaf_index]) {
        pw.set_target(target, value)?;
    }

    let index = builder.constant(F::from_canonical_usize(leaf_index));
    let index_bits = builder.split_le(index, 2);
    builder.verify_merkle_proof_to_cap::<PoseidonHash>(leaf_targets, &index_bits, &cap_t, &proof_t);

    let data = builder.build::<C>();
    let proof = data.prove(pw)?;
    data.verify(proof)
}

#[test]
fn reversed_ranges_forge_oob_views_rejected() {
    let backing = vec![
        F::from_canonical_u64(1),
        F::from_canonical_u64(2),
        F::from_canonical_u64(3),
        F::from_canonical_u64(4),
    ];
    let view = PackedStridedView::<F>::new(&backing, 1, 0);

    let reversed_exclusive = catch_unwind(AssertUnwindSafe(|| view.view(view.len()..0))).is_err();
    let reversed_inclusive = catch_unwind(AssertUnwindSafe(|| view.view(view.len()..=0))).is_err();

    assert!(reversed_exclusive);
    assert!(reversed_inclusive);

    let forward = view.view(1..3);
    assert_eq!(forward.len(), 2);
    assert_eq!(*forward.get(0).unwrap(), F::from_canonical_u64(2));
    assert_eq!(*forward.get(1).unwrap(), F::from_canonical_u64(3));
}

#[test]
fn packed_strided_view_offset_overflow_rejected() {
    let scalar_backing = vec![F::ONE];
    let scalar_overflow = catch_unwind(AssertUnwindSafe(|| {
        PackedStridedView::<F>::new(&scalar_backing, 1, usize::MAX);
    }))
    .is_err();
    assert!(scalar_overflow);

    let scalar_view = PackedStridedView::<F>::new(&scalar_backing, 1, 0);
    assert_eq!(scalar_view.len(), 1);
    assert_eq!(*scalar_view.get(0).unwrap(), F::ONE);

    type P = <F as Packable>::Packing;
    let packed_backing = vec![F::ONE; P::WIDTH];
    let packed_overflow = catch_unwind(AssertUnwindSafe(|| {
        PackedStridedView::<P>::new(&packed_backing, P::WIDTH, usize::MAX);
    }))
    .is_err();
    assert!(packed_overflow);

    let packed_view = PackedStridedView::<P>::new(&packed_backing, P::WIDTH, 0);
    assert_eq!(packed_view.len(), 1);
}

#[test]
fn strided_constraint_consumer_crosses_row_boundary_rejected() {
    type P = <F as Packable>::Packing;

    if P::WIDTH > 1 {
        let stride = P::WIDTH * 2;
        let offset = stride - 1;
        let mut packed_buffer = vec![F::ZERO; stride];
        let result = catch_unwind(AssertUnwindSafe(|| {
            StridedConstraintConsumer::<P>::new(&mut packed_buffer, stride, offset);
        }));
        assert!(
            result.is_err(),
            "P::WIDTH={} stride={} offset={} must be rejected",
            P::WIDTH,
            stride,
            offset
        );
    }

    let mut scalar_buffer = vec![F::ZERO; 4];
    let mut scalar_consumer = StridedConstraintConsumer::<F>::new(&mut scalar_buffer, 2, 1);
    scalar_consumer.one(F::from_canonical_u64(7));
    scalar_consumer.one(F::from_canonical_u64(9));
    assert_eq!(scalar_buffer[1], F::from_canonical_u64(7));
    assert_eq!(scalar_buffer[3], F::from_canonical_u64(9));

    let mut packed_buffer = vec![F::ZERO; P::WIDTH * 2];
    let mut packed_consumer = StridedConstraintConsumer::<P>::new(&mut packed_buffer, P::WIDTH, 0);
    packed_consumer.one(P::from(F::from_canonical_u64(11)));
    packed_consumer.one(P::from(F::from_canonical_u64(13)));
    assert!(packed_buffer[..P::WIDTH]
        .iter()
        .all(|&x| x == F::from_canonical_u64(11)));
    assert!(packed_buffer[P::WIDTH..]
        .iter()
        .all(|&x| x == F::from_canonical_u64(13)));
}

#[test]
fn gate_id_collision_distinct_structures_rejected() {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    builder.add_gate(GateIdCollisionWeakGate, vec![]);

    let result = catch_unwind(AssertUnwindSafe(|| {
        builder.add_gate(GateIdCollisionStrongGate, vec![]);
    }));

    assert!(result.is_err());
}

#[test]
fn gate_id_collision_common_data_rejected() {
    let config = CircuitConfig::standard_recursion_config();
    let mut common_data = CircuitBuilder::<F, D>::new(config).build::<C>().common;
    common_data.gates = vec![
        GateRef::new(GateIdCollisionWeakGate),
        GateRef::new(GateIdCollisionStrongGate),
    ];

    assert!(common_data.check_valid().is_err());
}

#[test]
fn gate_id_collision_repeated_builtin_gate_still_builds() {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    builder.add_gate(NoopGate, vec![]);
    builder.add_gate(NoopGate, vec![]);

    let _data = builder.build::<C>();
}

fn minimal_fri_auxiliary_case() -> (
    FriParams,
    FriInstanceInfo<F, D>,
    FriOpenings<F, D>,
    FriChallenges<F, D>,
    Vec<MerkleCap<F, H>>,
    FriProof<F, H, D>,
) {
    let params = FriParams {
        config: FriConfig {
            rate_bits: 0,
            cap_height: 0,
            proof_of_work_bits: 0,
            reduction_strategy: FriReductionStrategy::Fixed(vec![]),
            num_query_rounds: 1,
        },
        leaf_hiding: false,
        batch_masking: None,
        degree_bits: 0,
        reduction_arity_bits: vec![],
        final_poly_layout: FriFinalPolyLayout::Single,
    };
    let instance = FriInstanceInfo {
        oracles: vec![FriOracleInfo {
            num_polys: 1,
            blinding: false,
        }],
        batches: vec![FriBatchInfo {
            point: FF::ZERO,
            openings: vec![FriOpeningExpression::raw(FriPolynomialInfo {
                oracle_index: 0,
                polynomial_index: 0,
            })],
        }],
    };
    let openings = FriOpenings {
        batches: vec![FriOpeningBatch {
            values: vec![FF::ZERO],
        }],
    };
    let challenges = FriChallenges {
        fri_alpha: FF::ONE,
        fri_betas: vec![],
        fri_pow_response: F::ZERO,
        fri_query_indices: vec![0],
    };
    let initial_merkle_caps = vec![MerkleCap::<F, H>(vec![H::hash_merkle_leaf(&[F::ZERO])])];
    let proof = FriProof {
        commit_phase_merkle_caps: vec![],
        batch_mask_proof: None,
        query_round_proofs: vec![FriQueryRound {
            initial_trees_proof: FriInitialTreeProof {
                evals_proofs: vec![(vec![F::ZERO], MerkleProof { siblings: vec![] })],
            },
            steps: vec![],
        }],
        final_polys: FriFinalPolys {
            layout: FriFinalPolyLayout::Single,
            chunks: vec![PolynomialCoeffs::new(vec![FF::ZERO])],
        },
        pow_witness: F::ZERO,
    };

    (
        params,
        instance,
        openings,
        challenges,
        initial_merkle_caps,
        proof,
    )
}

#[test]
fn empty_query_indices_skip_all_standalone_fri_checks_rejected() {
    let (params, instance, openings, mut challenges, initial_merkle_caps, proof) =
        minimal_fri_auxiliary_case();
    challenges.fri_query_indices.clear();

    assert!(validate_fri_auxiliary_shape::<F, C, D>(
        &instance,
        &openings,
        &challenges,
        &initial_merkle_caps,
        &proof,
        &params,
    )
    .is_err());
}

#[test]
fn empty_query_indices_skip_all_batch_fri_checks_rejected() {
    let (params, instance, openings, mut challenges, initial_merkle_caps, proof) =
        minimal_fri_auxiliary_case();
    challenges.fri_query_indices.clear();

    assert!(validate_batch_fri_auxiliary_shape::<F, C, D>(
        &[params.degree_bits],
        &[instance],
        &[openings],
        &challenges,
        &initial_merkle_caps,
        &proof,
        &params,
    )
    .is_err());
}

#[test]
fn short_fri_initial_merkle_caps_rejected() {
    let (params, instance, openings, challenges, _initial_merkle_caps, proof) =
        minimal_fri_auxiliary_case();

    assert!(validate_fri_auxiliary_shape::<F, C, D>(
        &instance,
        &openings,
        &challenges,
        &[],
        &proof,
        &params,
    )
    .is_err());
}

#[test]
fn short_fri_opening_batches_rejected() {
    let (params, instance, mut openings, challenges, initial_merkle_caps, proof) =
        minimal_fri_auxiliary_case();
    openings.batches.clear();

    assert!(validate_fri_auxiliary_shape::<F, C, D>(
        &instance,
        &openings,
        &challenges,
        &initial_merkle_caps,
        &proof,
        &params,
    )
    .is_err());
}

#[test]
fn valid_fri_auxiliary_shapes_pass() {
    let (params, instance, openings, challenges, initial_merkle_caps, proof) =
        minimal_fri_auxiliary_case();

    validate_fri_auxiliary_shape::<F, C, D>(
        &instance,
        &openings,
        &challenges,
        &initial_merkle_caps,
        &proof,
        &params,
    )
    .unwrap();
    validate_batch_fri_auxiliary_shape::<F, C, D>(
        &[params.degree_bits],
        &[instance],
        &[openings],
        &challenges,
        &initial_merkle_caps,
        &proof,
        &params,
    )
    .unwrap();
}

fn poseidon2_two_leaf_tree() -> (Vec<Vec<F>>, MerkleTree<F, Poseidon2Hash>) {
    let leaves = vec![
        vec![F::from_canonical_u64(31), F::from_canonical_u64(32)],
        vec![F::from_canonical_u64(41), F::from_canonical_u64(42)],
    ];
    let tree = MerkleTree::<F, Poseidon2Hash>::new(leaves.clone(), 0);
    (leaves, tree)
}

#[test]
fn poseidon2_merkle_accepts_internal_digest_rejected() {
    let (_leaves, tree) = poseidon2_two_leaf_tree();
    let digest_as_leaf = tree.cap.0[0].elements.to_vec();
    let root_cap = MerkleCap::<F, Poseidon2Hash>(vec![tree.cap.0[0]]);
    let empty_proof = MerkleProof::<F, Poseidon2Hash> { siblings: vec![] };

    assert!(verify_merkle_proof_to_cap(digest_as_leaf, 0, &root_cap, &empty_proof).is_err());
}

#[test]
fn poseidon2_merkle_accepts_internal_node_preimage_rejected() {
    let (leaves, tree) = poseidon2_two_leaf_tree();
    let left = Poseidon2Hash::hash_merkle_leaf(&leaves[0]);
    let right = Poseidon2Hash::hash_merkle_leaf(&leaves[1]);
    let mut node_preimage_as_leaf = left.elements.to_vec();
    node_preimage_as_leaf.extend_from_slice(&right.elements);
    let root_cap = MerkleCap::<F, Poseidon2Hash>(vec![tree.cap.0[0]]);
    let empty_proof = MerkleProof::<F, Poseidon2Hash> { siblings: vec![] };

    assert!(verify_merkle_proof_to_cap(node_preimage_as_leaf, 0, &root_cap, &empty_proof).is_err());
}

#[test]
fn poseidon2_merkle_valid_proof_verifies() -> anyhow::Result<()> {
    let (leaves, tree) = poseidon2_two_leaf_tree();
    let proof = tree.prove(1);

    verify_merkle_proof_to_cap(leaves[1].clone(), 1, &tree.cap, &proof)
}

#[test]
fn poseidon2_merkle_recursive_matches_native() -> anyhow::Result<()> {
    let (leaves, tree) = poseidon2_two_leaf_tree();
    let leaf_index = 1;
    let proof = tree.prove(leaf_index);
    verify_merkle_proof_to_cap(leaves[leaf_index].clone(), leaf_index, &tree.cap, &proof)?;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let mut pw = PartialWitness::new();

    let proof_t = MerkleProofTarget {
        siblings: builder.add_virtual_hashes(proof.siblings.len()),
    };
    for (&target, &sibling) in proof_t.siblings.iter().zip(&proof.siblings) {
        pw.set_hash_target(target, sibling)?;
    }

    let cap_t = builder.add_virtual_cap(0);
    pw.set_cap_target::<Poseidon2Hash>(&cap_t, &tree.cap)?;

    let leaf_targets = builder.add_virtual_targets(leaves[leaf_index].len());
    for (&target, &value) in leaf_targets.iter().zip(&leaves[leaf_index]) {
        pw.set_target(target, value)?;
    }

    let index = builder.constant(F::from_canonical_usize(leaf_index));
    let index_bits = builder.split_le(index, 1);
    builder.verify_merkle_proof_to_cap::<Poseidon2Hash>(
        leaf_targets,
        &index_bits,
        &cap_t,
        &proof_t,
    );

    let data = builder.build::<C>();
    let proof = data.prove(pw)?;
    data.verify(proof)
}
