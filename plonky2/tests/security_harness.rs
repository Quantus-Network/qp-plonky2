use std::panic::{catch_unwind, AssertUnwindSafe};

use plonky2::field::packable::Packable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::{
    structure::{
        FriBatchInfo, FriInstanceInfo, FriOpeningExpression, FriOracleInfo, FriPolynomialInfo,
    },
    FriFinalPolyLayout, FriParams,
};
use plonky2::gates::coset_interpolation::CosetInterpolationGate;
use plonky2::gates::exponentiation::ExponentiationGate;
use plonky2::gates::gate::Gate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData, ProverOnlyCircuitData};
use plonky2::util::strided_view::PackedStridedView;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::serialization::{
    Buffer, DefaultGateSerializer, DefaultGeneratorSerializer, Write,
};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type FF = <C as GenericConfig<D>>::FE;

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
