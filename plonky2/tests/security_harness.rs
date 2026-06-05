use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::FriReductionStrategy;
use plonky2::gadgets::lookup::TIP5_TABLE;
use plonky2::gates::coset_interpolation::CosetInterpolationGate;
use plonky2::gates::exponentiation::ExponentiationGate;
use plonky2::gates::gate::Gate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, ProverOnlyCircuitData};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::serialization::{Buffer, DefaultGeneratorSerializer, Write};

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

fn sample_common_data() -> plonky2::plonk::circuit_data::CommonCircuitData<F, D> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let x = builder.add_virtual_public_input();
    let y = builder.mul(x, x);
    builder.register_public_input(y);
    builder.build::<C>().common
}

fn sample_lookup_common_data() -> plonky2::plonk::circuit_data::CommonCircuitData<F, D> {
    use plonky2::gates::lookup_table::LookupTable;

    let table: LookupTable = Arc::new((0u16..256).zip(TIP5_TABLE).collect());
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let index = builder.add_lookup_table_from_pairs(table);
    let output = builder.add_lookup_from_index(input, index);
    builder.register_public_input(input);
    builder.register_public_input(output);
    builder.build::<C>().common
}

fn sample_common_data_with_reductions() -> plonky2::plonk::circuit_data::CommonCircuitData<F, D> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let mut x = builder.add_virtual_public_input();
    for _ in 0..5_000 {
        x = builder.mul(x, x);
    }
    builder.register_public_input(x);
    let common = builder.build::<C>().common;
    assert!(!common.fri_params.reduction_arity_bits.is_empty());
    common
}

fn tamper_is_rejected(
    mut common: plonky2::plonk::circuit_data::CommonCircuitData<F, D>,
    mutate: impl FnOnce(&mut plonky2::plonk::circuit_data::CommonCircuitData<F, D>),
) -> bool {
    use plonky2::plonk::circuit_data::CommonCircuitData;
    use plonky2::util::serialization::DefaultGateSerializer;

    let gate_serializer = DefaultGateSerializer;
    mutate(&mut common);
    let bytes = common.to_bytes(&gate_serializer).unwrap();
    CommonCircuitData::<F, D>::from_bytes(bytes, &gate_serializer).is_err()
}

fn tampered_common_data_is_rejected(
    mutate: impl FnOnce(&mut plonky2::plonk::circuit_data::CommonCircuitData<F, D>),
) -> bool {
    tamper_is_rejected(sample_common_data(), mutate)
}

fn common_data_round_trips(common: plonky2::plonk::circuit_data::CommonCircuitData<F, D>) -> bool {
    use plonky2::plonk::circuit_data::CommonCircuitData;
    use plonky2::util::serialization::DefaultGateSerializer;

    let gate_serializer = DefaultGateSerializer;
    let bytes = common.to_bytes(&gate_serializer).unwrap();
    CommonCircuitData::<F, D>::from_bytes(bytes, &gate_serializer).is_ok()
}

#[test]
fn genuine_common_data_round_trips() {
    assert!(common_data_round_trips(sample_common_data()));
}

#[test]
fn genuine_lookup_common_data_round_trips() {
    assert!(common_data_round_trips(sample_lookup_common_data()));
}

#[test]
fn deserialization_rejects_zero_query_rounds() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.config.fri_config.num_query_rounds = 0;
    }));
}

#[test]
fn deserialization_rejects_excessive_reduction_arities() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.fri_params.reduction_arity_bits = vec![c.fri_params.degree_bits + 1];
    }));
}

#[test]
fn deserialization_rejects_inconsistent_partial_products() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.num_partial_products += 1;
    }));
}

#[test]
fn deserialization_rejects_zero_quotient_degree_factor() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.quotient_degree_factor = 0;
    }));
}

#[test]
fn deserialization_rejects_selector_index_length_mismatch() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.selectors_info.selector_indices.push(0);
    }));
}

#[test]
fn deserialization_rejects_wrong_k_is_length() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.k_is.push(F::ZERO);
    }));
}

#[test]
fn deserialization_rejects_routed_wires_exceeding_wires() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.config.num_routed_wires = c.config.num_wires + 1;
    }));
}

#[test]
fn deserialization_rejects_phantom_lookup_polys() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.num_lookup_polys = 1;
    }));
}

#[test]
fn deserialization_rejects_mismatched_fri_query_rounds() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.config.fri_config.num_query_rounds -= 1;
    }));
}

#[test]
fn deserialization_rejects_mismatched_fri_rate_bits() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.config.fri_config.rate_bits += 1;
    }));
}

#[test]
fn deserialization_rejects_mismatched_public_initial_degree_bits() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.public_initial_degree_bits += 1;
    }));
}

#[test]
fn genuine_common_data_with_reductions_round_trips() {
    assert!(common_data_round_trips(sample_common_data_with_reductions()));
}

#[test]
fn deserialization_rejects_cleared_reduction_arities() {
    assert!(tamper_is_rejected(
        sample_common_data_with_reductions(),
        |c| {
            c.fri_params.reduction_arity_bits.clear();
        }
    ));
}

#[test]
fn deserialization_rejects_altered_reduction_arity() {
    assert!(tamper_is_rejected(
        sample_common_data_with_reductions(),
        |c| {
            c.fri_params.reduction_arity_bits[0] = 1;
        }
    ));
}

#[test]
fn deserialization_rejects_flipped_leaf_hiding() {
    assert!(tampered_common_data_is_rejected(|c| {
        c.fri_params.leaf_hiding = !c.fri_params.leaf_hiding;
    }));
}

#[test]
fn deserialization_rejects_zero_constant_arity_strategy() {
    assert!(tampered_common_data_is_rejected(|c| {
        let strategy = FriReductionStrategy::ConstantArityBits(0, 0);
        c.config.fri_config.reduction_strategy = strategy.clone();
        c.fri_params.config.reduction_strategy = strategy;
    }));
}

#[test]
fn deserialization_rejects_zero_fixed_arity_strategy() {
    assert!(tampered_common_data_is_rejected(|c| {
        let strategy = FriReductionStrategy::Fixed(vec![0]);
        c.config.fri_config.reduction_strategy = strategy.clone();
        c.fri_params.config.reduction_strategy = strategy;
        c.fri_params.reduction_arity_bits = vec![0];
    }));
}

#[test]
fn deserialization_rejects_oversized_min_size_query_count() {
    assert!(tampered_common_data_is_rejected(|c| {
        let strategy = FriReductionStrategy::MinSize(None);
        c.config.fri_config.reduction_strategy = strategy.clone();
        c.config.fri_config.num_query_rounds = usize::MAX;
        c.fri_params.config.reduction_strategy = strategy;
        c.fri_params.config.num_query_rounds = usize::MAX;
    }));
}

#[test]
fn deserialization_rejects_selector_group_not_containing_gate() {
    assert!(tamper_is_rejected(sample_lookup_common_data(), |c| {
        let groups = &c.selectors_info.groups;
        assert!(
            groups.len() >= 2,
            "test requires a circuit with multiple selector groups"
        );
        // Reassign gate 0 to a different existing group that does not contain it.
        let original = c.selectors_info.selector_indices[0];
        let other = (0..groups.len())
            .find(|&g| g != original && !groups[g].contains(&0))
            .expect("expected a non-containing selector group");
        c.selectors_info.selector_indices[0] = other;
    }));
}

#[test]
fn deserialization_rejects_disabled_lookup_polys() {
    assert!(tamper_is_rejected(sample_lookup_common_data(), |c| {
        c.num_lookup_polys = 0;
    }));
}

#[test]
fn deserialization_rejects_degenerate_lookup_polys() {
    assert!(tamper_is_rejected(sample_lookup_common_data(), |c| {
        c.num_lookup_polys = 1;
    }));
}

#[test]
fn deserialization_rejects_wrong_lookup_selector_count() {
    assert!(tamper_is_rejected(sample_lookup_common_data(), |c| {
        c.num_lookup_selectors += 1;
    }));
}
