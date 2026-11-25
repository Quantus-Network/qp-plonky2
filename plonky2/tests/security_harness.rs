use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use plonky2::field::types::Field;
use plonky2::field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::fri::FriReductionStrategy;
use plonky2::gates::base_sum::BaseSumGate;
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
use plonky2::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
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

/// `range_check(x, 0)` used to add no constraint at all; it must pin `x` to zero.
#[test]
fn zero_width_range_check_binds_value_to_zero() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let x = builder.add_virtual_target();
    builder.range_check(x, 0);
    let data = builder.build::<C>();

    let mut honest = PartialWitness::new();
    honest.set_target(x, F::ZERO)?;
    data.verify(data.prove(honest)?)?;

    let mut forged = PartialWitness::new();
    forged.set_target(x, F::ONE)?;
    let result = catch_unwind(AssertUnwindSafe(|| data.prove(forged)));
    assert!(result.map_or(true, |proof| proof.is_err()));
    Ok(())
}

/// A 64-bit exponent target does not determine the power (`e` and `e + ORDER` both fit), so
/// `exp` must refuse the width at build time.
#[test]
fn field_width_exp_rejected() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let base = builder.constant(F::TWO);
        let exponent = builder.add_virtual_target();
        let _ = builder.exp(base, exponent, F::BITS);
    }));
    assert!(result.is_err());
}

#[test]
fn below_field_width_exp_still_proves() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let base = builder.constant(F::TWO);
    let exponent = builder.add_virtual_target();
    let power = builder.exp(base, exponent, F::BITS - 1);
    let expected = builder.constant(F::from_canonical_u64(1 << 20));
    builder.connect(power, expected);
    let data = builder.build::<C>();

    let mut pw = PartialWitness::new();
    pw.set_target(exponent, F::from_canonical_u64(20))?;
    data.verify(data.prove(pw)?)
}

/// Indexing a one-element vector skipped the gate entirely, leaving the index unconstrained.
#[test]
fn singleton_random_access_binds_index_to_zero() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let index = builder.add_virtual_target();
    let element = builder.constant(F::from_canonical_u64(7));
    let selected = builder.random_access(index, vec![element]);
    builder.connect(selected, element);
    let data = builder.build::<C>();

    let mut honest = PartialWitness::new();
    honest.set_target(index, F::ZERO)?;
    data.verify(data.prove(honest)?)?;

    let mut forged = PartialWitness::new();
    forged.set_target(index, F::ONE)?;
    let result = catch_unwind(AssertUnwindSafe(|| data.prove(forged)));
    assert!(result.map_or(true, |proof| proof.is_err()));
    Ok(())
}

/// A `BaseSumGate` wider than the field admits two limb vectors per sum, so circuit data
/// declaring one must be rejected on deserialization.
#[test]
fn overwide_base_sum_gate_deserialization_rejected() {
    let data = simple_circuit_data();
    let max_limbs = BaseSumGate::<2>::max_limbs::<F>();

    let mut buffer = Vec::new();
    buffer.write_usize(max_limbs + 1).unwrap();
    assert!(
        <BaseSumGate<2> as Gate<F, D>>::deserialize(&mut Buffer::new(&buffer), &data.common)
            .is_err()
    );

    let mut buffer = Vec::new();
    buffer.write_usize(max_limbs).unwrap();
    assert!(
        <BaseSumGate<2> as Gate<F, D>>::deserialize(&mut Buffer::new(&buffer), &data.common)
            .is_ok()
    );
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
fn fft_root_table_with_one_wrong_root_rejected() {
    let mut data = simple_circuit_data();
    let table = data.prover_only.fft_root_table.as_mut().unwrap();
    table[1][1] += F::ONE;
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
fn valid_fft_root_table_roundtrips() {
    let data = simple_circuit_data();
    let serializer = DefaultGeneratorSerializer::<C, D>::default();

    let bytes = data
        .prover_only
        .to_bytes(&serializer, &data.common)
        .unwrap();
    let decoded =
        ProverOnlyCircuitData::<F, C, D>::from_bytes(&bytes, &serializer, &data.common).unwrap();

    assert_eq!(decoded.fft_root_table, data.prover_only.fft_root_table);
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

/// A FRI rate of 1 or zero query rounds leaves FRI with nothing to test; the builder must refuse
/// such a config even when `security_bits` is set low enough to pass the security-level check.
#[test]
fn degenerate_fri_config_rejected_at_build() {
    for (rate_bits, num_query_rounds) in [(0, 28), (3, 0)] {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut config = CircuitConfig::standard_recursion_config();
            config.security_bits = 0;
            config.fri_config.rate_bits = rate_bits;
            config.fri_config.num_query_rounds = num_query_rounds;
            CircuitBuilder::<F, D>::new(config)
        }));
        assert!(result.is_err());
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// Blinding shifts and multiplies by the `Fixed` arities before any trace degree is known, so a
/// schedule the field cannot hold (`[32, 32]` wraps the product to zero; `[62]` wraps the opening
/// count) is rejected when the builder is created, as is a rate with no room for a trace.
#[test]
fn fixed_fri_schedule_beyond_the_field_rejected_at_new() {
    let rejected = |mutate: &dyn Fn(&mut CircuitConfig), message: &str| {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut config = CircuitConfig::standard_recursion_zk_config();
            mutate(&mut config);
            CircuitBuilder::<F, D>::new(config)
        }));
        assert_eq!(panic_message(result.unwrap_err()), message);
    };
    // The standard rate of 3 leaves 29 bits of trace in Goldilocks.
    for arity_bits in [
        vec![30],
        vec![40],
        vec![32, 32],
        vec![62],
        vec![usize::MAX, 1],
    ] {
        rejected(
            &|config| {
                config.fri_config.reduction_strategy =
                    FriReductionStrategy::Fixed(arity_bits.clone())
            },
            "FRI total reduction arity exceeds the field's largest domain",
        );
    }
    rejected(
        &|config| config.fri_config.rate_bits = 40,
        "FRI rate_bits exceed the field's two-adicity",
    );
}

/// A `Fixed` schedule that fits the field but that no trace can carry has blinding double its
/// degree estimate past the field's largest domain; the estimate stops there instead.
#[test]
fn impossible_fixed_fri_schedule_rejected_before_blinding() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut config = CircuitConfig::standard_recursion_zk_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![29]);
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        builder.register_public_input(x);
        builder.build::<C>()
    }));
    assert_eq!(
        panic_message(result.unwrap_err()),
        "the FRI schedule needs more blinding rows than the field's largest domain holds"
    );
}

/// Blinding sizes the trace from estimates below the final degree, so a `Fixed` schedule is only
/// checked against the degree the circuit ends up with. With `cap_height < rate_bits` this
/// schedule fits the final degree but not the first estimate.
#[test]
fn fixed_fri_schedule_that_fits_only_after_blinding_builds() -> anyhow::Result<()> {
    let mut config = CircuitConfig::standard_recursion_zk_config();
    config.fri_config.rate_bits = 8;
    config.fri_config.cap_height = 0;
    config.fri_config.num_query_rounds = 12;
    config.fri_config.proof_of_work_bits = 20;
    config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![4, 4]);
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let x = builder.add_virtual_target();
    let y = builder.mul(x, x);
    builder.register_public_input(y);
    let data = builder.build::<C>();

    let mut witness = PartialWitness::new();
    witness.set_target(x, F::from_canonical_u64(3))?;
    let proof = data.prove(witness)?;
    data.verify(proof)
}

/// A config too narrow for a gate to hold a single operation must fail loudly when that gate
/// is first needed, not misbehave in slot bookkeeping.
#[test]
fn zero_slot_gate_rejected() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let config = CircuitConfig {
            num_routed_wires: 3,
            max_quotient_degree_factor: 2,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        builder.add(x, x)
    }));
    let message = panic_message(result.unwrap_err());
    assert!(message.contains("no operation slots"), "{message}");
}

/// Only the config, degree, gates, public input count and lookup tables are serialized; the
/// loader derives the rest the way the builder does, after checking those primitives are ones the
/// builder could have fixed. Both crates reject the same bytes.
#[test]
fn common_data_primitives_are_checked_on_load() {
    let data = simple_circuit_data();
    let rejected = |mutate: fn(&mut CommonCircuitData<F, D>), error: &str| {
        let mut common = data.common.clone();
        mutate(&mut common);
        assert_eq!(
            CommonCircuitData::<F, D>::new(
                common.config.clone(),
                common.trace_degree_bits,
                common.gates.clone(),
                common.num_public_inputs,
                common.luts.clone(),
            )
            .err(),
            Some(error)
        );

        let gate_serializer = DefaultGateSerializer;
        let bytes = common.to_bytes(&gate_serializer).unwrap();
        assert!(CommonCircuitData::<F, D>::from_bytes(bytes.clone(), &gate_serializer).is_err());
        assert!(
            plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
                bytes,
                &plonky2_verifier::util::serialization::DefaultGateSerializer,
            )
            .is_err()
        );
    };

    // Consistent data that weakens FRI below `security_bits` fails the rule the builder applies.
    rejected(
        |common| {
            common.config.fri_config.num_query_rounds = 1;
            common.config.fri_config.proof_of_work_bits = 0;
        },
        "FRI params fall short of target security",
    );
    rejected(
        |common| common.trace_degree_bits = 40,
        "FRI LDE domain exceeds the field's two-adicity",
    );
    // The schedule search would wrap on this rate; the domain is bounded before it runs.
    rejected(
        |common| {
            common.config.fri_config.rate_bits = usize::MAX;
            common.config.fri_config.reduction_strategy = FriReductionStrategy::MinSize(None);
        },
        "FRI LDE domain exceeds the field's two-adicity",
    );
    rejected(
        |common| common.gates.swap(0, 1),
        "gates must be sorted by degree and id without duplicates",
    );
    rejected(
        |common| common.config.num_wires = common.config.num_routed_wires,
        "gate does not fit within the circuit config",
    );
    rejected(
        |common| common.config.num_wires = CircuitConfig::MAX_WIDTH + 1,
        "num_wires and num_constants must not exceed 2^16",
    );
}

/// Gate parameters are checked against the config before any gate method can overflow,
/// underflow or divide by zero on them and before the common data is derived from those methods.
/// The same bounds apply to bytes read by either crate and to common data built in memory.
#[test]
fn hostile_gate_parameters_rejected() {
    use plonky2::gates::arithmetic_base::ArithmeticGate;
    use plonky2::gates::base_sum::BaseSumGate;
    use plonky2::gates::gate::GateRef;
    use plonky2::gates::random_access::RandomAccessGate;
    use plonky2::util::serialization::GateSerializer;
    use plonky2_verifier::util::serialization::{
        Buffer as VerifierBuffer, DefaultGateSerializer as VerifierGateSerializer,
        GateSerializer as _,
    };

    let common = simple_circuit_data().common;
    let serializer = DefaultGateSerializer;
    let verifier_serializer = VerifierGateSerializer;
    let verifier_common =
        plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
            common.to_bytes(&serializer).unwrap(),
            &verifier_serializer,
        )
        .unwrap();

    let accepted = |gate: GateRef<F, D>| {
        let mut bytes = Vec::new();
        serializer.write_gate(&mut bytes, &gate, &common).unwrap();
        let prover_accepts = serializer
            .read_gate(&mut Buffer::new(&bytes), &common)
            .is_ok();
        let verifier_accepts = verifier_serializer
            .read_gate(&mut VerifierBuffer::new(&bytes), &verifier_common)
            .is_ok();
        assert_eq!(prover_accepts, verifier_accepts, "{gate:?}");

        let mut gates = common.gates.clone();
        gates.push(gate.clone());
        let in_memory_accepts = CommonCircuitData::<F, D>::new(
            common.config.clone(),
            common.trace_degree_bits,
            gates,
            common.num_public_inputs,
            common.luts.clone(),
        ) != Err("gate parameters do not fit within the circuit config");
        assert_eq!(prover_accepts, in_memory_accepts, "{gate:?}");
        prover_accepts
    };

    assert!(accepted(GateRef::new(ArithmeticGate::new_from_config(
        &common.config
    ))));
    // `num_wires()` would wrap to zero and slip past the width check.
    assert!(!accepted(GateRef::new(ArithmeticGate { num_ops: 1 << 62 })));

    assert!(accepted(GateRef::new(CosetInterpolationGate::<F, D>::new(
        4
    ))));
    let mut divides_by_zero = CosetInterpolationGate::<F, D>::new(4);
    divides_by_zero.degree = 1;
    assert!(!accepted(GateRef::new(divides_by_zero)));

    assert!(accepted(GateRef::new(ExponentiationGate::<F, D>::new(8))));
    assert!(!accepted(GateRef::new(ExponentiationGate::<F, D>::new(0))));
    assert!(!accepted(GateRef::new(ExponentiationGate::<F, D>::new(
        common.config.num_wires
    ))));

    // 2^64 exceeds the field order, so 64 base-2 limbs cannot decompose uniquely.
    assert!(accepted(GateRef::new(BaseSumGate::<2>::new(63))));
    assert!(!accepted(GateRef::new(BaseSumGate::<2>::new(64))));
    assert!(!accepted(GateRef::new(BaseSumGate::<2>::new(usize::MAX))));

    let widest = RandomAccessGate::<F, D>::new_from_config(&common.config, 4);
    assert!(accepted(GateRef::new(widest)));
    let mut too_many_constants = widest;
    too_many_constants.num_extra_constants += 1;
    assert!(!accepted(GateRef::new(too_many_constants)));

    assert!(accepted(GateRef::new(ReducingGate::<D>::new(4))));
    assert!(!accepted(GateRef::new(ReducingGate::<D>::new(
        common.config.num_wires
    ))));
}

/// Primitives may carry a config no honest gate fits; the config is rejected before the fit
/// checks take logarithms of its widths.
#[test]
fn zero_width_config_rejected_before_gate_checks() {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    builder.add_gate(CosetInterpolationGate::<F, D>::new(2), vec![]);
    let index = builder.add_virtual_target();
    let values = builder.add_virtual_targets(4);
    builder.random_access(index, values);
    let common = builder.build::<C>().common;
    let serializer = DefaultGateSerializer;
    let verifier_common =
        plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
            common.to_bytes(&serializer).unwrap(),
            &plonky2_verifier::util::serialization::DefaultGateSerializer,
        )
        .unwrap();

    for zero_width in [
        |config: &mut CircuitConfig| config.num_wires = 0,
        |config: &mut CircuitConfig| config.num_routed_wires = 0,
    ] {
        let mut config = common.config.clone();
        zero_width(&mut config);
        assert!(CommonCircuitData::<F, D>::new(
            config.clone(),
            common.trace_degree_bits,
            common.gates.clone(),
            common.num_public_inputs,
            common.luts.clone(),
        )
        .is_err());
        assert!(
            plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::new(
                config,
                verifier_common.trace_degree_bits,
                verifier_common.gates.clone(),
                verifier_common.num_public_inputs,
                verifier_common.luts.clone(),
            )
            .is_err()
        );
    }
}

/// Proofs built in memory (or via serde) skip `from_bytes`, so the verifier's own shape checks
/// must return errors rather than panic: on an odd cap, and on Merkle proofs that stop short of
/// the cap or whose layer heights never match the leaf data.
#[test]
fn malformed_in_memory_proof_rejected_not_panicking() -> anyhow::Result<()> {
    use plonky2::hash::merkle_proofs::{
        verify_batch_merkle_proof_to_cap, verify_merkle_proof_to_cap, MerkleProof,
    };
    use plonky2::hash::merkle_tree::MerkleTree;

    type H = <C as GenericConfig<D>>::Hasher;

    let data = simple_circuit_data();
    let mut odd_cap = data.prove(PartialWitness::new())?;
    odd_cap.proof.wires_cap.0.pop();
    assert!(data.verify(odd_cap).is_err());

    let leaves: Vec<Vec<F>> = (0..8u64).map(|i| vec![F::from_canonical_u64(i)]).collect();
    let tree = MerkleTree::<F, H>::new(leaves.clone(), 1);
    // With no siblings, leaf 7 would index the two-entry cap at 7.
    let no_siblings = MerkleProof::<F, H> { siblings: vec![] };
    assert!(verify_merkle_proof_to_cap(leaves[7].clone(), 7, &tree.cap, &no_siblings).is_err());

    let merkle_proof = tree.prove(3);
    assert!(
        verify_batch_merkle_proof_to_cap::<F, H>(&[], &[], 3, &tree.cap, &merkle_proof).is_err()
    );
    // No layer has height 9, so the second leaf is never absorbed.
    assert!(verify_batch_merkle_proof_to_cap::<F, H>(
        &[leaves[3].clone(), leaves[0].clone()],
        &[3, 9],
        3,
        &tree.cap,
        &merkle_proof
    )
    .is_err());
    Ok(())
}

/// Generators are read from prover-only bytes, which this harness treats as untrusted: a lookup
/// table index past `luts`, a slot count the gate cannot hold, or more limbs than a wire row has
/// are errors rather than panics.
#[test]
fn hostile_generator_parameters_rejected_on_deserialization() {
    use plonky2::util::serialization::WitnessGeneratorSerializer;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let table_index = builder.add_lookup_table_from_pairs(Arc::new(vec![(0, 1), (1, 2)]));
    let input = builder.add_virtual_target();
    builder.register_public_input(input);
    builder.add_lookup_from_index(input, table_index);
    builder.split_le(input, 8);
    let data = builder.build::<C>();
    let serializer = DefaultGeneratorSerializer::<C, D>::default();

    let generator_bytes = |id: &str| {
        let generator = data
            .prover_only
            .generators
            .iter()
            .find(|generator| generator.0.id() == id)
            .unwrap();
        let mut bytes = Vec::new();
        serializer
            .write_generator(&mut bytes, generator, &data.common)
            .unwrap();
        bytes
    };
    let accepted = |bytes: &[u8]| {
        serializer
            .read_generator(&mut Buffer::new(bytes), &data.common)
            .is_ok()
    };
    let word = 8;

    // The lookup table index is each lookup generator's last field.
    for id in ["LookupGenerator", "LookupTableGenerator"] {
        let mut bytes = generator_bytes(id);
        assert!(accepted(&bytes), "{id}");
        let lut_index = bytes.len() - word;
        bytes[lut_index..].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(!accepted(&bytes), "{id}");
    }

    // `num_slots` is the third-last field; the generator divides by it.
    let mut bytes = generator_bytes("LookupTableGenerator");
    let num_slots = bytes.len() - 3 * word;
    bytes[num_slots..num_slots + word].copy_from_slice(&0u64.to_le_bytes());
    assert!(!accepted(&bytes));

    let mut bytes = generator_bytes("BaseSplitGenerator + Base: 2");
    assert!(accepted(&bytes));
    let num_limbs = bytes.len() - word;
    bytes[num_limbs..].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(!accepted(&bytes));
}

/// Derived fields are not in the encoding, so altering one in memory changes nothing on the wire:
/// the bytes decode to what the builder produced. Gates are read against the circuit's own
/// config, so circuits wider than the default config round-trip too; that half only bites once
/// gate readers check their parameters against the config (#145).
#[test]
fn common_data_roundtrips_from_its_primitives() {
    let gate_serializer = DefaultGateSerializer;
    for (num_wires, num_routed_wires) in [(143, 120), (234, 200)] {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig {
            num_wires,
            num_routed_wires,
            ..CircuitConfig::standard_recursion_config()
        });
        let x = builder.add_virtual_target();
        let y = builder.mul(x, x);
        builder.register_public_input(y);
        let common = builder.build::<C>().common;
        let bytes = common.to_bytes(&gate_serializer).unwrap();
        assert_eq!(
            CommonCircuitData::<F, D>::from_bytes(bytes.clone(), &gate_serializer).unwrap(),
            common
        );
        assert!(
            plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
                bytes,
                &plonky2_verifier::util::serialization::DefaultGateSerializer,
            )
            .is_ok()
        );
    }

    let data = simple_circuit_data();
    for mutate in [
        |_: &mut CommonCircuitData<F, D>| {},
        |common: &mut CommonCircuitData<F, D>| common.num_constants += 1,
        |common: &mut CommonCircuitData<F, D>| common.fri_params.config.num_query_rounds = 1,
        |common: &mut CommonCircuitData<F, D>| common.k_is[1] = common.k_is[0],
        |common: &mut CommonCircuitData<F, D>| common.selectors_info.groups.truncate(1),
    ] {
        let mut common = data.common.clone();
        mutate(&mut common);
        let bytes = common.to_bytes(&gate_serializer).unwrap();
        assert_eq!(
            CommonCircuitData::<F, D>::from_bytes(bytes.clone(), &gate_serializer).unwrap(),
            data.common
        );
        assert!(
            plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
                bytes,
                &plonky2_verifier::util::serialization::DefaultGateSerializer,
            )
            .is_ok()
        );
    }
}

/// #64703: the in-circuit `hash_leaf` is length-binding, so a zero-suffixed leaf cannot be coerced
/// to share an honest (shorter) leaf's digest. Constraining the forged leaf's hash to equal the
/// honest leaf's hash must therefore be unsatisfiable (proof rejected).
#[test]
fn forged_zero_suffixed_leaf_rejected_in_circuit() -> anyhow::Result<()> {
    type H = <C as GenericConfig<D>>::Hasher;

    let honest: Vec<F> = (1..=5).map(F::from_canonical_u64).collect();
    let honest_hash = H::hash_leaf(&honest);

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    // The forged leaf is the honest leaf with a trailing zero (length + 1).
    let forged: Vec<Target> = (0..honest.len() + 1)
        .map(|_| builder.add_virtual_target())
        .collect();
    let hash = builder.hash_leaf::<H>(forged.clone());
    let expected = builder.constant_hash(honest_hash);
    builder.connect_hashes(hash, expected);

    let data = builder.build::<C>();
    let mut pw = PartialWitness::new();
    for (i, t) in forged.iter().enumerate() {
        pw.set_target(*t, honest.get(i).copied().unwrap_or(F::ZERO))?;
    }

    // A panic in witness generation, a `prove` error, or a failing `verify` all mean the forgery
    // was rejected; only a verifying proof would indicate a surviving collision.
    let rejected = catch_unwind(AssertUnwindSafe(|| match data.prove(pw) {
        Ok(proof) => data.verify(proof).is_err(),
        Err(_) => true,
    }))
    .unwrap_or(true);
    assert!(
        rejected,
        "forged zero-suffixed leaf must not satisfy the honest leaf's hash in-circuit"
    );

    Ok(())
}

#[test]
fn honest_leaf_hash_matches_in_circuit() -> anyhow::Result<()> {
    type H = <C as GenericConfig<D>>::Hasher;

    let honest: Vec<F> = (1..=5).map(F::from_canonical_u64).collect();
    let honest_hash = H::hash_leaf(&honest);

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let targets: Vec<Target> = (0..honest.len())
        .map(|_| builder.add_virtual_target())
        .collect();
    let hash = builder.hash_leaf::<H>(targets.clone());
    let expected = builder.constant_hash(honest_hash);
    builder.connect_hashes(hash, expected);

    let data = builder.build::<C>();
    let mut pw = PartialWitness::new();
    for (t, v) in targets.iter().zip(&honest) {
        pw.set_target(*t, *v)?;
    }

    let proof = data.prove(pw)?;
    data.verify(proof)
}

/// Proof bytes must have exactly one accepted encoding: trailing data and non-canonical field
/// elements are rejected by `from_bytes` in both crates rather than silently normalised.
#[test]
fn proof_bytes_are_canonical() -> anyhow::Result<()> {
    use plonky2::field::types::{Field64, PrimeField64};
    use plonky2::plonk::proof::ProofWithPublicInputs;
    use plonky2_verifier::plonk::proof::ProofWithPublicInputs as VerifierProof;

    let data = simple_circuit_data();
    let proof = data.prove(PartialWitness::new())?;
    let bytes = proof.to_bytes();
    let common = &data.common;

    let mut trailing = bytes.clone();
    trailing.push(0);

    // The single public input (the constant one) is the last field element written.
    let mut non_canonical = bytes.clone();
    let tail = non_canonical.len() - 8;
    assert_eq!(u64::from_le_bytes(bytes[tail..].try_into()?), 1);
    non_canonical[tail..].copy_from_slice(&(F::ORDER + 1).to_le_bytes());
    assert_eq!(F::from_noncanonical_u64(F::ORDER + 1).to_canonical_u64(), 1);

    // Hash limbs are field elements too; the first limb of `wires_cap` opens the proof. Poseidon
    // hashes `p + x` as `x`, so an unchecked limb would be a second encoding of the same proof.
    let mut non_canonical_cap = bytes.clone();
    non_canonical_cap[..8].copy_from_slice(&u64::MAX.to_le_bytes());

    let plonky2_rejects =
        |bytes: Vec<u8>| ProofWithPublicInputs::<F, C, D>::from_bytes(bytes, common).is_err();
    assert!(!plonky2_rejects(bytes.clone()));
    assert!(plonky2_rejects(trailing.clone()));
    assert!(plonky2_rejects(non_canonical.clone()));
    assert!(plonky2_rejects(non_canonical_cap.clone()));

    let verifier_common =
        plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
            common.to_bytes(&DefaultGateSerializer).unwrap(),
            &plonky2_verifier::util::serialization::DefaultGateSerializer,
        )
        .unwrap();
    let verifier_rejects =
        |bytes: Vec<u8>| VerifierProof::<F, C, D>::from_bytes(bytes, &verifier_common).is_err();
    assert!(!verifier_rejects(bytes));
    assert!(verifier_rejects(trailing));
    assert!(verifier_rejects(non_canonical));
    assert!(verifier_rejects(non_canonical_cap));
    Ok(())
}

#[test]
fn circuit_data_bytes_with_trailing_data_rejected() -> anyhow::Result<()> {
    use plonky2::plonk::circuit_data::VerifierOnlyCircuitData;

    let data = simple_circuit_data();
    let gate_serializer = DefaultGateSerializer;

    let mut common_bytes = data.common.to_bytes(&gate_serializer).unwrap();
    common_bytes.push(0);
    assert!(CommonCircuitData::<F, D>::from_bytes(common_bytes, &gate_serializer).is_err());

    let mut verifier_only_bytes = data.verifier_only.to_bytes().unwrap();
    verifier_only_bytes.push(0);
    assert!(VerifierOnlyCircuitData::<C, D>::from_bytes(verifier_only_bytes.clone()).is_err());

    let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();
    let mut prover_only_bytes = data
        .prover_only
        .to_bytes(&generator_serializer, &data.common)
        .unwrap();
    prover_only_bytes.push(0);
    assert!(ProverOnlyCircuitData::<F, C, D>::from_bytes(
        &prover_only_bytes,
        &generator_serializer,
        &data.common
    )
    .is_err());

    let mut circuit_data_bytes = data
        .to_bytes(&gate_serializer, &generator_serializer)
        .unwrap();
    circuit_data_bytes.push(0);
    assert!(
        plonky2::plonk::circuit_data::CircuitData::<F, C, D>::from_bytes(
            &circuit_data_bytes,
            &gate_serializer,
            &generator_serializer
        )
        .is_err()
    );

    let mut verifier_common_bytes = data.common.to_bytes(&gate_serializer).unwrap();
    verifier_common_bytes.push(0);
    assert!(
        plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
            verifier_common_bytes,
            &plonky2_verifier::util::serialization::DefaultGateSerializer,
        )
        .is_err()
    );
    assert!(
        plonky2_verifier::plonk::circuit_data::VerifierOnlyCircuitData::<C, D>::from_bytes(
            verifier_only_bytes
        )
        .is_err()
    );
    Ok(())
}

/// Prover data has one encoding per value too: the watch map must be in key order without
/// repeats, and a serialized FFT root table must be the one the prover would recompute. The same
/// bytes sit inside `CircuitData` and `ProverCircuitData`, which must reject them the same way.
#[test]
fn prover_only_bytes_are_canonical() {
    use plonky2::field::types::PrimeField64;
    use plonky2::plonk::circuit_data::{CircuitData, ProverCircuitData};
    use plonky2::util::serialization::Write;

    let gate_serializer = DefaultGateSerializer;
    let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();
    let data = simple_circuit_data();
    let prover_only_bytes = |data: &CircuitData<F, C, D>| {
        data.prover_only
            .to_bytes(&generator_serializer, &data.common)
            .unwrap()
    };
    let bytes = prover_only_bytes(&data);
    let decode = |bytes: &[u8]| {
        ProverOnlyCircuitData::<F, C, D>::from_bytes(bytes, &generator_serializer, &data.common)
    };
    assert_eq!(decode(&bytes).unwrap(), data.prover_only);

    // Serializing with a field cleared locates that field: the first differing byte.
    let offset_of = |cleared: &[u8]| bytes.iter().zip(cleared).position(|(a, b)| a != b).unwrap();

    let mut without_watches = simple_circuit_data();
    without_watches
        .prover_only
        .generator_indices_by_watches
        .clear();
    let map_len_offset = offset_of(&prover_only_bytes(&without_watches));
    let watches = &data.prover_only.generator_indices_by_watches;
    let (first_key, first_watchers) = watches.first_key_value().unwrap();
    let mut repeated_entry = bytes[..map_len_offset].to_vec();
    repeated_entry.write_usize(watches.len() + 1).unwrap();
    repeated_entry.write_usize(*first_key).unwrap();
    repeated_entry.write_usize_vec(first_watchers).unwrap();
    repeated_entry.extend_from_slice(&bytes[map_len_offset + 8..]);
    assert!(decode(&repeated_entry).is_err());

    let mut without_cache = simple_circuit_data();
    without_cache.prover_only.fft_root_table = None;
    let without_cache_bytes = prover_only_bytes(&without_cache);
    // `None` is a different value, not a second spelling of the table.
    assert_eq!(
        decode(&without_cache_bytes).unwrap(),
        without_cache.prover_only
    );
    // Skip the presence flag, the row count and the first row's length to reach its first limb.
    let first_limb = offset_of(&without_cache_bytes) + 1 + 8 + 8;
    let limb = F::from_canonical_u64(u64::from_le_bytes(
        bytes[first_limb..][..8].try_into().unwrap(),
    ));
    let mut altered_limb = bytes.clone();
    altered_limb[first_limb..][..8]
        .copy_from_slice(&(limb + F::ONE).to_canonical_u64().to_le_bytes());
    assert!(decode(&altered_limb).is_err());

    // The other prover-holding entry points carry the same bytes verbatim.
    let full_bytes = data
        .to_bytes(&gate_serializer, &generator_serializer)
        .unwrap();
    let prover_full_bytes = simple_circuit_data()
        .prover_data()
        .to_bytes(&gate_serializer, &generator_serializer)
        .unwrap();
    let decodes_full = |bytes: &[u8]| {
        CircuitData::<F, C, D>::from_bytes(bytes, &gate_serializer, &generator_serializer).is_ok()
    };
    let decodes_prover = |bytes: &[u8]| {
        ProverCircuitData::<F, C, D>::from_bytes(bytes, &gate_serializer, &generator_serializer)
            .is_ok()
    };
    let containers: [(&[u8], &dyn Fn(&[u8]) -> bool); 2] = [
        (&full_bytes, &decodes_full),
        (&prover_full_bytes, &decodes_prover),
    ];
    for (container, decodes) in containers {
        let at = container
            .windows(bytes.len())
            .position(|window| window == bytes)
            .unwrap();
        let spliced =
            |altered: &[u8]| [&container[..at], altered, &container[at + bytes.len()..]].concat();
        assert!(decodes(&spliced(&bytes)));
        assert!(!decodes(&spliced(&repeated_entry)));
        assert!(!decodes(&spliced(&altered_limb)));
    }
}
