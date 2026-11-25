use std::panic::{catch_unwind, AssertUnwindSafe};

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

/// A `Fixed` schedule the domain cannot hold is rejected when the schedule is first derived, before
/// blinding tries to pad the trace to a size that matches it.
#[test]
fn impossible_fixed_fri_schedule_rejected_before_blinding() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut config = CircuitConfig::standard_recursion_zk_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![40]);
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        builder.register_public_input(x);
        builder.build::<C>()
    }));
    let message = result.unwrap_err();
    let message = message
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert_eq!(message, "FRI reductions exceed the degree");
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
    let message = result.unwrap_err();
    let message = message
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| message.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(message.contains("no operation slots"), "{message}");
}

/// Every derived field of `CommonCircuitData` is recomputed from the config, degree, gates and
/// lookup tables and compared, so a tampered copy is reported whether it arrives through
/// `from_bytes` or is built in memory and handed straight to `verify`.
#[test]
fn tampered_common_data_is_reported_by_field() -> anyhow::Result<()> {
    let data = simple_circuit_data();
    let proof = data.prove(PartialWitness::new())?;
    data.verify(proof.clone())?;

    let tampered = |mutate: fn(&mut CommonCircuitData<F, D>), error: &str| -> anyhow::Result<()> {
        let mut verifier_data = data.verifier_data();
        mutate(&mut verifier_data.common);
        assert_eq!(verifier_data.common.check_valid(), Err(error));
        assert!(verifier_data.verify(proof.clone()).is_err());

        let gate_serializer = DefaultGateSerializer;
        let bytes = verifier_data.common.to_bytes(&gate_serializer).unwrap();
        assert!(CommonCircuitData::<F, D>::from_bytes(bytes.clone(), &gate_serializer).is_err());
        assert!(
            plonky2_verifier::plonk::circuit_data::CommonCircuitData::<F, D>::from_bytes(
                bytes,
                &plonky2_verifier::util::serialization::DefaultGateSerializer,
            )
            .is_err()
        );
        Ok(())
    };

    // #64700: FRI query indices are sampled from `public_initial_degree_bits`, so a smaller value
    // than the committed `fri_params.degree_bits` would draw queries from a smaller domain.
    tampered(
        |common| common.public_initial_degree_bits -= 1,
        "trace_degree_bits, public_initial_degree_bits and FRI degree_bits must agree",
    )?;
    tampered(
        |common| common.trace_degree_bits -= 1,
        "trace_degree_bits, public_initial_degree_bits and FRI degree_bits must agree",
    )?;
    // The verifier takes its query count and reduction schedule from `fri_params`, which is
    // serialized separately from `config.fri_config`.
    tampered(
        |common| common.fri_params.config.num_query_rounds = 1,
        "fri_params must be derived from config",
    )?;
    // Consistent data that weakens FRI below `security_bits` fails the rule the builder applies.
    tampered(
        |common| {
            common.config.fri_config.num_query_rounds = 1;
            common.config.fri_config.proof_of_work_bits = 0;
            common.fri_params.config = common.config.fri_config.clone();
        },
        "FRI params fall short of target security",
    )?;
    tampered(
        |common| common.fri_params.reduction_arity_bits.push(1),
        "fri_params must be derived from config",
    )?;
    tampered(
        |common| common.quotient_degree_factor -= 1,
        "quotient_degree_factor must equal config.max_quotient_degree_factor",
    )?;
    tampered(
        |common| common.gates.swap(0, 1),
        "gates must be sorted by degree and id without duplicates",
    )?;
    tampered(
        |common| common.selectors_info.groups.truncate(1),
        "selectors_info must be derived from the gates",
    )?;
    tampered(
        |common| common.num_constants += 1,
        "num_constants must be derived from the selectors and gates",
    )?;
    tampered(
        |common| common.num_gate_constraints += 1,
        "num_gate_constraints must be the maximum over the gates",
    )?;
    tampered(
        |common| common.num_partial_products += 1,
        "num_partial_products must be derived from the config",
    )?;
    tampered(
        |common| common.k_is[1] = common.k_is[0],
        "k_is must be the unique coset shifts for the degree",
    )?;
    tampered(
        |common| common.num_lookup_polys += 1,
        "num_lookup_polys must be derived from the config and lookup tables",
    )?;
    Ok(())
}

/// The recursive verifier builds its constraints from `inner_common_data`, so it applies the same
/// consistency check as the native verifier rather than enforcing whatever it is handed.
#[test]
fn recursive_verifier_rejects_tampered_inner_common_data() {
    let inner = simple_circuit_data();
    let mut tampered = inner.common.clone();
    tampered.fri_params.config.num_query_rounds = 1;

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let proof = builder.add_virtual_proof_with_pis(&tampered);
        let verifier_data =
            builder.add_virtual_verifier_data(tampered.config.fri_config.cap_height);
        builder.verify_proof::<C>(&proof, &verifier_data, &tampered);
    }));
    let message = result.unwrap_err();
    let message = message
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        message.contains("fri_params must be derived from config"),
        "{message}"
    );
}

#[test]
fn untampered_common_data_roundtrips() {
    let data = simple_circuit_data();
    let gate_serializer = DefaultGateSerializer;
    let bytes = data.common.to_bytes(&gate_serializer).unwrap();

    assert!(CommonCircuitData::<F, D>::from_bytes(bytes, &gate_serializer).is_ok());
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
