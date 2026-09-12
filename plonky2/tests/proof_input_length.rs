use plonky2::field::types::Field;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::util::serialization::DefaultGateSerializer;
use plonky2_verifier::{CommonCircuitData, C, D, F};

#[test]
fn both_decoders_check_public_input_lengths() {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    for _ in 0..8 {
        let input = builder.constant(F::ONE);
        builder.register_public_input(input);
    }
    let data = builder.build::<C>();
    let proof = data.prove(PartialWitness::new()).unwrap();
    let common = CommonCircuitData::from_bytes(
        data.common.to_bytes(&DefaultGateSerializer).unwrap(),
        &plonky2_verifier::util::serialization::DefaultGateSerializer,
    )
    .unwrap();
    let bytes = proof.to_bytes();
    let count_offset = bytes.len() - (proof.public_inputs.len() + 1) * 8;
    assert_eq!(
        ProofWithPublicInputs::<F, C, D>::from_bytes(bytes.clone(), &data.common)
            .unwrap()
            .to_bytes(),
        bytes
    );
    assert_eq!(
        plonky2_verifier::ProofWithPublicInputs::<F, C, D>::from_bytes(bytes.clone(), &common)
            .unwrap()
            .to_bytes(),
        bytes
    );

    for count in [0u64, 7, 9, (1 << 32) + 8, u64::MAX] {
        let mut malformed = bytes[..count_offset].to_vec();
        malformed.extend_from_slice(&count.to_le_bytes());
        malformed.resize(count_offset + 8 + count.min(9) as usize * 8, 0);
        assert!(
            ProofWithPublicInputs::<F, C, D>::from_bytes(malformed.clone(), &data.common).is_err()
        );
        assert!(
            plonky2_verifier::ProofWithPublicInputs::<F, C, D>::from_bytes(malformed, &common)
                .is_err()
        );
    }
    assert!(ProofWithPublicInputs::<F, C, D>::from_bytes(
        bytes[..bytes.len() - 1].to_vec(),
        &data.common
    )
    .is_err());
    assert!(
        plonky2_verifier::ProofWithPublicInputs::<F, C, D>::from_bytes(
            bytes[..bytes.len() - 1].to_vec(),
            &common
        )
        .is_err()
    );
}

#[test]
fn length_validation_rejects_allocation_amplification() {
    use qp_plonky2_core::proof::checked_public_input_len;
    assert_eq!(
        checked_public_input_len(8u64.to_le_bytes(), 8, 64).unwrap(),
        8
    );
    assert!(checked_public_input_len((4 * 1024 * 1024u64).to_le_bytes(), 8, 64).is_err());
    assert!(checked_public_input_len(8u64.to_le_bytes(), 8, 63).is_err());
    assert!(
        checked_public_input_len((usize::MAX as u64).to_le_bytes(), usize::MAX, usize::MAX)
            .is_err()
    );
}
