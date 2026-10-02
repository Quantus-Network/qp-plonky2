#![cfg(feature = "constraint-export")]

use plonky2::constraint_export::export_gate_constraints;
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::gates::noop::NoopGate;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::PoseidonGoldilocksConfig;

#[test]
fn external_backend_can_read_opening_description_and_constraint_program() {
    let mut builder =
        CircuitBuilder::<GoldilocksField, 2>::new(CircuitConfig::standard_recursion_config());
    builder.add_gate(NoopGate, vec![]);
    let data = builder.build::<PoseidonGoldilocksConfig>();
    let instance = data.common.get_fri_instance(QuadraticExtension::ONE);
    assert_eq!(instance.oracles.len(), 4);
    assert_eq!(instance.batches.len(), 2);
    let program = export_gate_constraints(&data.common).unwrap();
    assert_eq!(program.gates.len(), data.common.gates.len());
}
