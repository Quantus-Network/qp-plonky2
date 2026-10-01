use plonky2::constraint_export::{
    export_gate_constraints, ArithmeticNode, ConstraintExportError, ConstraintInput,
    GateConstraintProgram,
};
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::types::Field;
use plonky2::fri::structure::{FriCoefficient, FriPolynomialInfo};
use plonky2::gates::arithmetic_base::ArithmeticGate;
use plonky2::gates::arithmetic_extension::ArithmeticExtensionGate;
use plonky2::gates::base_sum::BaseSumGate;
use plonky2::gates::constant::ConstantGate;
use plonky2::gates::coset_interpolation::CosetInterpolationGate;
use plonky2::gates::exponentiation::ExponentiationGate;
use plonky2::gates::gate::Gate;
use plonky2::gates::multiplication_extension::MulExtensionGate;
use plonky2::gates::noop::NoopGate;
use plonky2::gates::poseidon::PoseidonGate;
use plonky2::gates::poseidon2::Poseidon2Gate;
use plonky2::gates::poseidon2_int_mix::Poseidon2IntMixGate;
use plonky2::gates::poseidon2_mds::Poseidon2MdsGate;
use plonky2::gates::poseidon_mds::PoseidonMdsGate;
use plonky2::gates::public_input::PublicInputGate;
use plonky2::gates::random_access::RandomAccessGate;
use plonky2::gates::reducing::ReducingGate;
use plonky2::gates::reducing_extension::ReducingExtensionGate;
use plonky2::hash::hash_types::HashOut;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::WitnessGeneratorRef;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::vanishing_poly::evaluate_gate_constraints_base_batch;
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBaseBatch};
use plonky2::util::serialization::{Buffer, IoResult};

use crate as plonky2;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Config4;

impl GenericConfig<4> for Config4 {
    type F = F;
    type FE = plonky2::field::extension::quartic::QuarticExtension<F>;
    type Hasher = <C as GenericConfig<D>>::Hasher;
    type InnerHasher = <C as GenericConfig<D>>::InnerHasher;
}

fn common_with_default_gates() -> CommonCircuitData<F, D> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    // All non-lookup gate families in DefaultGateSerializer. Building this
    // small circuit requires neither private proofs nor a proving run.
    builder.add_gate(ArithmeticGate::new_from_config(&config), vec![]);
    builder.add_gate(
        ArithmeticExtensionGate::<D>::new_from_config(&config),
        vec![],
    );
    builder.add_gate(BaseSumGate::<2>::new(16), vec![]);
    builder.add_gate(ConstantGate::new(2), vec![]);
    builder.add_gate(CosetInterpolationGate::<F, D>::new(2), vec![]);
    builder.add_gate(ExponentiationGate::<F, D>::new(8), vec![]);
    builder.add_gate(MulExtensionGate::<D>::new_from_config(&config), vec![]);
    builder.add_gate(NoopGate, vec![]);
    builder.add_gate(PoseidonMdsGate::<F, D>::new(), vec![]);
    builder.add_gate(PoseidonGate::<F, D>::new(), vec![]);
    builder.add_gate(Poseidon2MdsGate::<F, D>::new(), vec![]);
    builder.add_gate(Poseidon2IntMixGate::<F, D>::new(), vec![]);
    builder.add_gate(Poseidon2Gate::<F, D>::new(), vec![]);
    builder.add_gate(PublicInputGate, vec![]);
    builder.add_gate(
        RandomAccessGate::<F, D>::new_from_config(&config, 3),
        vec![],
    );
    builder.add_gate(ReducingExtensionGate::<D>::new(8), vec![]);
    builder.add_gate(ReducingGate::<D>::new(8), vec![]);
    builder.build::<C>().common
}

fn interpret(
    program: &GateConstraintProgram<F>,
    constants: &[F],
    wires: &[F],
    hash: &HashOut<F>,
) -> Vec<Vec<F>> {
    program
        .gates
        .iter()
        .map(|gate| {
            assert_eq!(gate.outputs.len(), program.num_constraints);
            let mut values: Vec<F> = Vec::with_capacity(gate.nodes.len());
            for (i, node) in gate.nodes.iter().enumerate() {
                let value = match *node {
                    ArithmeticNode::Constant(value) => value,
                    ArithmeticNode::Input(kind) => match kind {
                        ConstraintInput::Constant(i) => constants[i],
                        ConstraintInput::Wire(i) => wires[i],
                        ConstraintInput::PublicInputHash(i) => hash.elements[i],
                    },
                    ArithmeticNode::Add(a, b) => {
                        assert!(a < i && b < i, "graph must be topologically ordered");
                        values[a] + values[b]
                    }
                    ArithmeticNode::Mul(a, b) => {
                        assert!(a < i && b < i, "graph must be topologically ordered");
                        values[a] * values[b]
                    }
                };
                values.push(value);
            }
            gate.outputs.iter().map(|&i| values[i]).collect()
        })
        .collect()
}

#[test]
fn exported_default_gate_constraints_match_cpu_on_arbitrary_inputs() {
    let common = common_with_default_gates();
    let program = export_gate_constraints(&common).unwrap();
    assert_eq!(program.gates.len(), common.gates.len());
    assert!(program.gates.len() >= 17);
    assert!(common.selectors_info.num_selectors() > 1);
    for (exported, gate) in program.gates.iter().zip(&common.gates) {
        assert_eq!(exported.gate_id, gate.0.id());
    }
    // Deliberately not valid witness rows: nonzero residuals expose missing
    // operations, wrong selectors, and incorrect constraint output ordering.
    for sample in 0..8 {
        let mut constants: Vec<_> = (0..common.num_constants)
            .map(|i| -F::from_canonical_usize(31 * i + 17 * sample + 5))
            .collect();
        let wires: Vec<_> = (0..common.config.num_wires)
            .map(|i| -F::from_canonical_usize(71 * i + 23 * sample + 3))
            .collect();
        let hash = HashOut {
            elements: core::array::from_fn(|i| F::from_canonical_usize(i + sample + 9)),
        };
        // Exercise active, unused, and off-domain selector values.
        for (i, group) in common.selectors_info.groups.iter().enumerate() {
            constants[i] = match sample % 3 {
                0 => F::from_canonical_usize(group.start),
                1 => F::from_canonical_u64(plonky2::gates::selectors::UNUSED_SELECTOR as u64),
                _ => F::from_canonical_usize(common.gates.len() + i + 2),
            };
        }
        let vars = EvaluationVarsBaseBatch::new(1, &constants, &wires, &hash);
        let exported = interpret(&program, &constants, &wires, &hash);
        for (i, gate) in common.gates.iter().enumerate() {
            let selector = common.selectors_info.selector_indices[i];
            let mut expected = gate.0.eval_filtered_base_batch(
                vars,
                i,
                selector,
                common.selectors_info.groups[selector].clone(),
                common.selectors_info.num_selectors(),
                0,
            );
            expected.resize(common.num_gate_constraints, F::ZERO);
            assert_eq!(
                exported[i],
                expected,
                "gate {} sample {sample}",
                gate.0.id()
            );
        }
        let mut combined = vec![F::ZERO; common.num_gate_constraints];
        for outputs in exported {
            for (sum, output) in combined.iter_mut().zip(outputs) {
                *sum += output;
            }
        }
        assert_eq!(
            combined,
            evaluate_gate_constraints_base_batch(&common, vars)
        );
        if sample % 3 != 1 {
            assert!(combined.iter().any(|&v| v != F::ZERO));
        }
    }
}

#[test]
fn canonical_fri_description_is_available_to_external_backends() {
    let common = common_with_default_gates();
    let zeta = QuadraticExtension([F::from_canonical_u64(19), F::from_canonical_u64(7)]);
    let instance = common.get_fri_instance(zeta);
    let widths = [
        common.num_constants + common.config.num_routed_wires,
        common.config.num_wires,
        common.config.num_challenges * (1 + common.num_partial_products),
        common.config.num_challenges * common.quotient_degree_factor,
    ];
    assert_eq!(instance.oracles.len(), widths.len());
    for (oracle, width) in instance.oracles.iter().zip(widths) {
        assert_eq!(oracle.num_polys, width);
    }
    assert_eq!(instance.batches.len(), 2);
    assert_eq!(instance.batches[0].point, zeta);
    assert_eq!(
        instance.batches[1].point,
        zeta * QuadraticExtension::<F>::primitive_root_of_unity(common.degree_bits())
    );
    let all: Vec<_> = widths
        .iter()
        .enumerate()
        .flat_map(|(oracle_index, &width)| {
            (0..width).map(move |polynomial_index| FriPolynomialInfo {
                oracle_index,
                polynomial_index,
            })
        })
        .collect();
    for (batch, expected) in instance.batches.iter().zip([
        all,
        (0..common.config.num_challenges)
            .map(|polynomial_index| FriPolynomialInfo {
                oracle_index: 2,
                polynomial_index,
            })
            .collect(),
    ]) {
        assert_eq!(batch.openings.len(), expected.len());
        for (expression, polynomial) in batch.openings.iter().zip(expected) {
            assert_eq!(expression.terms.len(), 1);
            assert!(matches!(
                expression.terms[0].coefficient,
                FriCoefficient::One
            ));
            assert_eq!(
                expression.terms[0].polynomial.oracle_index,
                polynomial.oracle_index
            );
            assert_eq!(
                expression.terms[0].polynomial.polynomial_index,
                polynomial.polynomial_index
            );
        }
    }
}

#[test]
fn unsupported_configurations_return_errors() {
    let mut common = common_with_default_gates();
    common.num_lookup_polys = 1;
    assert_eq!(
        export_gate_constraints(&common).unwrap_err(),
        ConstraintExportError::UnsupportedLookups
    );
    common.num_lookup_polys = 0;
    common.selectors_info.selector_indices.clear();
    assert_eq!(
        export_gate_constraints(&common).unwrap_err(),
        ConstraintExportError::InvalidSelectorMetadata
    );
    let mut builder = CircuitBuilder::<F, 4>::new(CircuitConfig::standard_recursion_config());
    builder.add_gate(NoopGate, vec![]);
    let common = builder.build::<Config4>().common;
    assert_eq!(
        export_gate_constraints(&common).unwrap_err(),
        ConstraintExportError::UnsupportedExtensionDegree(4)
    );
}

#[derive(Debug, Clone, Copy)]
enum UnsupportedGate {
    Copy,
    UnknownTarget,
    NonBase,
    NewGate,
    ConfiguredGate,
}

impl Gate<F, D> for UnsupportedGate {
    fn id(&self) -> String {
        format!("UnsupportedGate::{self:?}")
    }
    fn serialize(&self, _: &mut Vec<u8>, _: &CommonCircuitData<F, D>) -> IoResult<()> {
        Ok(())
    }
    fn deserialize(_: &mut Buffer, _: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::Copy)
    }
    fn eval_unfiltered(&self, _: EvaluationVars<F, D>) -> Vec<QuadraticExtension<F>> {
        vec![QuadraticExtension::ZERO]
    }
    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let zero = builder.zero_extension();
        match self {
            Self::Copy => {
                builder.connect(vars.local_wires[0].0[0], vars.local_wires[1].0[0]);
            }
            Self::UnknownTarget => return vec![builder.add_virtual_extension_target()],
            Self::NonBase => {
                let value = ExtensionTarget([zero.0[0], vars.local_wires[0].0[0]]);
                return vec![builder.add_extension(value, zero)];
            }
            Self::NewGate => {
                builder.add_gate(NoopGate, vec![]);
            }
            Self::ConfiguredGate => {
                // These constructors used to underflow with the exporter's
                // zero-routed-wire configuration. Reject the emitted gate,
                // not the otherwise valid configuration or its constructors.
                let gate = BaseSumGate::<2>::new_from_config::<F>(&builder.config);
                let _ = ExponentiationGate::<F, D>::new_from_config(&builder.config);
                builder.config.validate();
                builder.add_gate(gate, vec![]);
            }
        }
        vec![zero]
    }
    fn generators(&self, _: usize, _: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![]
    }
    fn num_wires(&self) -> usize {
        2
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
fn unsupported_gate_operations_return_errors_instead_of_partial_programs() {
    for (gate, reason) in [
        (UnsupportedGate::Copy, "untraced copy constraint"),
        (UnsupportedGate::UnknownTarget, "untraced target"),
        (UnsupportedGate::NonBase, "non-base expression"),
        (UnsupportedGate::NewGate, "untraced gate instance"),
        (UnsupportedGate::ConfiguredGate, "untraced gate instance"),
    ] {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        builder.add_gate(gate, vec![]);
        let common = builder.build::<C>().common;
        assert_eq!(
            export_gate_constraints(&common).unwrap_err(),
            ConstraintExportError::UnsupportedGate {
                gate_id: gate.id(),
                reason
            }
        );
    }
}

#[test]
fn mds_gate_selection_preserves_configuration_and_normal_policy() {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    assert_eq!(
        builder.use_mds_gate(),
        config.num_routed_wires >= PoseidonMdsGate::<F, D>::new().num_wires()
    );
    builder.constraint_expression = Some(super::Expression::default());
    assert!(!builder.use_mds_gate());
    assert_eq!(builder.config, config);
    builder.constraint_expression = None;
    builder.config.num_routed_wires = 1;
    assert!(!builder.use_mds_gate());
}
