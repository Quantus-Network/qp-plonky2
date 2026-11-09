//! Circuit access discovery for witness-generation planning.

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::{GeneratedValues, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::witness::PartitionWitness;
use crate::iop::witness_generation_plan::{WitnessGenerationPlan, WitnessGenerationPlanning};
use crate::iop::witness_generation_schedule::GeneratorDependencies;
use crate::plonk::circuit_data::{CircuitData, CommonCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, Read};

/// Collects declared reads and discovers output targets of built-in generators.
/// Discovery recognizes generator IDs. Each generator must be the corresponding
/// built-in implementation, with input-independent output targets and emission order.
/// Debug execution checks emitted targets against discovery.
#[derive(Debug)]
pub struct WitnessGenerationPlanner<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    circuit: &'a CircuitData<F, C, D>,
    dependencies: GeneratorDependencies,
    #[cfg(debug_assertions)]
    expected_output_targets: Arc<Vec<Vec<Target>>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGenerationPlanner<'a, F, C, D>
{
    /// Collects generator read and output targets for the circuit.
    pub fn new(circuit: &'a CircuitData<F, C, D>) -> Result<Self> {
        let common = &circuit.common;
        let prover = &circuit.prover_only;
        let generators = &prover.generators;
        let resolve = |target: Target| -> Result<usize> {
            let target_index = target.index(common.config.num_wires, common.degree());
            let rep_index = *prover
                .representative_map
                .get(target_index)
                .ok_or_else(|| anyhow!("target {target:?} is outside the witness"))?;
            ensure!(
                rep_index < prover.representative_map.len(),
                "invalid representative index {rep_index}"
            );
            ensure!(
                prover.representative_map[rep_index] == rep_index,
                "representative index {rep_index} is not a root"
            );
            Ok(rep_index)
        };
        let mut probe = PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &prover.representative_map,
        );
        let mut reads = Vec::with_capacity(generators.len());
        let mut writes = Vec::with_capacity(generators.len());
        #[cfg(debug_assertions)]
        let mut expected_output_targets = Vec::with_capacity(generators.len());
        for (generator_index, generator) in generators.iter().enumerate() {
            let targets = generator.0.reads().ok_or_else(|| {
                anyhow!(
                    "generator {generator_index} ({}) does not declare its reads",
                    generator.0.id()
                )
            })?;
            reads.push(
                targets
                    .into_iter()
                    .map(&resolve)
                    .collect::<Result<Vec<_>>>()?,
            );
            let discovered =
                discover_outputs(generator.0.as_ref(), common, &mut probe).map_err(|error| {
                    anyhow!(
                        "generator {generator_index} ({}): {error}",
                        generator.0.id()
                    )
                })?;
            writes.push(
                discovered
                    .iter()
                    .copied()
                    .map(&resolve)
                    .collect::<Result<Vec<_>>>()?,
            );
            #[cfg(debug_assertions)]
            expected_output_targets.push(discovered);
        }
        let dependencies =
            GeneratorDependencies::new(reads, writes, prover.representative_map.len())?;
        Ok(Self {
            circuit,
            #[cfg(debug_assertions)]
            expected_output_targets: Arc::new(expected_output_targets),
            dependencies,
        })
    }

    /// Starts building a witness generation plan.
    pub fn new_plan(&self) -> WitnessGenerationPlanning<'_, 'a, F, C, D> {
        WitnessGenerationPlanning::new(self)
    }

    /// Builds a complete single-stage plan from explicit caller input targets.
    pub(super) fn input_plan(
        &self,
        input_targets: impl IntoIterator<Item = Target>,
    ) -> Result<Arc<WitnessGenerationPlan<'a, F, C, D>>> {
        let mut planning = self.new_plan();
        for target in input_targets {
            planning.assume_assigned(target)?;
        }
        planning.add_stage();
        planning.finish().map_err(|_| {
            anyhow!(
                "cannot create a complete witness generation plan from the supplied input targets"
            )
        })
    }

    pub(super) fn circuit(&self) -> &'a CircuitData<F, C, D> {
        self.circuit
    }

    pub(super) fn dependencies(&self) -> &GeneratorDependencies {
        &self.dependencies
    }

    #[cfg(debug_assertions)]
    pub(super) fn expected_output_targets(&self) -> &Arc<Vec<Vec<Target>>> {
        &self.expected_output_targets
    }
}

/// Collects output targets in emission order by running a built-in with sample inputs.
/// Duplicate emissions are preserved.
///
/// The supplied witness must be dense, initially unassigned, and use matching
/// circuit data. Declared read targets are assigned for the call and cleared
/// on success or a returned error. The generator must be the built-in implementation
/// corresponding to its ID, with an input-independent output target sequence.
fn discover_outputs<F: RichField + Extendable<D>, const D: usize>(
    generator: &dyn WitnessGenerator<F, D>,
    common: &CommonCircuitData<F, D>,
    probe_witness: &mut PartitionWitness<'_, F>,
) -> Result<Vec<Target>> {
    let id = generator.id();
    let input_rule = dummy_input_rule(&id)?;
    let read_targets = generator
        .reads()
        .ok_or_else(|| anyhow!("generator {id} does not declare its reads"))?;
    let override_value = match input_rule {
        DummyInputRule::Zero => None,
        DummyInputRule::Quotient => {
            ensure!(
                D > 0 && D.checked_mul(2) == Some(read_targets.len()),
                "invalid read declaration for generator {id}"
            );
            Some((read_targets[D], F::ONE))
        }
        DummyInputRule::Interpolation => Some((
            *read_targets
                .first()
                .ok_or_else(|| anyhow!("{id} has no shift input"))?,
            F::ONE,
        )),
        DummyInputRule::Lookup => {
            ensure!(
                read_targets.len() == 1,
                "invalid read declaration for generator {id}"
            );
            let mut parameters = Vec::new();
            generator
                .serialize(&mut parameters, common)
                .map_err(|_| anyhow!("failed to serialize generator {id}"))?;
            let mut reader = Buffer::new(&parameters);
            let table = (|| {
                reader.read_usize()?; // row
                reader.read_usize()?; // slot within the row
                reader.read_usize() // common lookup-table index
            })()
            .map_err(|_| anyhow!("invalid serialized lookup parameters for {id}"))?;
            let &(key, _) = common
                .luts
                .get(table)
                .and_then(|lut| lut.first())
                .ok_or_else(|| anyhow!("{id} references missing or empty lookup table {table}"))?;
            Some((read_targets[0], F::from_canonical_u16(key)))
        }
    };
    let rep_index_of = |target: Target| -> Result<usize> {
        let target_index = target.index(probe_witness.num_wires, probe_witness.degree);
        let rep_index = *probe_witness
            .representative_map
            .get(target_index)
            .ok_or_else(|| {
                anyhow!("target {target:?} is outside the witness for generator {id}")
            })?;
        ensure!(
            rep_index < probe_witness.representative_map.len(),
            "invalid representative index {rep_index} for generator {id}"
        );
        Ok(rep_index)
    };
    let override_value = override_value
        .map(|(target, value)| rep_index_of(target).map(|rep_index| (rep_index, value)))
        .transpose()?;
    let mut rep_indices = read_targets
        .into_iter()
        .map(rep_index_of)
        .collect::<Result<Vec<_>>>()?;
    rep_indices.sort_unstable();
    rep_indices.dedup();

    let mut output_buffer = GeneratedValues::empty();
    let result = (|| -> Result<()> {
        let dense = probe_witness.dense_mut();
        for &rep_index in &rep_indices {
            dense.clear_assignment(rep_index);
        }
        for &rep_index in &rep_indices {
            // Apply overrides by representative so aliased read targets agree.
            let value = override_value
                .filter(|&(special_rep_index, _)| special_rep_index == rep_index)
                .map_or(F::ZERO, |(_, value)| value);
            dense.set(rep_index, value)?;
        }
        ensure!(
            generator.run(probe_witness, &mut output_buffer),
            "failed to determine output targets for generator {id}"
        );
        Ok(())
    })();
    let dense = probe_witness.dense_mut();
    for rep_index in rep_indices {
        dense.clear_assignment(rep_index);
    }
    result?;
    Ok(output_buffer
        .target_values
        .into_iter()
        .map(|(target, _)| target)
        .collect())
}

enum DummyInputRule {
    Zero,
    Quotient,
    Interpolation,
    Lookup,
}

fn dummy_input_rule(id: &str) -> Result<DummyInputRule> {
    match id {
        "QuotientGeneratorExtension" => Ok(DummyInputRule::Quotient),
        "InterpolationGenerator" => Ok(DummyInputRule::Interpolation),
        "LookupGenerator" => Ok(DummyInputRule::Lookup),
        "ArithmeticBaseGenerator"
        | "ArithmeticExtensionGenerator"
        | "MulExtensionGenerator"
        | "ReducingGenerator"
        | "ReducingExtensionGenerator"
        | "RandomAccessGenerator"
        | "ExponentiationGenerator"
        | "PoseidonGenerator"
        | "PoseidonMdsGenerator"
        | "Poseidon2FullGen"
        | "Poseidon2MdsGenerator"
        | "Poseidon2IntMixGenerator"
        | "LookupTableGenerator"
        | "DummyProofGenerator"
        | "SplitGenerator"
        | "WireSplitGenerator"
        | "EqualityGenerator"
        | "LowHighGenerator"
        | "CopyGenerator"
        | "RandomValueGenerator"
        | "NonzeroTestGenerator"
        | "ConstantGenerator" => Ok(DummyInputRule::Zero),
        _ => {
            if let Some(base) = id.strip_prefix("BaseSplitGenerator + Base: ") {
                let base = parse_base(id, base)?;
                ensure!(base != 0, "invalid zero base for {id}");
                return Ok(DummyInputRule::Zero);
            }
            if let Some(base) = id.strip_prefix("BaseSumGenerator + Base: ") {
                parse_base(id, base)?;
                return Ok(DummyInputRule::Zero);
            }
            Err(anyhow!("unsupported generator ID: {id}"))
        }
    }
}

fn parse_base(id: &str, base: &str) -> Result<usize> {
    base.parse()
        .map_err(|_| anyhow!("invalid base in built-in generator ID: {id}"))
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use crate::field::types::Field;
    use crate::gadgets::split_join::SplitGenerator;
    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
    use crate::gates::coset_interpolation::CosetInterpolationGate;
    use crate::gates::exponentiation::ExponentiationGate;
    use crate::gates::multiplication_extension::MulExtensionGate;
    use crate::gates::poseidon::PoseidonGate;
    use crate::gates::poseidon2::Poseidon2Gate;
    use crate::gates::poseidon2_int_mix::Poseidon2IntMixGate;
    use crate::gates::poseidon2_mds::Poseidon2MdsGate;
    use crate::gates::poseidon_mds::PoseidonMdsGate;
    use crate::gates::random_access::RandomAccessGate;
    use crate::gates::reducing::ReducingGate;
    use crate::gates::reducing_extension::ReducingExtensionGate;
    use crate::iop::generator::{NonzeroTestGenerator, SimpleGenerator, WitnessGeneratorRef};
    use crate::iop::witness::{PartialWitness, Witness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::recursion::dummy_circuit::DummyProofGenerator;
    use crate::util::serialization::{IoError, IoResult, Write};
    use crate::{C, D, F};

    fn family_fixture() -> Result<CircuitData<F, C, D>> {
        let config = CircuitConfig::standard_recursion_config();
        let inner = CircuitBuilder::<F, D>::new(config.clone()).build::<C>();
        let proof = inner.prove(PartialWitness::new())?;
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        builder.add_gate(
            ArithmeticGate::new_from_config(&config),
            vec![F::ONE, F::TWO],
        );
        builder.add_gate(
            ArithmeticExtensionGate::<D>::new_from_config(&config),
            vec![F::ONE, F::TWO],
        );
        builder.add_gate(
            MulExtensionGate::<D>::new_from_config(&config),
            vec![F::TWO],
        );
        for num_coeffs in [1, 3] {
            builder.add_gate(ReducingGate::<D>::new(num_coeffs), vec![]);
            builder.add_gate(ReducingExtensionGate::<D>::new(num_coeffs), vec![]);
        }
        for bits in [1, 3] {
            builder.add_gate(
                RandomAccessGate::<F, D>::new_from_config(&config, bits),
                vec![],
            );
            builder.add_gate(ExponentiationGate::<F, D>::new(bits), vec![]);
        }
        builder.add_gate(PoseidonGate::<F, D>::new(), vec![]);
        builder.add_gate(PoseidonMdsGate::<F, D>::new(), vec![]);
        builder.add_gate(Poseidon2Gate::<F, D>::new(), vec![]);
        builder.add_gate(Poseidon2MdsGate::<F, D>::new(), vec![]);
        builder.add_gate(Poseidon2IntMixGate::<F, D>::new(), vec![]);
        builder.add_gate(CosetInterpolationGate::<F, D>::new(2), vec![]);
        builder.add_gate(
            CosetInterpolationGate::<F, D>::with_max_degree(3, 3),
            vec![],
        );

        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let copied = builder.add_virtual_target();
        builder.generate_copy(a, copied);
        builder.is_equal(a, b);
        builder.split_low_high(a, 4, 8);
        builder.split_le(a, 8);
        builder.split_le_base::<3>(a, 4)?;
        let bits: Vec<_> = (0..30)
            .map(|_| builder.add_virtual_bool_target_unsafe())
            .collect();
        builder.le_sum(bits.iter());
        let extension = builder.add_virtual_extension_target();
        builder.inverse_extension(extension);
        builder.add_simple_generator(NonzeroTestGenerator {
            to_test: a,
            dummy: b,
        });
        builder.constant(F::from_canonical_u64(7));
        #[cfg(feature = "rand")]
        builder
            .add_simple_generator(crate::iop::generator::RandomValueGenerator { target: copied });

        // Deserialize a SplitGenerator with eight output targets.
        let mut parameters = Vec::new();
        parameters.write_target(a).unwrap();
        parameters
            .write_target_vec(&builder.add_virtual_targets(8))
            .unwrap();
        let split = <SplitGenerator as SimpleGenerator<F, D>>::deserialize(
            &mut Buffer::new(&parameters),
            &inner.common,
        )
        .unwrap();
        builder.add_simple_generator(split);

        let table = builder.add_lookup_table_from_pairs(Arc::new(vec![(3, 11), (7, 19)]));
        builder.add_lookup_from_index(a, table);
        let proof_with_pis_target = builder.add_virtual_proof_with_pis(&inner.common);
        let verifier_data_target = builder.add_virtual_verifier_data(config.fri_config.cap_height);
        builder.add_simple_generator(DummyProofGenerator {
            proof_with_pis_target,
            proof_with_pis: proof,
            verifier_data_target,
            verifier_data: inner.verifier_data(),
        });
        Ok(builder.build::<C>())
    }

    fn check_output_order(
        generator: &dyn WitnessGenerator<F, D>,
        data: &CircuitData<F, C, D>,
        probe: &mut PartitionWitness<'_, F>,
    ) -> Result<()> {
        let discovered = discover_outputs(generator, &data.common, probe)?;
        let reads = generator.reads().unwrap();
        assert!(
            !discovered.is_empty(),
            "{} fixture has no outputs",
            generator.id()
        );
        let rep = |target: Target| {
            data.prover_only.representative_map
                [target.index(data.common.config.num_wires, data.common.degree())]
        };
        for &target in &reads {
            assert!(
                !probe.contains(target),
                "{} left a sample input assigned",
                generator.id()
            );
        }
        for &target in &discovered {
            assert!(
                !probe.contains(target),
                "{} assigned an output during discovery",
                generator.id()
            );
        }
        // Exercise each generator with valid inputs for its computation.
        for value in [F::ZERO, F::ONE] {
            let mut witness = PartitionWitness::new(
                data.common.config.num_wires,
                data.common.degree(),
                &data.prover_only.representative_map,
            );
            let special = match generator.id().as_str() {
                "QuotientGeneratorExtension" => Some((reads[D], F::TWO)),
                "InterpolationGenerator" => Some((reads[0], F::from_canonical_u64(3))),
                "LookupGenerator" => Some((
                    reads[0],
                    if value == F::ZERO {
                        F::from_canonical_u64(3)
                    } else {
                        F::from_canonical_u64(7)
                    },
                )),
                "EqualityGenerator" if value == F::ZERO => Some((reads[1], F::ONE)),
                _ => None,
            };
            for &target in &reads {
                let assigned = special
                    .filter(|&(special_target, _)| rep(special_target) == rep(target))
                    .map_or(value, |(_, special_value)| special_value);
                witness.set_target(target, assigned)?;
            }
            let mut output_buffer = GeneratedValues::empty();
            generator.run_ready(&witness, &mut output_buffer)?;
            let actual: Vec<_> = output_buffer
                .target_values
                .iter()
                .map(|&(target, _)| target)
                .collect();
            assert_eq!(
                actual,
                discovered,
                "{} output order changed",
                generator.id()
            );
        }
        Ok(())
    }

    #[test]
    fn built_in_families_discover_the_same_output_order_as_execution() -> Result<()> {
        let data = family_fixture()?;
        // Check planner construction and output-target discovery for each supported family.
        WitnessGenerationPlanner::new(&data)?;
        let mut probe = PartitionWitness::new_dense(
            data.common.config.num_wires,
            data.common.degree(),
            &data.prover_only.representative_map,
        );
        let mut families = BTreeSet::new();
        for generator in &data.prover_only.generators {
            check_output_order(generator.0.as_ref(), &data, &mut probe)?;
            families.insert(generator.0.id().split(" + ").next().unwrap().to_owned());
        }
        let mut expected: BTreeSet<_> = [
            "ArithmeticBaseGenerator",
            "ArithmeticExtensionGenerator",
            "MulExtensionGenerator",
            "ReducingGenerator",
            "ReducingExtensionGenerator",
            "RandomAccessGenerator",
            "ExponentiationGenerator",
            "PoseidonGenerator",
            "PoseidonMdsGenerator",
            "Poseidon2FullGen",
            "Poseidon2MdsGenerator",
            "Poseidon2IntMixGenerator",
            "LookupGenerator",
            "LookupTableGenerator",
            "DummyProofGenerator",
            "SplitGenerator",
            "WireSplitGenerator",
            "EqualityGenerator",
            "LowHighGenerator",
            "CopyGenerator",
            "NonzeroTestGenerator",
            "ConstantGenerator",
            "BaseSplitGenerator",
            "BaseSumGenerator",
            "QuotientGeneratorExtension",
            "InterpolationGenerator",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        #[cfg(feature = "rand")]
        expected.insert("RandomValueGenerator".to_owned());
        assert_eq!(families, expected);
        Ok(())
    }

    #[derive(Debug)]
    struct UnsupportedGenerator {
        declares_reads: bool,
    }

    impl WitnessGenerator<F, D> for UnsupportedGenerator {
        fn id(&self) -> String {
            "UnsupportedTestGenerator".to_owned()
        }
        fn watch_list(&self) -> Vec<Target> {
            vec![]
        }
        fn reads(&self) -> Option<Vec<Target>> {
            self.declares_reads.then(Vec::new)
        }
        fn run(&self, _: &PartitionWitness<F>, _: &mut GeneratedValues<F>) -> bool {
            panic!("unsupported generator must be rejected before execution")
        }
        fn serialize(&self, _: &mut Vec<u8>, _: &CommonCircuitData<F, D>) -> IoResult<()> {
            Err(IoError)
        }
        fn deserialize(_: &mut Buffer, _: &CommonCircuitData<F, D>) -> IoResult<Self> {
            Err(IoError)
        }
    }

    #[test]
    fn preparation_rejects_unknown_generators_and_missing_read_declarations() {
        for declares_reads in [false, true] {
            let mut builder =
                CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
            builder.add_generators(vec![WitnessGeneratorRef::new(UnsupportedGenerator {
                declares_reads,
            })]);
            let data = builder.build::<C>();
            let error = WitnessGenerationPlanner::new(&data)
                .unwrap_err()
                .to_string();
            assert!(error.contains("UnsupportedTestGenerator"), "{error}");
            assert!(
                error.contains(if declares_reads {
                    "unsupported generator ID"
                } else {
                    "does not declare its reads"
                }),
                "{error}"
            );
        }
    }

    #[test]
    fn preparation_rejects_representatives_that_are_not_roots() {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let a = builder.add_virtual_target();
        let b = builder.add_virtual_target();
        let c = builder.add_virtual_target();
        builder.generate_copy(a, b);
        let mut data = builder.build::<C>();
        let index =
            |target: Target| target.index(data.common.config.num_wires, data.common.degree());
        let [a, b, c] = [a, b, c].map(index);
        data.prover_only.representative_map[a] = b;
        data.prover_only.representative_map[b] = c;
        data.prover_only.representative_map[c] = c;
        let error = WitnessGenerationPlanner::new(&data).unwrap_err();
        assert!(error.to_string().contains("is not a root"), "{error}");
    }

    #[test]
    fn malformed_base_generator_ids_are_rejected() {
        for id in [
            "BaseSplitGenerator + Base: 0",
            "BaseSplitGenerator + Base: invalid",
            "BaseSumGenerator + Base: invalid",
        ] {
            let error = dummy_input_rule(id)
                .err()
                .expect("invalid base should be rejected");
            assert!(error.to_string().contains("base"), "{error}");
        }
    }
}
