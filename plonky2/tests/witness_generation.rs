#![cfg(feature = "std")]

use std::collections::HashSet;

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::gates::noop::NoopGate;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::generation_schedule::GenerationSchedule;
use plonky2::iop::generator::{generate_partial_witness, GeneratedValues};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use plonky2::iop::witness_generator_executor::WitnessGeneratorExecutor;
use plonky2::iop::witness_generator_scheduler::WitnessGeneratorScheduler;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;
use plonky2::{C, D, F};

type Scheduler<'a> = WitnessGeneratorScheduler<'a, F, C, D>;

fn make_schedule(
    data: &CircuitData<F, C, D>,
    scheduler: &Scheduler<'_>,
    inputs: &PartialWitness<F>,
) -> Result<GenerationSchedule> {
    let mut assumed_assigned = vec![0; data.prover_only.representative_map.len()];
    for &target in inputs.target_values.keys() {
        let target_index = target.index(data.common.config.num_wires, data.common.degree());
        assumed_assigned[data.prover_only.representative_map[target_index]] = 1;
    }
    scheduler.make_complete_schedule(&assumed_assigned)
}

fn generate<'a>(
    scheduler: &Scheduler<'a>,
    schedule: &GenerationSchedule,
    inputs: &PartialWitness<F>,
    num_workers: Option<usize>,
) -> Result<PartitionWitness<'a, F>> {
    let mut witness = scheduler.new_witness();
    let mut executor = WitnessGeneratorExecutor::new(scheduler, &mut witness)?;
    for (&target, &value) in &inputs.target_values {
        executor.set_target(target, value)?;
    }
    if let Some(num_workers) = num_workers {
        // SAFETY: These fixtures use built-in generators with complete read
        // declarations and fixed outputs, through target-based witness access.
        unsafe { executor.run_schedule_parallel(schedule, num_workers) }?;
    } else {
        executor.run_schedule(schedule)?;
    }
    drop(executor);
    Ok(witness)
}

fn assert_witnesses_match(
    data: &CircuitData<F, C, D>,
    scheduler: &Scheduler<'_>,
    expected: &PartitionWitness<'_, F>,
    actual: &PartitionWitness<'_, F>,
) {
    let random_rep_indices: HashSet<_> = data
        .prover_only
        .generators
        .iter()
        .enumerate()
        .filter(|(_, generator)| generator.0.id() == "RandomValueGenerator")
        .flat_map(|(generator_index, _)| scheduler.writes(generator_index).iter().copied())
        .collect();
    for generator_index in 0..data.prover_only.generators.len() {
        assert!(
            scheduler
                .reads(generator_index)
                .iter()
                .all(|rep_index| !random_rep_indices.contains(rep_index)),
            "fixture randomness must not affect generator computations"
        );
    }
    let num_wires = data.common.config.num_wires;
    let num_wire_targets = num_wires * data.common.degree();
    for (target_index, &rep_index) in data.prover_only.representative_map.iter().enumerate() {
        let target = if target_index < num_wire_targets {
            Target::wire(target_index / num_wires, target_index % num_wires)
        } else {
            Target::VirtualTarget {
                index: target_index - num_wire_targets,
            }
        };
        let expected = expected.try_get_target(target);
        let actual = actual.try_get_target(target);
        assert_eq!(
            actual.is_some(),
            expected.is_some(),
            "presence at {target:?}"
        );
        // Unused public-input wires are randomized independently by each run.
        if !random_rep_indices.contains(&rep_index) {
            assert_eq!(actual, expected, "value at {target:?}");
        }
    }
}

#[test]
fn mixed_circuit_matches_original_generation_and_proves() -> Result<()> {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let alias = builder.add_virtual_target();
    builder.connect(a, alias);
    let b = builder.add_virtual_target();
    let copied = builder.add_virtual_target();
    builder.generate_copy(alias, copied);
    let sum = builder.add(copied, b);
    let product = builder.mul(sum, b);
    builder.split_le(a, 8);
    let extension = builder.add_virtual_extension_target();
    let squared = builder.mul_extension(extension, extension);
    let extension_sum = builder.add_extension(squared, extension);
    let selector = builder.add_virtual_target();
    let selected = builder.random_access(selector, vec![a, b, sum, product]);
    let hash = builder.hash_n_to_hash_no_pad::<PoseidonHash>(vec![
        selected,
        product,
        extension_sum.0[0],
        extension_sum.0[1],
    ]);
    builder.register_public_inputs(&hash.elements);
    builder.register_public_input(sum);
    let unused = builder.add_virtual_target();
    let data = builder.build::<C>();
    let scheduler = Scheduler::new(&data)?;
    for (case_index, (a_value, b_value, extension_values, selector_value)) in [
        (0, 0, [0, 0], 0),
        (1, 2, [3, 4], 1),
        (17, 9, [5, 7], 2),
        (255, 13, [11, 17], 3),
    ]
    .into_iter()
    .enumerate()
    {
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, F::from_canonical_u64(a_value))?;
        inputs.set_target(alias, F::from_canonical_u64(a_value))?;
        inputs.set_target(b, F::from_canonical_u64(b_value))?;
        inputs.set_target(selector, F::from_canonical_u64(selector_value))?;
        for (target, value) in extension.0.into_iter().zip(extension_values) {
            inputs.set_target(target, F::from_canonical_u64(value))?;
        }
        let schedule = make_schedule(&data, &scheduler, &inputs)?;
        assert_eq!(
            schedule.generator_indices().len(),
            data.prover_only.generators.len()
        );
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common)?;
        for num_workers in [None, Some(1), Some(4)] {
            let actual = generate(&scheduler, &schedule, &inputs, num_workers)?;
            assert_witnesses_match(&data, &scheduler, &expected, &actual);
            assert_eq!(actual.get_target(a), actual.get_target(alias));
            assert!(!actual.contains(unused));
            if case_index == 1 && num_workers == Some(4) {
                let proof = prove_with_partition_witness(
                    &data.prover_only,
                    &data.common,
                    actual,
                    &mut TimingTree::default(),
                )?;
                assert_eq!(
                    proof.public_inputs,
                    expected.get_targets(&data.prover_only.public_inputs)
                );
                data.verify(proof)?;
            }
        }
    }
    Ok(())
}

#[test]
fn duplicate_writers_are_compared_without_readers() -> Result<()> {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();
    let row = builder.add_gate(NoopGate, vec![]);
    let first = Target::wire(row, 0);
    let second = Target::wire(row, 1);
    builder.generate_copy(a, first);
    builder.generate_copy(b, second);
    builder.connect(first, second);
    let data = builder.build::<C>();
    let scheduler = Scheduler::new(&data)?;
    let rep_index = data.prover_only.representative_map
        [first.index(data.common.config.num_wires, data.common.degree())];
    for generator_index in 0..data.prover_only.generators.len() {
        assert!(!scheduler.reads(generator_index).contains(&rep_index));
    }
    for b_value in [3, 4] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, F::from_canonical_u64(3))?;
        inputs.set_target(b, F::from_canonical_u64(b_value))?;
        let schedule = make_schedule(&data, &scheduler, &inputs)?;
        let writer_local_indices: Vec<_> = schedule
            .generator_indices()
            .iter()
            .enumerate()
            .filter(|(_, &generator_index)| scheduler.writes(generator_index).contains(&rep_index))
            .map(|(local_index, _)| local_index)
            .collect();
        assert_eq!(writer_local_indices.len(), 2);
        assert!(schedule
            .successors(writer_local_indices[0])
            .contains(&writer_local_indices[1]));
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common);
        if b_value == 3 {
            let expected = expected?;
            for num_workers in [None, Some(1), Some(4)] {
                let actual = generate(&scheduler, &schedule, &inputs, num_workers)?;
                assert_witnesses_match(&data, &scheduler, &expected, &actual);
                assert_eq!(actual.get_target(first), F::from_canonical_u64(3));
                assert_eq!(actual.get_target(first), actual.get_target(second));
            }
        } else {
            assert!(expected.is_err());
            for num_workers in [None, Some(1), Some(4)] {
                let error = generate(&scheduler, &schedule, &inputs, num_workers).unwrap_err();
                assert!(error.to_string().contains("different values"), "{error}");
            }
        }
    }
    Ok(())
}

#[test]
fn repeated_emissions_to_one_representative_must_agree() -> Result<()> {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let input = builder.add_virtual_target();
    let bits = builder.split_le(input, 4);
    builder.connect(bits[0].target, bits[1].target);
    let data = builder.build::<C>();
    let scheduler = Scheduler::new(&data)?;
    for value in [3, 1] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(input, F::from_canonical_u64(value))?;
        let schedule = make_schedule(&data, &scheduler, &inputs)?;
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common);
        if value == 3 {
            let expected = expected?;
            // Confirm that the fixture exercises repeated emissions by one generator.
            let rep_index = data.prover_only.representative_map[bits[0]
                .target
                .index(data.common.config.num_wires, data.common.degree())];
            let mut buffer = GeneratedValues::empty();
            let mut repeated = false;
            for generator in &data.prover_only.generators {
                buffer.target_values.clear();
                generator.0.run_ready(&expected, &mut buffer)?;
                let count = buffer
                    .target_values
                    .iter()
                    .filter(|(target, _)| {
                        data.prover_only.representative_map
                            [target.index(data.common.config.num_wires, data.common.degree())]
                            == rep_index
                    })
                    .count();
                repeated |= count > 1;
            }
            assert!(repeated);
            for num_workers in [None, Some(1), Some(4)] {
                let actual = generate(&scheduler, &schedule, &inputs, num_workers)?;
                assert_witnesses_match(&data, &scheduler, &expected, &actual);
            }
        } else {
            assert!(expected.is_err());
            for num_workers in [None, Some(1), Some(4)] {
                let error = generate(&scheduler, &schedule, &inputs, num_workers).unwrap_err();
                assert!(error.to_string().contains("different values"), "{error}");
            }
        }
    }
    Ok(())
}

#[test]
fn caller_prefills_must_agree_with_aliases_and_generated_values() -> Result<()> {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let alias = builder.add_virtual_target();
    builder.connect(a, alias);
    let b = builder.add_virtual_target();
    let sum = builder.add(a, b);
    let data = builder.build::<C>();
    let scheduler = Scheduler::new(&data)?;
    for (alias_value, sum_value) in [(3, 8), (3, 9), (4, 8)] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, F::from_canonical_u64(3))?;
        inputs.set_target(alias, F::from_canonical_u64(alias_value))?;
        inputs.set_target(b, F::from_canonical_u64(5))?;
        inputs.set_target(sum, F::from_canonical_u64(sum_value))?;
        let schedule = make_schedule(&data, &scheduler, &inputs)?;
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common);
        if alias_value == 3 && sum_value == 8 {
            let expected = expected?;
            for num_workers in [None, Some(1), Some(4)] {
                let actual = generate(&scheduler, &schedule, &inputs, num_workers)?;
                assert_witnesses_match(&data, &scheduler, &expected, &actual);
            }
        } else {
            assert!(expected.is_err());
            for num_workers in [None, Some(1), Some(4)] {
                let error = generate(&scheduler, &schedule, &inputs, num_workers).unwrap_err();
                assert!(error.to_string().contains("different values"), "{error}");
            }
        }
    }
    Ok(())
}
