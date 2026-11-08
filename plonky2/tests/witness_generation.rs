#![cfg(feature = "std")]

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::gates::noop::NoopGate;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::iop::generator::{generate_partial_witness, GeneratedValues};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use plonky2::iop::witness_generation_executor::{
    WitnessGenerationExecution as Execution, WitnessGenerationExecutor as Executor,
};
use plonky2::iop::witness_generation_plan::WitnessGenerationPlan as Plan;
use plonky2::iop::witness_generation_planner::WitnessGenerationPlanner as Planner;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::prover::prove_with_partition_witness;
use plonky2::util::timing::TimingTree;
use plonky2::{C, D, F};

fn make_plan<'a>(
    data: &'a CircuitData<F, C, D>,
    inputs: impl IntoIterator<Item = Target>,
) -> Result<Arc<Plan<'a, F, C, D>>> {
    let planner = Planner::new(data)?;
    let mut planning = planner.new_plan();
    for target in inputs {
        planning.set_assumed_assigned(target)?;
    }
    planning.add_stage();
    Ok(planning
        .finish()
        .expect("fixture inputs must enable all generators"))
}

fn execute_stage(
    execution: &mut Execution<'_, '_, F, C, D>,
    num_workers: Option<usize>,
) -> Result<()> {
    if let Some(num_workers) = num_workers {
        // SAFETY: The fixtures use genuine built-ins with fixed, complete access sets.
        unsafe { execution.execute_parallel(num_workers) }
    } else {
        execution.execute()
    }
}

fn generate<'a>(
    plan: &Arc<Plan<'a, F, C, D>>,
    inputs: &PartialWitness<F>,
    num_workers: Option<usize>,
) -> Result<PartitionWitness<'a, F>> {
    let mut executor = Executor::new(Arc::clone(plan));
    let mut witness = executor.new_witness();
    for (&target, &value) in &inputs.target_values {
        witness.set_target(target, value)?;
    }
    if let Some(num_workers) = num_workers {
        // SAFETY: The fixtures use genuine built-ins with fixed, complete access sets.
        unsafe { executor.execute_parallel(&mut witness, num_workers)? };
    } else {
        executor.execute(&mut witness)?;
    }
    Ok(witness)
}

fn assert_witnesses_match(
    data: &CircuitData<F, C, D>,
    expected: &PartitionWitness<'_, F>,
    actual: &PartitionWitness<'_, F>,
) {
    let rep_index = |target: Target| {
        data.prover_only.representative_map
            [target.index(data.common.config.num_wires, data.common.degree())]
    };
    let mut random_rep_indices = HashSet::new();
    let mut buffer = GeneratedValues::empty();
    for generator in &data.prover_only.generators {
        if generator.0.id() == "RandomValueGenerator" {
            buffer.target_values.clear();
            generator.0.run_ready(expected, &mut buffer).unwrap();
            random_rep_indices.extend(
                buffer
                    .target_values
                    .iter()
                    .map(|&(target, _)| rep_index(target)),
            );
        }
    }
    for generator in &data.prover_only.generators {
        assert!(
            generator
                .0
                .reads()
                .unwrap()
                .into_iter()
                .all(|target| !random_rep_indices.contains(&rep_index(target))),
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
    let plan = make_plan(
        &data,
        [a, alias, b, selector].into_iter().chain(extension.0),
    )?;
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
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common)?;
        for num_workers in [None, Some(1), Some(4)] {
            let actual = generate(&plan, &inputs, num_workers)?;
            assert_witnesses_match(&data, &expected, &actual);
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
    let plan = Arc::clone(Executor::new_with_inputs(&data, [a, b])?.plan());
    let rep_index = data.prover_only.representative_map
        [first.index(data.common.config.num_wires, data.common.degree())];
    let representative = |target: Target| {
        data.prover_only.representative_map
            [target.index(data.common.config.num_wires, data.common.degree())]
    };
    for generator in &data.prover_only.generators {
        assert!(generator
            .0
            .reads()
            .unwrap()
            .into_iter()
            .all(|target| representative(target) != rep_index));
    }
    for b_value in [3, 4] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, F::from_canonical_u64(3))?;
        inputs.set_target(b, F::from_canonical_u64(b_value))?;
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common);
        if b_value == 3 {
            let expected = expected?;
            for num_workers in [None, Some(1), Some(4)] {
                let actual = generate(&plan, &inputs, num_workers)?;
                assert_witnesses_match(&data, &expected, &actual);
                assert_eq!(actual.get_target(first), F::from_canonical_u64(3));
                assert_eq!(actual.get_target(first), actual.get_target(second));
            }
        } else {
            assert!(expected.is_err());
            for num_workers in [None, Some(1), Some(4)] {
                let error = generate(&plan, &inputs, num_workers).unwrap_err();
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
    let error = Executor::new_with_inputs(&data, []).unwrap_err();
    assert!(error
        .to_string()
        .contains("complete witness generation plan"));
    let plan = Arc::clone(Executor::new_with_inputs(&data, [input])?.plan());
    for value in [3, 1] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(input, F::from_canonical_u64(value))?;
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
                let actual = generate(&plan, &inputs, num_workers)?;
                assert_witnesses_match(&data, &expected, &actual);
            }
        } else {
            assert!(expected.is_err());
            for num_workers in [None, Some(1), Some(4)] {
                let error = generate(&plan, &inputs, num_workers).unwrap_err();
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
    let plan = Arc::clone(Executor::new_with_inputs(&data, [a, alias, b, sum])?.plan());
    for (alias_value, sum_value) in [(3, 8), (3, 9), (4, 8)] {
        let mut inputs = PartialWitness::new();
        inputs.set_target(a, F::from_canonical_u64(3))?;
        inputs.set_target(alias, F::from_canonical_u64(alias_value))?;
        inputs.set_target(b, F::from_canonical_u64(5))?;
        inputs.set_target(sum, F::from_canonical_u64(sum_value))?;
        let expected = generate_partial_witness(inputs.clone(), &data.prover_only, &data.common);
        if alias_value == 3 && sum_value == 8 {
            let expected = expected?;
            for num_workers in [None, Some(1), Some(4)] {
                let actual = generate(&plan, &inputs, num_workers)?;
                assert_witnesses_match(&data, &expected, &actual);
            }
        } else {
            assert!(expected.is_err());
            for num_workers in [None, Some(1), Some(4)] {
                let error = generate(&plan, &inputs, num_workers).unwrap_err();
                assert!(error.to_string().contains("different values"), "{error}");
            }
        }
    }
    Ok(())
}

#[test]
fn staged_plans_validate_inputs_and_support_reset_and_reuse() -> Result<()> {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let a = builder.add_virtual_target();
    let alias = builder.add_virtual_target();
    builder.connect(a, alias);
    let b = builder.add_virtual_target();
    let x = builder.add_virtual_target();
    let y = builder.add_virtual_target();
    builder.generate_copy(a, x);
    builder.generate_copy(b, y);
    let sum = builder.add(x, y);
    let data = builder.build::<C>();
    let plan = {
        let planner = Planner::new(&data)?;
        let mut planning = planner.new_plan();
        planning.add_stage();
        let mut planning = planning.finish().expect_err("inputs are still unavailable");
        planning.set_assumed_assigned(a)?;
        planning.set_assumed_assigned(alias)?;
        planning.add_stage();
        planning.add_stage();
        planning.set_assumed_assigned(b)?;
        assert!(!planning.is_done());
        planning.add_stage();
        assert!(planning.is_done());
        planning.finish().unwrap()
    };
    // The plan and its debug declarations outlive the planner.
    assert!(core::ptr::eq(plan.circuit(), &data));
    assert_eq!(plan.num_stages(), 4);
    assert!(plan.stages()[2].schedule().generator_indices().is_empty());

    for (dense, num_workers) in [
        (false, None),
        (true, None),
        (true, Some(1)),
        (true, Some(4)),
    ] {
        let mut executor = Executor::new(Arc::clone(&plan));
        let mut witness = if dense {
            executor.new_witness()
        } else {
            PartitionWitness::new(
                data.common.config.num_wires,
                data.common.degree(),
                &data.prover_only.representative_map,
            )
        };
        let values_pointer = if dense {
            witness.dense_values_and_assignment_flags()?.0.as_ptr() as usize
        } else {
            witness.values.as_ptr() as usize
        };
        let execution = executor.bind(&mut witness)?;
        let mut execution = execution.finish().expect_err("no stages have run");
        if !dense {
            let before = execution.witness().values.clone();
            // SAFETY: The fixture uses genuine built-ins with complete access sets.
            let error = unsafe { execution.execute_parallel(4) }.unwrap_err();
            assert!(
                error.to_string().contains("requires dense storage"),
                "{error}"
            );
            assert_eq!(execution.witness().values, before);
        }
        for (a_value, b_value) in [(3, 5), (7, 11)] {
            execute_stage(&mut execution, num_workers)?;
            let error = execute_stage(&mut execution, num_workers).unwrap_err();
            assert!(error.to_string().contains("not assigned"), "{error}");
            assert!(!execution.witness().contains(x));
            execution.set_target(alias, F::from_canonical_u64(a_value))?;
            if a_value == 3 {
                execution.set_target(b, F::from_canonical_u64(b_value))?;
            }
            execute_stage(&mut execution, num_workers)?;
            assert_eq!(
                execution.witness().get_target(x),
                F::from_canonical_u64(a_value)
            );
            assert!(!execution.witness().contains(y));
            execute_stage(&mut execution, num_workers)?;
            assert!(!execution.witness().contains(y));
            if a_value != 3 {
                execution.set_target(b, F::from_canonical_u64(b_value))?;
            }
            execute_stage(&mut execution, num_workers)?;
            assert!(execution.is_done());
            assert!(execute_stage(&mut execution, num_workers).is_err());
            assert!(execution.is_done());
            let mut inputs = PartialWitness::new();
            inputs.set_target(a, F::from_canonical_u64(a_value))?;
            inputs.set_target(b, F::from_canonical_u64(b_value))?;
            let expected = generate_partial_witness(inputs, &data.prover_only, &data.common)?;
            assert_witnesses_match(&data, &expected, execution.witness());
            assert_eq!(
                execution.witness().get_target(sum),
                F::from_canonical_u64(a_value + b_value)
            );
            if a_value == 3 {
                execution.reset();
                assert!(!execution.is_done());
                for target in [a, alias, b, x, y, sum] {
                    assert!(!execution.witness().contains(target));
                }
            }
        }
        execution.finish().unwrap();
        let current_pointer = if dense {
            witness.dense_values_and_assignment_flags()?.0.as_ptr() as usize
        } else {
            witness.values.as_ptr() as usize
        };
        assert_eq!(current_pointer, values_pointer);
    }
    Ok(())
}

#[test]
fn incompatible_witness_is_rejected_without_changes() -> Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    let input = builder.add_virtual_target();
    builder.add(input, input);
    let data = builder.build::<C>();
    let mut executor = Executor::new_with_inputs(&data, [input])?;
    let mut other_builder = CircuitBuilder::<F, D>::new(config);
    let other_input = other_builder.add_virtual_target();
    other_builder.add(other_input, other_input);
    let other_data = other_builder.build::<C>();
    let other = Executor::new_with_inputs(&other_data, [other_input])?;
    let mut witness = other.new_witness();
    witness.set_target(other_input, F::TWO)?;
    let error = executor.bind(&mut witness).unwrap_err();
    assert!(
        error.to_string().contains("does not match plan circuit"),
        "{error}"
    );
    assert_eq!(witness.get_target(other_input), F::TWO);
    Ok(())
}

#[test]
fn invalid_random_access_index_errors_and_allows_recovery() -> Result<()> {
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
    let selector = builder.add_virtual_target();
    let values = builder.add_virtual_targets(4);
    let selected = builder.random_access(selector, values.clone());
    let data = builder.build::<C>();
    let plan = make_plan(
        &data,
        core::iter::once(selector).chain(values.iter().copied()),
    )?;
    for num_workers in [None, Some(1), Some(4)] {
        let mut executor = Executor::new(Arc::clone(&plan));
        let mut witness = executor.new_witness();
        let mut execution = executor.bind(&mut witness)?;
        for invalid in [F::from_canonical_u64(4), F::NEG_ONE] {
            execution.set_target(selector, invalid)?;
            for (index, &target) in values.iter().enumerate() {
                execution.set_target(target, F::from_canonical_usize(index + 10))?;
            }
            let error = execute_stage(&mut execution, num_workers).unwrap_err();
            assert!(error.to_string().contains("random access index"), "{error}");
            execution.reset();
        }
        execution.set_target(selector, F::TWO)?;
        for (index, &target) in values.iter().enumerate() {
            execution.set_target(target, F::from_canonical_usize(index + 10))?;
        }
        execute_stage(&mut execution, num_workers)?;
        assert_eq!(
            execution.witness().get_target(selected),
            F::from_canonical_u64(12)
        );
        execution
            .finish()
            .expect("recovered execution must complete");
    }
    Ok(())
}
