//! Assumed generation state, stage construction, and immutable execution plans.

#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::target::Target;
use crate::iop::witness_generation_planner::WitnessGenerationPlanner;
use crate::iop::witness_generation_schedule::{make_schedule, WitnessGenerationSchedule};
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// Tracks assignments and generator completion assumed during planning.
#[derive(Debug)]
pub(super) struct WitnessGenerationState {
    assumed_assigned_flags: Vec<u8>,
    scheduled_generator_flags: Vec<u8>,
}

impl WitnessGenerationState {
    fn new(num_targets: usize, num_generators: usize) -> Self {
        Self {
            assumed_assigned_flags: vec![0; num_targets],
            scheduled_generator_flags: vec![0; num_generators],
        }
    }

    pub(super) fn assumed_assigned_flags(&self) -> &[u8] {
        &self.assumed_assigned_flags
    }

    pub(super) fn scheduled_generator_flags(&self) -> &[u8] {
        &self.scheduled_generator_flags
    }

    pub(super) fn mark_assigned(&mut self, rep_index: usize) {
        self.assumed_assigned_flags[rep_index] = 1;
    }

    pub(super) fn mark_scheduled(&mut self, generator_index: usize) {
        self.scheduled_generator_flags[generator_index] = 1;
    }

    fn is_done(&self) -> bool {
        self.scheduled_generator_flags.iter().all(|&flag| flag != 0)
    }
}

/// An immutable sequence of witness generation stages for a circuit.
#[derive(Debug)]
pub struct WitnessGenerationPlan<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    circuit: &'a CircuitData<F, C, D>,
    stages: Vec<WitnessGenerationStage>,
    #[cfg(debug_assertions)]
    expected_output_targets: Arc<Vec<Vec<Target>>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGenerationPlan<'a, F, C, D>
{
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Circuit used to construct this plan.
    pub fn circuit(&self) -> &'a CircuitData<F, C, D> {
        self.circuit
    }

    /// Read-only stages in execution order.
    pub fn stages(&self) -> &[WitnessGenerationStage] {
        &self.stages
    }

    #[cfg(debug_assertions)]
    pub(super) fn expected_output_targets(&self, generator_index: usize) -> &[Target] {
        &self.expected_output_targets[generator_index]
    }
}

/// Required assignments and the generator schedule for one stage.
#[derive(Debug)]
pub struct WitnessGenerationStage {
    promised_rep_indices: Vec<usize>,
    schedule: WitnessGenerationSchedule,
}

impl WitnessGenerationStage {
    pub(super) fn promised_rep_indices(&self) -> &[usize] {
        &self.promised_rep_indices
    }

    /// Read-only generator graph for this stage.
    pub fn schedule(&self) -> &WitnessGenerationSchedule {
        &self.schedule
    }
}

/// Builds stages from assignment assumptions and generator dependencies.
#[derive(Debug)]
pub struct WitnessGenerationPlanning<
    'p,
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    planner: &'p WitnessGenerationPlanner<'a, F, C, D>,
    promised_rep_indices: Vec<usize>,
    newly_produced: Vec<usize>,
    state: WitnessGenerationState,
    plan: WitnessGenerationPlan<'a, F, C, D>,
}

impl<'p, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGenerationPlanning<'p, 'a, F, C, D>
{
    pub(super) fn new(planner: &'p WitnessGenerationPlanner<'a, F, C, D>) -> Self {
        let circuit = planner.circuit();
        Self {
            planner,
            promised_rep_indices: Vec::new(),
            newly_produced: Vec::new(),
            state: WitnessGenerationState::new(
                circuit.prover_only.representative_map.len(),
                planner.reads().len(),
            ),
            plan: WitnessGenerationPlan {
                circuit,
                stages: Vec::new(),
                #[cfg(debug_assertions)]
                expected_output_targets: Arc::clone(planner.expected_output_targets()),
            },
        }
    }

    /// Records a target assignment required before the next stage executes.
    /// Targets connected by copy constraints share one assignment requirement.
    pub fn assume_assigned(&mut self, target: Target) -> Result<()> {
        let circuit = self.planner.circuit();
        let target_index = target.index(circuit.common.config.num_wires, circuit.common.degree());
        let rep_index = *circuit
            .prover_only
            .representative_map
            .get(target_index)
            .ok_or_else(|| anyhow!("target {target:?} is outside the witness"))?;
        self.assume_representative(rep_index);
        Ok(())
    }

    fn assume_representative(&mut self, rep_index: usize) {
        if self.state.assumed_assigned_flags()[rep_index] == 0 {
            self.promised_rep_indices.push(rep_index);
        }
    }

    /// Schedules generators enabled by the recorded assignments and appends a stage.
    pub fn add_stage(&mut self) {
        let mut promised_rep_indices = core::mem::take(&mut self.promised_rep_indices);
        promised_rep_indices.sort_unstable();
        promised_rep_indices.dedup();
        for &rep_index in &promised_rep_indices {
            self.state.mark_assigned(rep_index);
        }
        let (schedule, newly_produced) = make_schedule(
            self.planner.reads(),
            self.planner.writes(),
            self.planner.consumers(),
            &mut self.state,
        );
        self.newly_produced = newly_produced;
        self.plan.stages.push(WitnessGenerationStage {
            promised_rep_indices,
            schedule,
        });
    }

    /// Returns whether all generators are scheduled and all assignment requirements belong to stages.
    pub fn is_done(&self) -> bool {
        self.state.is_done() && self.promised_rep_indices.is_empty()
    }

    /// Returns the immutable plan when all generators are scheduled and all
    /// assignment requirements belong to stages. Returns the planning object
    /// when more stages are required.
    pub fn finish(self) -> core::result::Result<Arc<WitnessGenerationPlan<'a, F, C, D>>, Self> {
        if self.is_done() {
            Ok(Arc::new(self.plan))
        } else {
            Err(self)
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use hashbrown::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::field::types::Field;
    use crate::iop::generator::{GeneratedValues, WitnessGenerator, WitnessGeneratorRef};
    use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
    use crate::iop::witness_generation_executor::WitnessGenerationExecutor;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
    use crate::util::serialization::{Buffer, IoError, IoResult};
    use crate::{C, D, F};

    #[cfg(feature = "parallel")]
    const EXECUTION_MODES: &[Option<usize>] = &[None, Some(1), Some(4)];
    #[cfg(not(feature = "parallel"))]
    const EXECUTION_MODES: &[Option<usize>] = &[None];

    #[test]
    fn schedules_preserve_dependencies_and_leave_blocked_work_available_for_later() {
        // Fan-out, join, two reader-free writers, and a separate two-generator cycle.
        let reads = vec![
            vec![0],
            vec![1],
            vec![1],
            vec![2, 3],
            vec![0],
            vec![5],
            vec![6],
        ];
        let writes = vec![
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![4],
            vec![6],
            vec![5],
        ];
        let consumers = HashMap::from_iter([
            (0, vec![0, 4]),
            (1, vec![1, 2]),
            (2, vec![3]),
            (3, vec![3]),
            (5, vec![5]),
            (6, vec![6]),
        ]);
        let mut state = WitnessGenerationState::new(7, 7);
        let (empty, produced) = make_schedule(&reads, &writes, &consumers, &mut state);
        assert!(empty.generator_indices().is_empty());
        assert!(produced.is_empty());
        state.mark_assigned(0);
        let (first, produced) = make_schedule(&reads, &writes, &consumers, &mut state);
        assert_eq!(first.generator_indices(), &[0, 1, 2, 3, 4]);
        assert_eq!(first.dependency_counts(), &[0, 1, 1, 2, 1]);
        assert_eq!(first.initial_ready(), &[0]);
        assert_eq!(first.successors(0), &[1, 2]);
        assert_eq!(first.successors(1), &[3]);
        assert_eq!(first.successors(2), &[3]);
        assert_eq!(first.successors(3), &[4]);
        assert_eq!(produced, vec![1, 2, 3, 4]);
        assert!(!state.is_done());
        state.mark_assigned(5);
        let (second, produced) = make_schedule(&reads, &writes, &consumers, &mut state);
        assert_eq!(second.generator_indices(), &[5, 6]);
        assert_eq!(second.dependency_counts(), &[0, 1]);
        assert_eq!(second.successors(0), &[1]);
        assert_eq!(produced, vec![6]);
        assert!(state.is_done());
    }

    #[derive(Debug)]
    struct FallibleGenerator {
        input: Target,
        output: Target,
    }

    impl WitnessGenerator<F, D> for FallibleGenerator {
        fn id(&self) -> String {
            "FallibleTestGenerator".to_owned()
        }
        fn watch_list(&self) -> Vec<Target> {
            vec![self.input]
        }
        fn reads(&self) -> Option<Vec<Target>> {
            Some(vec![self.input])
        }
        fn run(
            &self,
            witness: &PartitionWitness<F>,
            output_buffer: &mut GeneratedValues<F>,
        ) -> bool {
            self.run_ready(witness, output_buffer).is_ok()
        }
        fn run_ready(
            &self,
            witness: &PartitionWitness<F>,
            output_buffer: &mut GeneratedValues<F>,
        ) -> Result<()> {
            let value = witness.get_target(self.input);
            anyhow::ensure!(value != F::ZERO, "injected generator error");
            assert!(value != F::ONE, "injected generator panic");
            output_buffer.set_target(self.output, value)
        }
        fn serialize(&self, _: &mut Vec<u8>, _: &CommonCircuitData<F, D>) -> IoResult<()> {
            Err(IoError)
        }
        fn deserialize(_: &mut Buffer, _: &CommonCircuitData<F, D>) -> IoResult<Self> {
            Err(IoError)
        }
    }

    #[test]
    fn errors_and_panics_invalidate_execution_and_allow_reset_and_executor_reuse() -> Result<()> {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let input = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        let mut data = builder.build::<C>();
        // Exercise errors, panics and recovery with fixed read and output targets.
        data.prover_only.generators = vec![WitnessGeneratorRef::new(FallibleGenerator {
            input,
            output,
        })];
        let rep = |target: Target| {
            data.prover_only.representative_map
                [target.index(data.common.config.num_wires, data.common.degree())]
        };
        let input_rep = rep(input);
        let output_rep = rep(output);
        let mut state = WitnessGenerationState::new(data.prover_only.representative_map.len(), 1);
        state.mark_assigned(input_rep);
        let (schedule, _) = make_schedule(
            &[vec![input_rep]],
            &[vec![output_rep]],
            &HashMap::new(),
            &mut state,
        );
        let plan = Arc::new(WitnessGenerationPlan {
            circuit: &data,
            stages: vec![WitnessGenerationStage {
                promised_rep_indices: vec![input_rep],
                schedule,
            }],
            #[cfg(debug_assertions)]
            expected_output_targets: Arc::new(vec![vec![output]]),
        });
        for bad_value in [F::ZERO, F::ONE] {
            for &num_workers in EXECUTION_MODES {
                let mut executor = WitnessGenerationExecutor::new(Arc::clone(&plan));
                let mut witness = executor.new_witness();
                let mut execution = executor.bind(&mut witness)?;
                execution.set_target(input, bad_value)?;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    match num_workers {
                        None => execution.execute(),
                        #[cfg(feature = "parallel")]
                        Some(workers) => {
                            // SAFETY: The test generator declares its exact fixed access sets.
                            unsafe { execution.execute_parallel(workers) }
                        }
                        #[cfg(not(feature = "parallel"))]
                        Some(_) => unreachable!("parallel execution requires the parallel feature"),
                    }
                }));
                if bad_value == F::ZERO {
                    assert!(result
                        .unwrap()
                        .unwrap_err()
                        .to_string()
                        .contains("injected generator error"));
                } else {
                    assert!(result.is_err());
                }
                assert!(!execution.is_done());
                assert!(execution
                    .execute()
                    .unwrap_err()
                    .to_string()
                    .contains("reset before continuing"));
                assert!(execution.set_target(input, F::TWO).is_err());
                let mut execution = execution
                    .finish()
                    .expect_err("failed execution cannot finish");
                execution.reset();
                assert!(!execution.witness().contains(input));
                assert!(!execution.witness().contains(output));
                execution.set_target(input, F::TWO)?;
                match num_workers {
                    None => execution.execute()?,
                    #[cfg(feature = "parallel")]
                    Some(workers) => {
                        // SAFETY: The same fixed access contract holds after reset.
                        unsafe { execution.execute_parallel(workers)? };
                    }
                    #[cfg(not(feature = "parallel"))]
                    Some(_) => unreachable!("parallel execution requires the parallel feature"),
                }
                assert_eq!(execution.witness().get_target(output), F::TWO);
                execution.finish().unwrap();
                let mut next = executor.new_witness();
                next.set_target(input, F::from_canonical_u64(3))?;
                executor.execute(&mut next)?;
                assert_eq!(next.get_target(output), F::from_canonical_u64(3));
            }
        }
        Ok(())
    }
}
