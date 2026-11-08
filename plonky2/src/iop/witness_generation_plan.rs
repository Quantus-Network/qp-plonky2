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

/// Assumed assignments and completed generators, independently of execution.
#[derive(Debug)]
pub(super) struct WitnessGenerationState {
    assumed_assigned: Vec<u8>,
    generators_scheduled: Vec<u8>,
}

impl WitnessGenerationState {
    fn new(num_targets: usize, num_generators: usize) -> Self {
        Self {
            assumed_assigned: vec![0; num_targets],
            generators_scheduled: vec![0; num_generators],
        }
    }

    pub(super) fn assumed_assigned(&self) -> &[u8] {
        &self.assumed_assigned
    }

    pub(super) fn generators_scheduled(&self) -> &[u8] {
        &self.generators_scheduled
    }

    pub(super) fn mark_assigned(&mut self, rep_index: usize) {
        self.assumed_assigned[rep_index] = 1;
    }

    pub(super) fn mark_scheduled(&mut self, generator_index: usize) {
        self.generators_scheduled[generator_index] = 1;
    }

    fn is_done(&self) -> bool {
        self.generators_scheduled.iter().all(|&flag| flag != 0)
    }
}

/// Immutable stages and circuit data shared by independent executors.
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
    outputs: Arc<Vec<Vec<Target>>>,
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
    pub(super) fn outputs(&self, generator_index: usize) -> &[Target] {
        &self.outputs[generator_index]
    }
}

/// New caller-assignment requirements and the work enabled for one stage.
#[derive(Debug)]
pub struct WitnessGenerationStage {
    newly_assumed_assigned: Vec<usize>,
    schedule: WitnessGenerationSchedule,
}

impl WitnessGenerationStage {
    pub(super) fn newly_assumed_assigned(&self) -> &[usize] {
        &self.newly_assumed_assigned
    }

    /// Read-only generator graph for this stage.
    pub fn schedule(&self) -> &WitnessGenerationSchedule {
        &self.schedule
    }
}

/// Builds successive stages while borrowing circuit dependencies from a planner.
/// Assumptions describe future availability; no generators are executed.
#[derive(Debug)]
pub struct WitnessGenerationPlanning<
    'p,
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    planner: &'p WitnessGenerationPlanner<'a, F, C, D>,
    newly_assumed_assigned: Vec<usize>,
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
            newly_assumed_assigned: Vec::new(),
            newly_produced: Vec::new(),
            state: WitnessGenerationState::new(
                circuit.prover_only.representative_map.len(),
                planner.reads().len(),
            ),
            plan: WitnessGenerationPlan {
                circuit,
                stages: Vec::new(),
                #[cfg(debug_assertions)]
                outputs: Arc::clone(planner.outputs()),
            },
        }
    }

    /// Promises a target assignment before the next stage executes.
    /// Aliases are resolved internally; already available representatives need
    /// no new promise. Assignments become assumed state when a stage is added.
    pub fn set_assumed_assigned(&mut self, target: Target) -> Result<()> {
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
        if self.state.assumed_assigned()[rep_index] == 0 {
            self.newly_assumed_assigned.push(rep_index);
        }
    }

    /// Records pending promises, schedules runnable work, and appends a stage.
    /// Generators still blocked on unavailable values remain unscheduled.
    pub fn add_stage(&mut self) {
        let mut promised = core::mem::take(&mut self.newly_assumed_assigned);
        promised.sort_unstable();
        promised.dedup();
        for &rep_index in &promised {
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
            newly_assumed_assigned: promised,
            schedule,
        });
    }

    /// All generators are accounted for and no promises await a stage.
    pub fn is_done(&self) -> bool {
        self.state.is_done() && self.newly_assumed_assigned.is_empty()
    }

    /// Returns a shared immutable plan only when all generators are accounted for.
    /// Otherwise returns this planning object so more stages can be added.
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
        fn run(&self, witness: &PartitionWitness<F>, outputs: &mut GeneratedValues<F>) -> bool {
            self.run_ready(witness, outputs).is_ok()
        }
        fn run_ready(
            &self,
            witness: &PartitionWitness<F>,
            outputs: &mut GeneratedValues<F>,
        ) -> Result<()> {
            let value = witness.get_target(self.input);
            anyhow::ensure!(value != F::ZERO, "injected generator error");
            assert!(value != F::ONE, "injected generator panic");
            outputs.set_target(self.output, value)
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
        // Isolate execution from admission: this test-only generator has exact
        // fixed access sets and deliberately returns errors or panics in any build.
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
                newly_assumed_assigned: vec![input_rep],
                schedule,
            }],
            #[cfg(debug_assertions)]
            outputs: Arc::new(vec![vec![output]]),
        });
        for bad_value in [F::ZERO, F::ONE] {
            for num_workers in [None, Some(1), Some(4)] {
                let mut executor = WitnessGenerationExecutor::new(Arc::clone(&plan));
                let mut witness = executor.new_witness();
                let mut execution = executor.bind(&mut witness)?;
                execution.set_target(input, bad_value)?;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    if let Some(workers) = num_workers {
                        // SAFETY: The test generator declares its exact fixed access sets.
                        unsafe { execution.execute_parallel(workers) }
                    } else {
                        execution.execute()
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
                    .contains("after failed execution"));
                assert!(execution.set_target(input, F::TWO).is_err());
                let mut execution = execution
                    .finish()
                    .expect_err("failed execution cannot finish");
                execution.reset();
                assert!(!execution.witness().contains(input));
                assert!(!execution.witness().contains(output));
                execution.set_target(input, F::TWO)?;
                if let Some(workers) = num_workers {
                    // SAFETY: The same fixed access contract holds after reset.
                    unsafe { execution.execute_parallel(workers)? };
                } else {
                    execution.execute()?;
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
