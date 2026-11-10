//! Assumed generation state, stage construction, and immutable execution plans.

#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::target::Target;
use crate::iop::witness_generation_planner::WitnessGenerationPlanner;
use crate::iop::witness_generation_schedule::{
    make_schedule, WitnessGenerationSchedule, WitnessGenerationState,
};
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::{GenericConfig, GenericHashOut};

/// An immutable sequence of witness generation stages for a circuit.
pub struct WitnessGenerationPlan {
    circuit_fingerprint: CircuitFingerprint,
    stages: Vec<WitnessGenerationStage>,
    #[cfg(debug_assertions)]
    expected_output_targets: Arc<Vec<Vec<Target>>>,
}

impl WitnessGenerationPlan {
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Read-only stages in execution order.
    pub fn stages(&self) -> &[WitnessGenerationStage] {
        &self.stages
    }

    pub(super) fn matches_circuit<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &self,
        circuit: &CircuitData<F, C, D>,
    ) -> bool {
        self.circuit_fingerprint == CircuitFingerprint::from_circuit(circuit)
    }

    #[cfg(debug_assertions)]
    pub(super) fn expected_output_targets(&self, generator_index: usize) -> &[Target] {
        &self.expected_output_targets[generator_index]
    }
}

impl core::fmt::Debug for WitnessGenerationPlan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WitnessGenerationPlan")
            .field("num_stages", &self.stages.len())
            .field(
                "num_generators",
                &self
                    .stages
                    .iter()
                    .map(|stage| stage.schedule.generator_indices().len())
                    .sum::<usize>(),
            )
            .finish_non_exhaustive()
    }
}

#[derive(PartialEq, Eq)]
struct CircuitFingerprint {
    circuit_digest: Vec<u8>,
    num_generators: usize,
    representative_map_len: usize,
    num_wires: usize,
    degree: usize,
    extension_degree: usize,
}

impl CircuitFingerprint {
    fn from_circuit<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        circuit: &CircuitData<F, C, D>,
    ) -> Self {
        Self {
            circuit_digest: circuit.verifier_only.circuit_digest.to_bytes(),
            num_generators: circuit.prover_only.generators.len(),
            representative_map_len: circuit.prover_only.representative_map.len(),
            num_wires: circuit.common.config.num_wires,
            degree: circuit.common.degree(),
            extension_degree: D,
        }
    }
}

/// Required assignments and the generator schedule for one stage.
pub struct WitnessGenerationStage {
    promised_rep_indices: Vec<usize>,
    schedule: WitnessGenerationSchedule,
}

impl WitnessGenerationStage {
    /// Read-only generator graph for this stage.
    pub fn schedule(&self) -> &WitnessGenerationSchedule {
        &self.schedule
    }

    pub(super) fn promised_rep_indices(&self) -> &[usize] {
        &self.promised_rep_indices
    }
}

impl core::fmt::Debug for WitnessGenerationStage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WitnessGenerationStage")
            .field("num_promised_assignments", &self.promised_rep_indices.len())
            .field("schedule", &self.schedule)
            .finish_non_exhaustive()
    }
}

/// Builds stages from assignment assumptions and generator dependencies.
pub struct WitnessGenerationPlanning<
    'p,
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    planner: &'p WitnessGenerationPlanner<'a, F, C, D>,
    promised_rep_indices: Vec<usize>,
    state: WitnessGenerationState,
    plan: WitnessGenerationPlan,
}

impl<'p, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGenerationPlanning<'p, 'a, F, C, D>
{
    pub(super) fn new(planner: &'p WitnessGenerationPlanner<'a, F, C, D>) -> Self {
        let circuit = planner.circuit();
        Self {
            planner,
            promised_rep_indices: Vec::new(),
            state: WitnessGenerationState::new(
                circuit.prover_only.representative_map.len(),
                planner.dependencies().reads().len(),
            ),
            plan: WitnessGenerationPlan {
                circuit_fingerprint: CircuitFingerprint::from_circuit(circuit),
                stages: Vec::new(),
                #[cfg(debug_assertions)]
                expected_output_targets: Arc::clone(planner.expected_output_targets()),
            },
        }
    }

    /// Records a target assignment required before the next stage executes.
    /// Targets connected by copy constraints share one assignment requirement.
    pub fn assume_assigned(&mut self, target: Target) -> Result<()> {
        let rep_index = self.planner.representative_index(target)?;
        self.assume_representative(rep_index);
        Ok(())
    }

    /// Schedules generators enabled by the recorded assignments and appends a stage.
    pub fn add_stage(&mut self) {
        let mut promised_rep_indices = core::mem::take(&mut self.promised_rep_indices);
        promised_rep_indices.sort_unstable();
        promised_rep_indices.dedup();
        for &rep_index in &promised_rep_indices {
            self.state.mark_assigned(rep_index);
        }
        let (schedule, newly_produced) = make_schedule(self.planner.dependencies(), &self.state);
        for &generator_index in schedule.generator_indices() {
            self.state.mark_scheduled(generator_index);
        }
        for rep_index in newly_produced {
            self.state.mark_assigned(rep_index);
        }
        self.plan.stages.push(WitnessGenerationStage {
            promised_rep_indices,
            schedule,
        });
    }

    /// Returns the immutable plan when all generators are scheduled and all
    /// assignment requirements belong to stages. Returns the planning object
    /// when more stages are required.
    pub fn finish(self) -> core::result::Result<Arc<WitnessGenerationPlan>, Self> {
        if self.is_done() {
            Ok(Arc::new(self.plan))
        } else {
            Err(self)
        }
    }

    /// Returns whether all generators are scheduled and all assignment requirements belong to stages.
    pub fn is_done(&self) -> bool {
        self.state.is_done() && self.promised_rep_indices.is_empty()
    }

    fn assume_representative(&mut self, rep_index: usize) {
        if self.state.assumed_assigned_flags()[rep_index] == 0 {
            self.promised_rep_indices.push(rep_index);
        }
    }
}

impl<'p, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    core::fmt::Debug for WitnessGenerationPlanning<'p, 'a, F, C, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WitnessGenerationPlanning")
            .field("num_stages", &self.plan.num_stages())
            .field("num_pending_assignments", &self.promised_rep_indices.len())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::field::types::Field;
    use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
    use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
    use crate::iop::witness_generation_executor::{
        WitnessGenerationExecution, WitnessGenerationExecutor,
    };
    use crate::iop::witness_generation_schedule::GeneratorDependencies;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
    use crate::util::serialization::{Buffer, IoError, IoResult};
    use crate::{C, D, F};

    #[cfg(feature = "parallel")]
    const EXECUTION_MODES: &[Option<usize>] = &[None, Some(1), Some(4)];
    #[cfg(not(feature = "parallel"))]
    const EXECUTION_MODES: &[Option<usize>] = &[None];

    #[derive(Debug)]
    struct FallibleGenerator {
        input: Target,
        output: Target,
    }

    impl SimpleGenerator<F, D> for FallibleGenerator {
        fn id(&self) -> String {
            "FallibleTestGenerator".to_owned()
        }
        fn dependencies(&self) -> Vec<Target> {
            vec![self.input]
        }
        fn run_once(
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

    fn execute_stage(
        execution: &mut WitnessGenerationExecution<'_, '_, F, C, D>,
        num_workers: Option<usize>,
    ) -> Result<()> {
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
    }

    #[test]
    fn execution_recovers_from_errors_and_panics() -> Result<()> {
        let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_config());
        let input = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        let mut data = builder.build::<C>();
        // Exercise errors, panics and recovery with fixed read and output targets.
        data.prover_only.generators = vec![WitnessGeneratorRef::new(
            FallibleGenerator { input, output }.adapter(),
        )];
        let rep = |target: Target| {
            data.prover_only.representative_map
                [target.index(data.common.config.num_wires, data.common.degree())]
        };
        let input_rep = rep(input);
        let output_rep = rep(output);
        let mut state = WitnessGenerationState::new(data.prover_only.representative_map.len(), 1);
        state.mark_assigned(input_rep);
        let dependencies = GeneratorDependencies::new(
            vec![vec![input_rep]],
            vec![vec![output_rep]],
            data.prover_only.representative_map.len(),
        )?;
        let (schedule, _) = make_schedule(&dependencies, &state);
        let plan = Arc::new(WitnessGenerationPlan {
            circuit_fingerprint: CircuitFingerprint::from_circuit(&data),
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
                let mut witness = executor.new_witness(&data);
                let mut execution = executor.bind(&data, &mut witness)?;
                execution.set_target(input, bad_value)?;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    execute_stage(&mut execution, num_workers)
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
                execute_stage(&mut execution, num_workers)?;
                assert_eq!(execution.witness().get_target(output), F::TWO);
                execution.finish().unwrap();
                let mut next = executor.new_witness(&data);
                next.set_target(input, F::from_canonical_u64(3))?;
                executor.execute(&data, &mut next)?;
                assert_eq!(next.get_target(output), F::from_canonical_u64(3));
            }
        }
        Ok(())
    }
}
