//! Serial and parallel execution of immutable witness-generation plans.

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use crossbeam_deque::Worker;
#[cfg(feature = "std")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
#[cfg(feature = "std")]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use anyhow::{anyhow, ensure, Result};

use crate::field::extension::Extendable;
#[cfg(feature = "std")]
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::generator::GeneratedValues;
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::iop::witness_generation_plan::WitnessGenerationPlan;
use crate::iop::witness_generation_planner::WitnessGenerationPlanner;
#[cfg(feature = "std")]
use crate::iop::witness_generation_schedule::WitnessGenerationSchedule;
#[cfg(feature = "std")]
use crate::iop::worker_coordinator::WorkerCoordinator;
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// Executes a witness generation plan with reusable worker allocations.
#[derive(Debug)]
pub struct WitnessGenerationExecutor<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    plan: Arc<WitnessGenerationPlan<'a, F, C, D>>,
    #[cfg(feature = "std")]
    scratch: GenerationScratch<F>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGenerationExecutor<'a, F, C, D>
{
    pub fn new(plan: Arc<WitnessGenerationPlan<'a, F, C, D>>) -> Self {
        Self {
            plan,
            #[cfg(feature = "std")]
            scratch: GenerationScratch::new(),
        }
    }

    /// Creates an executor with a complete single-stage plan for the supplied input targets.
    pub fn new_with_inputs(
        circuit: &'a CircuitData<F, C, D>,
        input_targets: impl IntoIterator<Item = Target>,
    ) -> Result<Self> {
        let planner = WitnessGenerationPlanner::new(circuit)?;
        let plan = planner.input_plan(input_targets)?;
        Ok(Self::new(plan))
    }

    pub fn plan(&self) -> &Arc<WitnessGenerationPlan<'a, F, C, D>> {
        &self.plan
    }

    /// Creates a matching empty dense witness. Its public `values` vector is
    /// empty; access assignments through witness methods.
    pub fn new_witness(&self) -> PartitionWitness<'a, F> {
        let circuit = self.plan.circuit();
        PartitionWitness::new_dense(
            circuit.common.config.num_wires,
            circuit.common.degree(),
            &circuit.prover_only.representative_map,
        )
    }

    /// Borrows the executor and witness for execution starting at the first stage.
    pub fn bind<'b>(
        &'b mut self,
        witness: &'b mut PartitionWitness<'a, F>,
    ) -> Result<WitnessGenerationExecution<'b, 'a, F, C, D>> {
        let circuit = self.plan.circuit();
        ensure!(
            core::ptr::eq(
                witness.representative_map,
                circuit.prover_only.representative_map.as_slice()
            ) && witness.num_wires == circuit.common.config.num_wires
                && witness.degree == circuit.common.degree(),
            "witness does not match plan circuit"
        );
        Ok(WitnessGenerationExecution {
            executor: self,
            witness,
            next_stage: 0,
            failed: false,
        })
    }

    /// Executes the entire plan with inputs supplied before this call.
    pub fn execute(&mut self, witness: &mut PartitionWitness<'a, F>) -> Result<()> {
        let mut execution = self.bind(witness)?;
        while !execution.is_done() {
            execution.execute()?;
        }
        Ok(())
    }

    /// Executes the entire plan with one OS thread per worker in each stage.
    ///
    /// # Safety
    /// The generator access requirements of
    /// [`WitnessGenerationExecution::execute_parallel`] must hold.
    #[cfg(feature = "std")]
    pub unsafe fn execute_parallel(
        &mut self,
        witness: &mut PartitionWitness<'a, F>,
        num_workers: usize,
    ) -> Result<()> {
        ensure!(
            num_workers > 0,
            "witness generation requires at least one worker"
        );
        ensure!(
            witness.is_dense(),
            "parallel witness execution requires dense storage"
        );
        let mut execution = self.bind(witness)?;
        while !execution.is_done() {
            // SAFETY: The caller supplies the bound execution's access guarantees.
            unsafe { execution.execute_parallel(num_workers)? };
        }
        Ok(())
    }

    #[cfg(feature = "std")]
    unsafe fn run_parallel(
        &mut self,
        schedule: &WitnessGenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        num_workers: usize,
    ) -> Result<()> {
        let plan = &self.plan;
        let witness = &*witness;
        let scratch = &mut self.scratch;
        scratch.reset(schedule.dependency_counts(), num_workers);
        scratch.seed_ready(schedule.initial_ready(), num_workers);
        if schedule.initial_ready().is_empty() {
            ensure!(
                schedule.generator_indices().is_empty(),
                "witness generation schedule has no ready generators"
            );
            return Ok(());
        }
        let pending_dependencies = &scratch.pending_dependencies;
        let mut workers = core::mem::take(&mut scratch.workers);
        let inactive = workers.split_off(num_workers);
        let coordinator = WorkerCoordinator::new(
            workers
                .iter()
                .map(|worker| worker.ready.stealer())
                .collect(),
        );
        let (workers, completed) = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_workers);
            for (worker_index, mut worker) in workers.into_iter().enumerate() {
                let witness = &*witness;
                let coordinator = &coordinator;
                // Each deque moves with its owning worker; only its Stealer is shared.
                let handle = std::thread::Builder::new().spawn_scoped(scope, move || {
                    let mut completed = 0;
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let mut next_victim_index = (worker_index + 1) % num_workers;
                        while let Some(local_index) =
                            coordinator.next(&worker.ready, worker_index, &mut next_victim_index)
                        {
                            let generator_index = schedule.generator_indices()[local_index];
                            // SAFETY: The caller guarantees stable, complete access
                            // sets. Counter AcqRel publication and deque handoff
                            // order every read/comparison after the selected writer.
                            let result = unsafe {
                                Self::run_generator_parallel(
                                    plan,
                                    generator_index,
                                    witness,
                                    &mut worker.output_buffer,
                                )
                            };
                            if let Err(error) = result {
                                coordinator.fail(error);
                                return;
                            }
                            let pushed = Self::release_successors(
                                schedule,
                                local_index,
                                pending_dependencies,
                                &worker.ready,
                            );
                            if pushed {
                                coordinator.notify_work();
                            }
                            completed += 1;
                        }
                    }));
                    if let Err(panic) = result {
                        coordinator.abort();
                        resume_unwind(panic);
                    }
                    (worker, completed)
                });
                match handle {
                    Ok(handle) => handles.push(handle),
                    Err(error) => {
                        coordinator.fail(anyhow!(
                            "could not spawn witness worker {worker_index}: {error}"
                        ));
                        break;
                    }
                }
            }
            let mut workers = Vec::with_capacity(num_workers);
            let mut completed = 0;
            let mut panic = None;
            for handle in handles {
                match handle.join() {
                    Ok((worker, count)) => {
                        workers.push(worker);
                        completed += count;
                    }
                    Err(payload) => {
                        if panic.is_none() {
                            panic = Some(payload);
                        }
                    }
                }
            }
            if let Some(panic) = panic {
                resume_unwind(panic);
            }
            (workers, completed)
        });
        scratch.workers = workers;
        scratch.workers.extend(inactive);
        coordinator.finish()?;
        ensure!(
            completed == schedule.generator_indices().len(),
            "completed {completed} of {} scheduled witness generators",
            schedule.generator_indices().len()
        );
        Ok(())
    }

    #[cfg(feature = "std")]
    fn release_successors(
        schedule: &WitnessGenerationSchedule,
        local_index: usize,
        pending_dependencies: &[AtomicUsize],
        local_queue: &Worker<usize>,
    ) -> bool {
        let mut pushed = false;
        for &successor_local_index in schedule.successors(local_index) {
            let previous =
                pending_dependencies[successor_local_index].fetch_sub(1, Ordering::AcqRel);
            assert!(
                previous > 0,
                "generator {} dependency count underflow",
                schedule.generator_indices()[successor_local_index]
            );
            if previous == 1 {
                local_queue.push(successor_local_index);
                pushed = true;
            }
        }
        pushed
    }

    fn run_generator(
        plan: &WitnessGenerationPlan<'a, F, C, D>,
        generator_index: usize,
        witness: &mut PartitionWitness<'_, F>,
        output_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        Self::run_generator_outputs(plan, generator_index, witness, output_buffer)?;
        let generator = &plan.circuit().prover_only.generators[generator_index].0;
        for (target, value) in output_buffer.target_values.drain(..) {
            witness.set_target(target, value).map_err(|error| {
                anyhow!("generator {generator_index} ({}): {error}", generator.id())
            })?;
        }
        Ok(())
    }

    #[cfg(feature = "std")]
    unsafe fn run_generator_parallel(
        plan: &WitnessGenerationPlan<'a, F, C, D>,
        generator_index: usize,
        witness: &PartitionWitness<'_, F>,
        output_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        Self::run_generator_outputs(plan, generator_index, witness, output_buffer)?;
        // SAFETY: The caller guarantees complete read/write declarations.
        // Dependencies order reads and comparisons after initial assignment.
        unsafe { Self::write_generator_outputs(plan, generator_index, witness, output_buffer)? };
        Ok(())
    }

    #[cfg(feature = "std")]
    /// # Safety
    /// Each output must satisfy the assignment and publication requirements
    /// of [`PartitionWitness::set_target_shared`].
    unsafe fn write_generator_outputs(
        plan: &WitnessGenerationPlan<'a, F, C, D>,
        generator_index: usize,
        witness: &PartitionWitness<'_, F>,
        output_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        for (target, value) in output_buffer.target_values.drain(..) {
            // SAFETY: The caller guarantees exclusive assignment followed by
            // concurrent read-only comparisons, with readers after assignment.
            unsafe { witness.set_target_shared(target, value) }.map_err(|error| {
                anyhow!(
                    "generator {generator_index} ({}): {error}",
                    plan.circuit().prover_only.generators[generator_index]
                        .0
                        .id()
                )
            })?;
        }
        Ok(())
    }

    fn run_generator_outputs(
        plan: &WitnessGenerationPlan<'a, F, C, D>,
        generator_index: usize,
        witness: &PartitionWitness<'_, F>,
        output_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let generator = &plan.circuit().prover_only.generators[generator_index].0;
        generator
            .run_ready(witness, output_buffer)
            .map_err(|error| {
                anyhow!("generator {generator_index} ({}): {error}", generator.id())
            })?;
        #[cfg(debug_assertions)]
        {
            let expected_output_targets = plan.expected_output_targets(generator_index);
            debug_assert_eq!(
                output_buffer.target_values.len(),
                expected_output_targets.len(),
                "generator {generator_index} ({}) emitted an unexpected number of outputs",
                generator.id()
            );
            for (position, ((target, _), declared)) in output_buffer
                .target_values
                .iter()
                .zip(expected_output_targets)
                .enumerate()
            {
                debug_assert_eq!(
                    target,
                    declared,
                    "generator {generator_index} ({}) emitted an unexpected output target at position {position}",
                    generator.id()
                );
            }
        }
        Ok(())
    }
}

/// Executes successive stages while borrowing an executor and one witness.
#[derive(Debug)]
pub struct WitnessGenerationExecution<
    'b,
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    executor: &'b mut WitnessGenerationExecutor<'a, F, C, D>,
    witness: &'b mut PartitionWitness<'a, F>,
    next_stage: usize,
    failed: bool,
}

impl<'b, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGenerationExecution<'b, 'a, F, C, D>
{
    pub fn witness(&self) -> &PartitionWitness<'a, F> {
        self.witness
    }

    /// Releases the executor and witness borrows, preserving witness assignments.
    pub fn unbind(self) {}

    /// Releases the executor and witness borrows after successful completion.
    /// Returns the execution when further work or recovery is required.
    pub fn finish(self) -> core::result::Result<(), Self> {
        if self.is_done() {
            Ok(())
        } else {
            Err(self)
        }
    }

    pub fn is_done(&self) -> bool {
        !self.failed && self.next_stage == self.executor.plan.num_stages()
    }

    /// Clears witness assignments and restarts at the first stage, retaining
    /// the executor's worker allocations. Also permits recovery after failure.
    pub fn reset(&mut self) {
        self.witness.clear();
        self.next_stage = 0;
        self.failed = false;
    }

    fn begin_stage(&mut self) -> Result<()> {
        ensure!(
            !self.failed,
            "witness execution failed; reset before continuing"
        );
        let stage = self
            .executor
            .plan
            .stages()
            .get(self.next_stage)
            .ok_or_else(|| anyhow!("all witness generation stages have completed"))?;
        let num_wires = self.witness.num_wires;
        let num_wire_targets = self.witness.degree * num_wires;
        for &rep_index in stage.promised_rep_indices() {
            let target = if rep_index < num_wire_targets {
                Target::wire(rep_index / num_wires, rep_index % num_wires)
            } else {
                Target::VirtualTarget {
                    index: rep_index - num_wire_targets,
                }
            };
            ensure!(
                self.witness.try_get_target(target).is_some(),
                "target {target:?} is not assigned before stage {}",
                self.next_stage
            );
        }
        self.failed = true;
        Ok(())
    }

    fn finish_stage(&mut self) {
        self.next_stage += 1;
        self.failed = false;
    }

    /// Executes the next stage serially. Validation errors permit retry;
    /// execution errors or panics require reset or unbinding.
    pub fn execute(&mut self) -> Result<()> {
        self.begin_stage()?;
        let plan = &self.executor.plan;
        let schedule = plan.stages()[self.next_stage].schedule();
        let mut output_buffer = GeneratedValues::empty();
        for &generator_index in schedule.generator_indices() {
            WitnessGenerationExecutor::run_generator(
                plan,
                generator_index,
                self.witness,
                &mut output_buffer,
            )?;
        }
        self.finish_stage();
        Ok(())
    }

    /// Executes the next stage in parallel, retaining worker allocations.
    ///
    /// # Safety
    /// The caller must ensure that every generator:
    /// - declares every target it may read, including checks whether a target is assigned;
    /// - emits the output target sequence discovered during planning on every
    ///   successful call, independently of input values;
    /// - accesses the witness through target-based methods;
    /// - completes all witness access before execution returns.
    #[cfg(feature = "std")]
    pub unsafe fn execute_parallel(&mut self, num_workers: usize) -> Result<()> {
        ensure!(
            num_workers > 0,
            "witness generation requires at least one worker"
        );
        ensure!(
            self.witness.is_dense(),
            "parallel witness execution requires dense storage"
        );
        self.begin_stage()?;
        let plan = Arc::clone(&self.executor.plan);
        let schedule = plan.stages()[self.next_stage].schedule();
        // SAFETY: The caller supplies generator access guarantees. Prior stages
        // succeeded synchronously and this stage's initial promises were checked.
        unsafe {
            self.executor
                .run_parallel(schedule, self.witness, num_workers)?
        };
        self.finish_stage();
        Ok(())
    }
}

impl<'b, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessWrite<F> for WitnessGenerationExecution<'b, 'a, F, C, D>
{
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        ensure!(
            !self.failed,
            "witness execution failed; reset before continuing"
        );
        self.witness.set_target(target, value)
    }
}

#[cfg(feature = "std")]
/// Reusable dependency counters, work queues and generator output buffers.
#[derive(Debug)]
struct GenerationScratch<F: Field> {
    pending_dependencies: Vec<AtomicUsize>,
    workers: Vec<WorkerScratch<F>>,
}

#[cfg(feature = "std")]
impl<F: Field> GenerationScratch<F> {
    fn new() -> Self {
        Self {
            pending_dependencies: Vec::new(),
            workers: Vec::new(),
        }
    }

    fn reset(&mut self, dependency_counts: &[usize], num_workers: usize) {
        assert!(
            num_workers > 0,
            "witness generation requires at least one worker"
        );
        self.pending_dependencies
            .resize_with(dependency_counts.len(), || AtomicUsize::new(0));
        for (pending_dependencies, &count) in
            self.pending_dependencies.iter_mut().zip(dependency_counts)
        {
            *pending_dependencies.get_mut() = count;
        }
        if self.workers.len() < num_workers {
            self.workers.resize_with(num_workers, WorkerScratch::new);
        }
        for worker in &mut self.workers {
            worker.output_buffer.target_values.clear();
            while worker.ready.pop().is_some() {}
        }
    }

    /// Seeds balanced worker slices from roots already in circuit-builder order.
    fn seed_ready(&mut self, initial_ready: &[usize], num_workers: usize) {
        debug_assert!(self.workers[..num_workers]
            .iter()
            .all(|worker| worker.ready.is_empty()));
        // Owners start at the slice's beginning; thieves take from its end.
        let per_worker = initial_ready.len() / num_workers;
        let extra = initial_ready.len() % num_workers;
        let mut start = 0;
        for (worker_index, worker) in self.workers.iter().take(num_workers).enumerate() {
            let end = start + per_worker + usize::from(worker_index < extra);
            for &local_index in initial_ready[start..end].iter().rev() {
                worker.ready.push(local_index);
            }
            start = end;
        }
    }
}

#[cfg(feature = "std")]
#[derive(Debug)]
struct WorkerScratch<F: Field> {
    output_buffer: GeneratedValues<F>,
    ready: Worker<usize>,
}

#[cfg(feature = "std")]
impl<F: Field> WorkerScratch<F> {
    fn new() -> Self {
        Self {
            output_buffer: GeneratedValues::empty(),
            ready: Worker::new_lifo(),
        }
    }
}
