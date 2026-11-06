//! Execution of planned generator chains on a caller-owned witness.

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(all(not(feature = "std"), debug_assertions))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
#[cfg(feature = "std")]
use std::sync::{atomic::AtomicUsize, Arc};

#[cfg(feature = "std")]
use crossbeam_deque::Worker;

use anyhow::{anyhow, ensure, Result};

use crate::field::extension::Extendable;
#[cfg(feature = "std")]
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::generation_schedule::GenerationSchedule;
use crate::iop::generator::GeneratedValues;
use crate::iop::scheduling_state::ScheduleChainId;
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::iop::witness_generator_scheduler::WitnessGeneratorScheduler;
#[cfg(feature = "std")]
use crate::iop::worker_coordinator::WorkerCoordinator;
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// Immutable execution data, shared without retaining scheduling tables.
#[derive(Debug)]
pub(super) struct ExecutionData<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    circuit: &'a CircuitData<F, C, D>,
    identity: Arc<()>,
    #[cfg(debug_assertions)]
    outputs: Vec<Vec<Target>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ExecutionData<'a, F, C, D>
{
    pub(super) fn new(
        circuit: &'a CircuitData<F, C, D>,
        #[cfg(debug_assertions)] outputs: Vec<Vec<Target>>,
    ) -> Self {
        Self {
            circuit,
            identity: Arc::new(()),
            #[cfg(debug_assertions)]
            outputs,
        }
    }

    pub(super) fn circuit(&self) -> &'a CircuitData<F, C, D> {
        self.circuit
    }

    pub(super) fn identity(&self) -> &Arc<()> {
        &self.identity
    }

    fn run_generator(
        &self,
        generator_index: usize,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        self.run_generator_outputs(generator_index, witness, buffer)?;
        let generator = &self.circuit.prover_only.generators[generator_index].0;
        for (target, value) in buffer.target_values.drain(..) {
            witness.set_target(target, value).map_err(|error| {
                anyhow!("generator {generator_index} ({}): {error}", generator.id())
            })?;
        }
        Ok(())
    }

    #[cfg(feature = "std")]
    unsafe fn run_generator_parallel(
        &self,
        generator_index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        self.run_generator_outputs(generator_index, witness, buffer)?;
        // SAFETY: The prepared graph and run_schedule_parallel's access contract
        // guarantee the required ordering of reads and writes.
        unsafe { self.write_generator_outputs(generator_index, witness, buffer)? };
        Ok(())
    }

    #[cfg(feature = "std")]
    /// # Safety
    /// Output slots must have exclusive assignment followed by concurrent
    /// read-only comparisons, with readers ordered after assignment.
    unsafe fn write_generator_outputs(
        &self,
        generator_index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        for (target, value) in buffer.target_values.drain(..) {
            // SAFETY: The caller guarantees exclusive assignment followed by
            // concurrent read-only comparisons, with readers after assignment.
            unsafe { witness.set_target_shared(target, value) }.map_err(|error| {
                anyhow!(
                    "generator {generator_index} ({}): {error}",
                    self.circuit.prover_only.generators[generator_index].0.id()
                )
            })?;
        }
        Ok(())
    }

    fn run_generator_outputs(
        &self,
        generator_index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let generator = &self.circuit.prover_only.generators[generator_index].0;
        generator.run_ready(witness, buffer).map_err(|error| {
            anyhow!("generator {generator_index} ({}): {error}", generator.id())
        })?;
        #[cfg(debug_assertions)]
        {
            let expected = &self.outputs[generator_index];
            debug_assert_eq!(
                buffer.target_values.len(),
                expected.len(),
                "generator {generator_index} ({}) changed its output count",
                generator.id()
            );
            for (position, ((target, _), declared)) in
                buffer.target_values.iter().zip(expected).enumerate()
            {
                debug_assert_eq!(
                    target,
                    declared,
                    "generator {generator_index} ({}) changed output {position}",
                    generator.id()
                );
            }
        }
        Ok(())
    }
}

/// Reusable execution data and worker allocations for a circuit.
#[derive(Debug)]
pub struct WitnessGeneratorExecutor<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    execution: Arc<ExecutionData<'a, F, C, D>>,
    #[cfg(feature = "std")]
    scratch: GenerationScratch<F>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGeneratorExecutor<'a, F, C, D>
{
    /// Creates a reusable executor. The scheduler can be dropped afterward;
    /// execution retains only its immutable execution data.
    pub fn new(scheduler: &WitnessGeneratorScheduler<'a, F, C, D>) -> Self {
        Self {
            execution: scheduler.execution_data(),
            #[cfg(feature = "std")]
            scratch: GenerationScratch::new(),
        }
    }

    /// Creates an empty dense witness for explicit caller filling and execution.
    /// Access values through witness methods; its public `values` vector is empty.
    pub fn new_witness(circuit: &'a CircuitData<F, C, D>) -> PartitionWitness<'a, F> {
        PartitionWitness::new_dense(
            circuit.common.config.num_wires,
            circuit.common.degree(),
            &circuit.prover_only.representative_map,
        )
    }

    /// Borrows a matching witness and starts new schedule-chain progress.
    pub fn bind<'b>(
        &'b mut self,
        witness: &'b mut PartitionWitness<'a, F>,
    ) -> Result<BoundWitnessGeneratorExecutor<'b, 'a, F, C, D>> {
        let circuit = self.execution.circuit();
        ensure!(
            core::ptr::eq(
                witness.representative_map,
                circuit.prover_only.representative_map.as_slice()
            ) && witness.num_wires == circuit.common.config.num_wires
                && witness.degree == circuit.common.degree(),
            "witness does not match scheduler circuit"
        );
        Ok(BoundWitnessGeneratorExecutor {
            executor: self,
            witness,
            chain_id: None,
            next_schedule: 0,
            failed: false,
        })
    }

    /// Binds a witness and executes a single schedule at index zero.
    pub fn run_schedule(
        &mut self,
        witness: &mut PartitionWitness<'a, F>,
        schedule: &GenerationSchedule,
    ) -> Result<()> {
        self.bind(witness)?.run_schedule(schedule)
    }

    /// Binds a witness and executes a single schedule at index zero in parallel.
    ///
    /// # Safety
    /// The generator access requirements of
    /// [`BoundWitnessGeneratorExecutor::run_schedule_parallel`] must hold.
    #[cfg(feature = "std")]
    pub unsafe fn run_schedule_parallel(
        &mut self,
        witness: &mut PartitionWitness<'a, F>,
        schedule: &GenerationSchedule,
        num_workers: usize,
    ) -> Result<()> {
        // SAFETY: The caller guarantees the bound method's access requirements.
        unsafe {
            self.bind(witness)?
                .run_schedule_parallel(schedule, num_workers)
        }
    }

    #[cfg(feature = "std")]
    unsafe fn run_parallel(
        &mut self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        num_workers: usize,
    ) -> Result<()> {
        let execution = &self.execution;
        let witness = &*witness;
        let scratch = &mut self.scratch;
        scratch.reset(schedule.dependency_counts(), num_workers);
        scratch.seed_ready(schedule.initial_ready(), num_workers);
        if schedule.initial_ready().is_empty() {
            ensure!(
                schedule.generator_indices().is_empty(),
                "scheduled witness generators did not complete"
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
                                execution.run_generator_parallel(
                                    generator_index,
                                    witness,
                                    &mut worker.outputs,
                                )
                            };
                            if let Err(error) = result {
                                coordinator.fail(error);
                                return;
                            }
                            let pushed = schedule.release_successors(
                                local_index,
                                pending_dependencies,
                                |successor_local_index| {
                                    worker.ready.push(successor_local_index);
                                },
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
}

/// Borrows an executor and witness for one schedule chain.
/// Dropping or unbinding releases both borrows without clearing witness values.
#[derive(Debug)]
pub struct BoundWitnessGeneratorExecutor<
    'b,
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    executor: &'b mut WitnessGeneratorExecutor<'a, F, C, D>,
    witness: &'b mut PartitionWitness<'a, F>,
    chain_id: Option<Arc<ScheduleChainId>>,
    next_schedule: usize,
    failed: bool,
}

impl<'b, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    BoundWitnessGeneratorExecutor<'b, 'a, F, C, D>
{
    /// Read-only access to the bound witness.
    pub fn witness(&self) -> &PartitionWitness<'a, F> {
        self.witness
    }

    /// Releases the witness and executor borrows, discarding chain progress.
    pub fn unbind(self) {}

    fn begin_schedule(&mut self, schedule: &GenerationSchedule) -> Result<()> {
        ensure!(
            !self.failed,
            "witness executor cannot continue after failed execution"
        );
        ensure!(
            schedule
                .chain_id()
                .matches_scheduler(self.executor.execution.identity()),
            "schedule belongs to another scheduler"
        );
        if let Some(chain_id) = &self.chain_id {
            ensure!(
                Arc::ptr_eq(chain_id, schedule.chain_id()),
                "schedule belongs to another chain"
            );
        }
        ensure!(
            schedule.index() == self.next_schedule,
            "expected schedule {}, got {}",
            self.next_schedule,
            schedule.index()
        );
        let num_wires = self.witness.num_wires;
        let num_wire_targets = self.witness.degree * num_wires;
        for &rep_index in schedule.newly_promised() {
            let target = if rep_index < num_wire_targets {
                Target::wire(rep_index / num_wires, rep_index % num_wires)
            } else {
                Target::VirtualTarget {
                    index: rep_index - num_wire_targets,
                }
            };
            ensure!(
                self.witness.try_get_target(target).is_some(),
                "promised representative {rep_index} is not assigned"
            );
        }
        if self.chain_id.is_none() {
            self.chain_id = Some(Arc::clone(schedule.chain_id()));
        }
        self.failed = true;
        Ok(())
    }

    fn finish_schedule(&mut self, schedule: &GenerationSchedule) {
        self.next_schedule = schedule
            .index()
            .checked_add(1)
            .expect("schedule index overflow");
        self.failed = false;
    }

    /// Checks new promises and executes the next schedule serially.
    /// Validation errors permit retry; execution errors or panics invalidate
    /// this binding, whose partially populated witness remains caller-owned.
    pub fn run_schedule(&mut self, schedule: &GenerationSchedule) -> Result<()> {
        self.begin_schedule(schedule)?;
        let mut outputs = GeneratedValues::empty();
        for &generator_index in schedule.generator_indices() {
            self.executor
                .execution
                .run_generator(generator_index, self.witness, &mut outputs)?;
        }
        self.finish_schedule(schedule);
        Ok(())
    }

    /// Checks new promises and executes the next schedule with one OS thread
    /// per worker. Worker allocations are retained for subsequent schedules.
    ///
    /// # Safety
    /// Generators must declare every read, including presence queries, emit
    /// their discovered output targets/order independently of input values,
    /// and access storage only through target-based witness methods. They must
    /// not retain witness access after execution. Built-in IDs and debug output
    /// checks do not establish these guarantees for arbitrary implementations.
    #[cfg(feature = "std")]
    pub unsafe fn run_schedule_parallel(
        &mut self,
        schedule: &GenerationSchedule,
        num_workers: usize,
    ) -> Result<()> {
        ensure!(
            num_workers > 0,
            "witness generation requires at least one worker"
        );
        ensure!(
            self.witness.is_dense(),
            "parallel witness execution requires dense storage"
        );
        self.begin_schedule(schedule)?;
        // SAFETY: The caller supplies generator access guarantees. Chain and
        // promise checks establish prior publication and initial availability.
        unsafe {
            self.executor
                .run_parallel(schedule, self.witness, num_workers)
        }?;
        self.finish_schedule(schedule);
        Ok(())
    }
}

impl<'b, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessWrite<F> for BoundWitnessGeneratorExecutor<'b, 'a, F, C, D>
{
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        ensure!(
            !self.failed,
            "witness executor cannot continue after failed execution"
        );
        self.witness.set_target(target, value)
    }
}

#[cfg(feature = "std")]
/// Executor-owned allocations reset for each schedule. Witness access, failure
/// state and schedule-chain progress are held separately by the binding.
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
            worker.outputs.target_values.clear();
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
    outputs: GeneratedValues<F>,
    ready: Worker<usize>,
}

#[cfg(feature = "std")]
impl<F: Field> WorkerScratch<F> {
    fn new() -> Self {
        Self {
            outputs: GeneratedValues::empty(),
            ready: Worker::new_lifo(),
        }
    }
}
