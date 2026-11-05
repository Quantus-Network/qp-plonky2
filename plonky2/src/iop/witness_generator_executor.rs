//! Execution of planned generator chains on a caller-owned dense witness.

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
#[cfg(feature = "std")]
use crate::iop::generation_pool::GenerationPool;
use crate::iop::generation_schedule::GenerationSchedule;
use crate::iop::generator::GeneratedValues;
use crate::iop::scheduling_state::ScheduleChain;
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, WitnessWrite};
use crate::iop::witness_generator_scheduler::WitnessGeneratorScheduler;
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
        index: usize,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        self.run_generator_outputs_ready(index, witness, buffer)?;
        let generator = &self.circuit.prover_only.generators[index].0;
        for (target, value) in buffer.target_values.drain(..) {
            witness
                .set_target(target, value)
                .map_err(|error| anyhow!("generator {index} ({}): {error}", generator.id()))?;
        }
        Ok(())
    }

    #[cfg(feature = "std")]
    unsafe fn run_generator_parallel(
        &self,
        index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        self.run_generator_outputs_ready(index, witness, buffer)?;
        // SAFETY: The prepared graph and run_schedule_parallel's access contract
        // guarantee the required ordering of reads and writes.
        unsafe { self.write_generator_outputs(index, witness, buffer)? };
        Ok(())
    }

    #[cfg(feature = "std")]
    /// # Safety
    /// Output slots must have exclusive assignment followed by concurrent
    /// read-only comparisons, with readers ordered after assignment.
    unsafe fn write_generator_outputs(
        &self,
        index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        for (target, value) in buffer.target_values.drain(..) {
            // SAFETY: The caller guarantees exclusive assignment followed by
            // concurrent read-only comparisons, with readers after assignment.
            unsafe { witness.set_target_shared(target, value) }.map_err(|error| {
                anyhow!(
                    "generator {index} ({}): {error}",
                    self.circuit.prover_only.generators[index].0.id()
                )
            })?;
        }
        Ok(())
    }

    fn run_generator_outputs_ready(
        &self,
        index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let generator = &self.circuit.prover_only.generators[index].0;
        generator
            .run_ready(witness, buffer)
            .map_err(|error| anyhow!("generator {index} ({}): {error}", generator.id()))?;
        #[cfg(debug_assertions)]
        {
            let expected = &self.outputs[index];
            debug_assert_eq!(
                buffer.target_values.len(),
                expected.len(),
                "generator {index} ({}) changed its output count",
                generator.id()
            );
            for (position, ((target, _), declared)) in
                buffer.target_values.iter().zip(expected).enumerate()
            {
                debug_assert_eq!(
                    target,
                    declared,
                    "generator {index} ({}) changed output {position}",
                    generator.id()
                );
            }
        }
        Ok(())
    }
}

/// Runs a schedule chain while borrowing its witness and retaining worker buffers.
/// Input filling delegates through WitnessWrite; the witness cannot be replaced
/// or reset through this object while schedule progress is retained.
#[derive(Debug)]
pub struct WitnessGeneratorExecutor<
    'w,
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    execution: Arc<ExecutionData<'a, F, C, D>>,
    witness: &'w mut PartitionWitness<'a, F>,
    outputs: GeneratedValues<F>,
    #[cfg(feature = "std")]
    scratch: GenerationScratch<F>,
    chain: Option<Arc<ScheduleChain>>,
    next_schedule: usize,
    failed: bool,
}

impl<'w, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGeneratorExecutor<'w, 'a, F, C, D>
{
    /// Borrows a matching dense witness. The scheduler can be dropped after
    /// construction; execution keeps only its immutable execution data.
    pub fn new(
        scheduler: &WitnessGeneratorScheduler<'a, F, C, D>,
        witness: &'w mut PartitionWitness<'a, F>,
    ) -> Result<Self> {
        let execution = scheduler.execution_data();
        let circuit = execution.circuit();
        ensure!(
            core::ptr::eq(
                witness.representative_map,
                circuit.prover_only.representative_map.as_slice()
            ) && witness.num_wires == circuit.common.config.num_wires
                && witness.degree == circuit.common.degree(),
            "witness does not match scheduler circuit"
        );
        ensure!(
            witness.is_dense(),
            "witness executor requires dense storage"
        );
        Ok(Self {
            execution,
            witness,
            outputs: GeneratedValues::empty(),
            #[cfg(feature = "std")]
            scratch: GenerationScratch::new(),
            chain: None,
            next_schedule: 0,
            failed: false,
        })
    }

    /// Read-only access to the borrowed witness.
    pub fn witness(&self) -> &PartitionWitness<'a, F> {
        self.witness
    }

    fn check_schedule(&mut self, schedule: &GenerationSchedule) -> Result<()> {
        ensure!(
            !self.failed,
            "witness executor cannot continue after failed execution"
        );
        ensure!(
            schedule
                .chain()
                .matches_scheduler(self.execution.identity()),
            "schedule belongs to another scheduler"
        );
        if let Some(chain) = &self.chain {
            ensure!(
                Arc::ptr_eq(chain, schedule.chain()),
                "schedule belongs to another chain"
            );
        }
        ensure!(
            schedule.index() == self.next_schedule,
            "expected schedule {}, got {}",
            self.next_schedule,
            schedule.index()
        );
        let dense = self.witness.dense_mut();
        for &rep_index in schedule.newly_supplied() {
            ensure!(
                dense.is_set(rep_index),
                "promised representative {rep_index} is not assigned"
            );
        }
        Ok(())
    }

    fn finish_schedule(&mut self, schedule: &GenerationSchedule) {
        if self.chain.is_none() {
            self.chain = Some(Arc::clone(schedule.chain()));
        }
        self.next_schedule = schedule
            .index()
            .checked_add(1)
            .expect("schedule index overflow");
        self.failed = false;
    }

    /// Checks new promises and executes the next schedule serially.
    /// Validation errors permit retry; execution errors or panics invalidate
    /// this executor, whose partially populated witness remains caller-owned.
    pub fn run_schedule(&mut self, schedule: &GenerationSchedule) -> Result<()> {
        self.check_schedule(schedule)?;
        self.failed = true;
        self.outputs.target_values.clear();
        for &generator in schedule.order() {
            self.execution
                .run_generator(generator, self.witness, &mut self.outputs)?;
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
        self.check_schedule(schedule)?;
        self.failed = true;
        // SAFETY: The caller supplies generator access guarantees. Chain and
        // promise checks establish prior publication and initial availability.
        unsafe { self.run_parallel(schedule, num_workers) }?;
        self.finish_schedule(schedule);
        Ok(())
    }

    #[cfg(feature = "std")]
    unsafe fn run_parallel(
        &mut self,
        schedule: &GenerationSchedule,
        num_threads: usize,
    ) -> Result<()> {
        let execution = &self.execution;
        let witness = &mut *self.witness;
        let scratch = &mut self.scratch;
        scratch.reset(schedule.dependency_counts(), num_threads);
        scratch.seed_ready(schedule.initial_ready(), num_threads);
        if schedule.initial_ready().is_empty() {
            ensure!(
                schedule.order().is_empty(),
                "scheduled witness generators did not complete"
            );
            return Ok(());
        }
        let pending_dependencies = &scratch.pending_dependencies;
        let mut workers = core::mem::take(&mut scratch.workers);
        let inactive = workers.split_off(num_threads);
        let pool = GenerationPool::new(
            workers
                .iter()
                .map(|worker| worker.ready.stealer())
                .collect(),
        );
        witness.dense_mut().set_parallel(true);
        let result = catch_unwind(AssertUnwindSafe(|| {
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(num_threads);
                for (index, mut worker) in workers.into_iter().enumerate() {
                    let witness = &*witness;
                    let pool = &pool;
                    // Each deque moves with its owning worker; only its Stealer is shared.
                    let handle = std::thread::Builder::new().spawn_scoped(scope, move || {
                        let mut completed = 0;
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            let mut victim = (index + 1) % num_threads;
                            while let Some(local) = pool.next(&worker.ready, index, &mut victim) {
                                let generator = schedule.order()[local];
                                // SAFETY: The caller guarantees stable, complete access
                                // sets. Counter AcqRel publication and deque handoff
                                // order every read/comparison after the selected writer.
                                let result = unsafe {
                                    execution.run_generator_parallel(
                                        generator,
                                        witness,
                                        &mut worker.outputs,
                                    )
                                };
                                if let Err(error) = result {
                                    pool.fail(error);
                                    return;
                                }
                                let pushed = schedule.release_successors(
                                    local,
                                    pending_dependencies,
                                    |consumer| {
                                        worker.ready.push(consumer);
                                    },
                                );
                                if pushed {
                                    pool.notify_work();
                                }
                                completed += 1;
                            }
                        }));
                        if let Err(panic) = result {
                            pool.abort();
                            resume_unwind(panic);
                        }
                        (worker, completed)
                    });
                    match handle {
                        Ok(handle) => handles.push(handle),
                        Err(error) => {
                            pool.fail(anyhow!("could not spawn witness worker {index}: {error}"));
                            break;
                        }
                    }
                }
                let mut workers = Vec::with_capacity(num_threads);
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
            })
        }));
        witness.dense_mut().set_parallel(false);
        let (workers, completed) = match result {
            Ok(result) => result,
            Err(panic) => resume_unwind(panic),
        };
        scratch.workers = workers;
        scratch.workers.extend(inactive);
        pool.finish()?;
        ensure!(
            completed == schedule.order().len(),
            "completed {completed} of {} scheduled witness generators",
            schedule.order().len()
        );
        Ok(())
    }
}

impl<'w, 'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessWrite<F> for WitnessGeneratorExecutor<'w, 'a, F, C, D>
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
/// Executor-owned allocations reset for each schedule. The witness borrow,
/// failure state and schedule-chain progress are held separately by the executor.
#[derive(Debug)]
struct GenerationScratch<F: Field> {
    pending_dependencies: Vec<AtomicUsize>,
    workers: Vec<WorkerScratch<F>>,
}

#[cfg(feature = "std")]
impl<F: Field> Default for GenerationScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl<F: Field> GenerationScratch<F> {
    fn new() -> Self {
        Self {
            pending_dependencies: Vec::new(),
            workers: Vec::new(),
        }
    }

    fn reset(&mut self, counts: &[usize], workers: usize) {
        assert!(
            workers > 0,
            "witness generation requires at least one worker"
        );
        self.pending_dependencies
            .resize_with(counts.len(), || AtomicUsize::new(0));
        for (pending_dependencies, &count) in self.pending_dependencies.iter_mut().zip(counts) {
            *pending_dependencies.get_mut() = count;
        }
        if self.workers.len() < workers {
            self.workers.resize_with(workers, WorkerScratch::new);
        }
        for worker in &mut self.workers {
            worker.outputs.target_values.clear();
            while worker.ready.pop().is_some() {}
        }
    }

    /// Seeds balanced worker slices from roots already in circuit-builder order.
    fn seed_ready(&mut self, initial_ready: &[usize], workers: usize) {
        debug_assert!(self.workers[..workers]
            .iter()
            .all(|worker| worker.ready.is_empty()));
        // Owners start at the slice's beginning; thieves take from its end.
        let per_worker = initial_ready.len() / workers;
        let extra = initial_ready.len() % workers;
        let mut start = 0;
        for (index, worker) in self.workers.iter().take(workers).enumerate() {
            let end = start + per_worker + usize::from(index < extra);
            for &generator in initial_ready[start..end].iter().rev() {
                worker.ready.push(generator);
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
