#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::{anyhow, ensure, Result};
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
#[cfg(feature = "std")]
use crate::iop::generation_pool::GenerationPool;
use crate::iop::generation_schedule::{make_schedule, GenerationSchedule};
#[cfg(feature = "std")]
use crate::iop::generation_scratch::GenerationScratch;
use crate::iop::generator::GeneratedValues;
use crate::iop::generator_outputs::discover_outputs;
use crate::iop::scheduling_state::SchedulingState;
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, WitnessWrite};
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// Discovers built-in generator access and constructs caller-directed schedules.
/// Output targets and emission order must be independent of input values.
/// Built-ins are recognized by ID; custom implementations reusing those IDs
/// are unsupported. Debug builds check emitted targets against discovery.
#[derive(Debug)]
pub struct WitnessGeneratorScheduler<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    circuit: &'a CircuitData<F, C, D>,
    reads: Vec<Vec<usize>>,
    writes: Vec<Vec<usize>>,
    consumers: HashMap<usize, Vec<usize>>,
    #[cfg(debug_assertions)]
    outputs: Vec<Vec<Target>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    WitnessGeneratorScheduler<'a, F, C, D>
{
    /// Discovers representative read/write indices, without selecting producers
    /// or assuming availability of any input, constant or random value.
    pub fn new(circuit: &'a CircuitData<F, C, D>) -> Result<Self> {
        let common = &circuit.common;
        let prover = &circuit.prover_only;
        let generators = &prover.generators;
        let resolve = |target: Target| -> Result<usize> {
            let index = target.index(common.config.num_wires, common.degree());
            let rep_index = *prover
                .representative_map
                .get(index)
                .ok_or_else(|| anyhow!("target {target:?} is outside the witness"))?;
            ensure!(
                rep_index < prover.representative_map.len(),
                "invalid representative index {rep_index}"
            );
            Ok(rep_index)
        };
        let mut reads = Vec::with_capacity(generators.len());
        let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
        for (index, generator) in generators.iter().enumerate() {
            let targets = generator.0.reads().ok_or_else(|| {
                anyhow!(
                    "generator {index} ({}) does not declare its reads",
                    generator.0.id()
                )
            })?;
            let mut representatives = targets
                .into_iter()
                .map(&resolve)
                .collect::<Result<Vec<_>>>()?;
            representatives.sort_unstable();
            representatives.dedup();
            for &rep_index in &representatives {
                consumers.entry(rep_index).or_default().push(index);
            }
            reads.push(representatives);
        }
        let mut probe = PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &prover.representative_map,
        );
        let mut writes = Vec::with_capacity(generators.len());
        #[cfg(debug_assertions)]
        let mut outputs = Vec::with_capacity(generators.len());
        for (index, generator) in generators.iter().enumerate() {
            let discovered = discover_outputs(generator.0.as_ref(), common, &mut probe)
                .map_err(|error| anyhow!("generator {index} ({}): {error}", generator.0.id()))?;
            let mut representatives = discovered
                .iter()
                .copied()
                .map(&resolve)
                .collect::<Result<Vec<_>>>()?;
            representatives.sort_unstable();
            representatives.dedup();
            writes.push(representatives);
            #[cfg(debug_assertions)]
            outputs.push(discovered);
        }
        Ok(Self {
            circuit,
            reads,
            writes,
            consumers,
            #[cfg(debug_assertions)]
            outputs,
        })
    }

    /// Distinct representative indices read by a circuit generator.
    pub fn reads(&self, generator: usize) -> &[usize] {
        &self.reads[generator]
    }

    /// Distinct representative indices written by a circuit generator.
    pub fn writes(&self, generator: usize) -> &[usize] {
        &self.writes[generator]
    }

    /// Creates empty planning state; it does not allocate or populate a witness.
    pub fn new_state(&self) -> SchedulingState {
        SchedulingState::new(
            self.circuit.prover_only.representative_map.len(),
            self.reads.len(),
        )
    }

    /// Appends a runnable depth-first closure to the supplied planned state.
    /// Nothing is executed. Blocked generators remain unscheduled; an empty
    /// schedule is valid. All nonzero state bytes mean assumed available.
    pub fn make_schedule(&self, state: &mut SchedulingState) -> Result<GenerationSchedule> {
        ensure!(
            state.classes_set().len() == self.circuit.prover_only.representative_map.len(),
            "scheduling state representative count does not match circuit"
        );
        make_schedule(&self.reads, &self.writes, &self.consumers, state)
    }

    /// Builds a one-shot schedule assuming the flagged representatives supplied.
    /// Returns an error if any generator remains blocked. Planning state is
    /// local and discarded; caller buffers are never modified.
    pub fn make_complete_schedule(&self, classes_set: &[u8]) -> Result<GenerationSchedule> {
        ensure!(
            classes_set.len() == self.circuit.prover_only.representative_map.len(),
            "supplied representative count does not match circuit"
        );
        let mut state = self.new_state();
        state.classes_set_mut().copy_from_slice(classes_set);
        let schedule = self.make_schedule(&mut state)?;
        ensure!(
            state.is_fully_scheduled(),
            "{} generators could not be scheduled with the supplied values",
            state.remaining_generators()
        );
        Ok(schedule)
    }

    /// Creates an empty dense witness for explicit caller filling and execution.
    /// Access values through witness methods; its public `values` vector is empty.
    pub fn new_witness(&self) -> PartitionWitness<'a, F> {
        let common = &self.circuit.common;
        PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &self.circuit.prover_only.representative_map,
        )
    }

    fn check_witness(&self, witness: &PartitionWitness<'_, F>) -> Result<()> {
        ensure!(
            core::ptr::eq(
                witness.representative_map,
                self.circuit.prover_only.representative_map.as_slice()
            ) && witness.num_wires == self.circuit.common.config.num_wires
                && witness.degree == self.circuit.common.degree(),
            "witness does not match the scheduler circuit"
        );
        Ok(())
    }

    /// Executes every generator in a schedule serially. Values promised during
    /// scheduling must already be present; unassigned reads panic. Generator
    /// errors and conflicting assignments are returned to the caller.
    pub fn run_schedule(
        &self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        self.check_witness(witness)?;
        buffer.target_values.clear();
        for &generator in schedule.order() {
            self.run_generator(generator, witness, buffer)?;
        }
        Ok(())
    }

    /// Executes every scheduled generator with shared dense storage.
    /// Worker errors return after joining; worker panics propagate. Missing
    /// promised values or unfinished scheduled work are execution failures.
    ///
    /// # Safety
    /// The schedule must come from this scheduler, and earlier schedules and
    /// promised assignments must have completed successfully on this witness.
    /// Every generator must declare every read (including presence queries),
    /// emit its discovered targets/order independently of input values, and
    /// access witness storage only through its target-based methods. Access
    /// must not be retained beyond execution. Built-in IDs and debug output
    /// checks do not establish these guarantees for arbitrary implementations.
    #[cfg(feature = "std")]
    pub unsafe fn run_schedule_parallel(
        &self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        scratch: &mut GenerationScratch<F>,
        num_threads: usize,
    ) -> Result<()> {
        self.check_witness(witness)?;
        ensure!(
            num_threads > 0,
            "witness generation requires at least one worker"
        );
        scratch.reset(
            schedule.dependency_counts(),
            schedule.initial_ready(),
            num_threads,
        );
        scratch.distribute_ready(schedule.order());
        if scratch.workers[..num_threads]
            .iter()
            .all(|worker| worker.ready.is_empty())
        {
            ensure!(
                scratch.is_complete(),
                "scheduled witness generators did not complete"
            );
            return Ok(());
        }
        scratch.failed = true;
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
                                    self.run_generator_parallel(
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
        scratch.remaining = scratch
            .remaining
            .checked_sub(completed)
            .expect("witness generators completed twice");
        ensure!(
            scratch.remaining == 0,
            "{} scheduled witness generators did not complete",
            scratch.remaining
        );
        scratch.failed = false;
        Ok(())
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
