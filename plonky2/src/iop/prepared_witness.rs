#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{anyhow, ensure, Result};
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
#[cfg(feature = "std")]
use crate::iop::generation_pool::GenerationPool;
use crate::iop::generation_schedule::{
    prepare_schedule, prepopulation_members, GenerationSchedule, ReadClass,
};
#[cfg(feature = "std")]
use crate::iop::generation_scratch::GenerationScratch;
use crate::iop::generator::GeneratedValues;
use crate::iop::generator_outputs::discover_outputs;
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// Prepared input-independent and remaining schedules for built-in generators.
/// Output discovery assumes input-independent output targets and emission order;
/// debug builds check subsequent successful runs against those declarations.
/// Built-ins are recognized by ID; custom implementations reusing those IDs
/// are unsupported.
#[derive(Debug)]
pub struct PreparedWitnessGenerator<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    circuit: &'a CircuitData<F, C, D>,
    prepopulation: GenerationSchedule,
    generation: GenerationSchedule,
    #[cfg(debug_assertions)]
    outputs: Vec<Vec<Target>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PreparedWitnessGenerator<'a, F, C, D>
{
    /// Discovers outputs and builds two deterministic depth-first readiness schedules.
    /// Prepopulation includes every generator runnable without caller inputs.
    /// The remaining order assumes caller inputs are available; its runtime
    /// dependency counts also wait for their actual assignment.
    /// Caller prefills do not change schedule membership or writer selection.
    pub fn prepare(circuit: &'a CircuitData<F, C, D>) -> Result<Self> {
        let common = &circuit.common;
        let prover = &circuit.prover_only;
        let generators = &prover.generators;
        let resolve = |target: Target| -> Result<usize> {
            let index = target.index(common.config.num_wires, common.degree());
            let slot = *prover
                .representative_map
                .get(index)
                .ok_or_else(|| anyhow!("target {target:?} is outside the witness"))?;
            ensure!(
                slot < prover.representative_map.len(),
                "invalid representative slot {slot}"
            );
            Ok(slot)
        };
        let mut classes: HashMap<usize, ReadClass> = HashMap::new();
        for (index, generator) in generators.iter().enumerate() {
            let reads = generator.0.reads().ok_or_else(|| {
                anyhow!(
                    "generator {index} ({}) does not declare its reads",
                    generator.0.id()
                )
            })?;
            let mut slots = reads
                .into_iter()
                .map(&resolve)
                .collect::<Result<Vec<_>>>()?;
            slots.sort_unstable();
            slots.dedup();
            for slot in slots {
                classes.entry(slot).or_default().readers.push(index);
            }
        }

        // Keep all output slots, including classes with no readers, so writer
        // ordering also covers classes that never unlock a generator.
        let mut scratch = PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &prover.representative_map,
        );
        let mut writes = Vec::with_capacity(generators.len());
        #[cfg(debug_assertions)]
        let mut outputs = Vec::with_capacity(generators.len());
        for (index, generator) in generators.iter().enumerate() {
            let discovered = discover_outputs(generator.0.as_ref(), common, &mut scratch)
                .map_err(|error| anyhow!("generator {index} ({}): {error}", generator.0.id()))?;
            let mut slots = Vec::new();
            for &target in &discovered {
                let slot = resolve(target)?;
                if let Some(class) = classes.get_mut(&slot) {
                    class.has_writer = true;
                }
                slots.push(slot);
            }
            slots.sort_unstable();
            slots.dedup();
            writes.push(slots);
            #[cfg(debug_assertions)]
            outputs.push(discovered);
        }
        let prepopulation_members = prepopulation_members(&classes, &writes)?;
        let prepopulation = prepare_schedule(&prepopulation_members, &classes, &writes, &[])?;
        let mut is_prepopulation = vec![false; generators.len()];
        let mut supplied_classes = Vec::new();
        for &generator in &prepopulation.order {
            is_prepopulation[generator] = true;
            supplied_classes.extend_from_slice(&writes[generator]);
        }
        supplied_classes.sort_unstable();
        supplied_classes.dedup();
        let generation_members: Vec<_> = (0..generators.len())
            .filter(|&g| !is_prepopulation[g])
            .collect();
        let generation =
            prepare_schedule(&generation_members, &classes, &writes, &supplied_classes)?;
        Ok(Self {
            circuit,
            prepopulation,
            generation,
            #[cfg(debug_assertions)]
            outputs,
        })
    }

    /// Generators runnable without caller inputs, including random generators.
    pub fn prepopulation_schedule(&self) -> &GenerationSchedule {
        &self.prepopulation
    }

    /// Remaining generators, whose dependencies exclude prepopulation outputs.
    pub fn generation_schedule(&self) -> &GenerationSchedule {
        &self.generation
    }

    /// Runs prepopulation, applies caller inputs, then runs the remaining schedule.
    /// Generates a dense witness, preserving caller-prefill and duplicate checks.
    /// Access the returned witness through its methods; its public `values` is empty.
    pub fn generate(&self, inputs: PartialWitness<F>) -> Result<PartitionWitness<'a, F>> {
        let mut witness = self.new_witness();
        let mut buffer = GeneratedValues::empty();
        self.run_schedule(&self.prepopulation, &mut witness, &mut buffer)?;
        Self::write_inputs(&mut witness, inputs, |_| {})?;
        self.check_required_inputs(&mut witness)?;
        self.run_schedule(&self.generation, &mut witness, &mut buffer)?;
        Ok(witness)
    }

    /// Runs both schedules using scoped threads and reusable scratch, applying
    /// caller inputs between them. Each nonempty schedule spawns and joins workers.
    /// Workers share one dense witness. Selected writers publish values before
    /// readers and other writers run; other writers only compare assigned slots.
    /// Worker errors return after all workers finish; worker panics propagate.
    ///
    /// # Safety
    /// Every generator must declare every read (including presence queries), and
    /// must emit exactly the targets observed during preparation, independently
    /// of input values. Generators may only access witness data through its
    /// target-based read methods and helpers; they must not inspect or modify
    /// backing storage or retain access beyond the call. Built-in IDs and debug
    /// output checks do not establish this contract for arbitrary implementations.
    #[cfg(feature = "std")]
    pub unsafe fn generate_parallel(
        &self,
        inputs: PartialWitness<F>,
        scratch: &mut GenerationScratch<F>,
        num_threads: usize,
    ) -> Result<PartitionWitness<'a, F>> {
        // SAFETY: The caller supplies the access declarations required by both
        // initialization and the subsequent parallel execution.
        let mut witness = unsafe { self.start_generation(scratch, num_threads)? };
        self.apply_inputs(&mut witness, scratch, inputs)?;
        self.check_required_inputs(&mut witness)?;
        // SAFETY: This witness and scratch were initialized together above.
        unsafe { self.run_available_parallel(&mut witness, scratch)? };
        ensure!(scratch.is_complete(), "witness generation did not complete");
        Ok(witness)
    }

    /// Creates a dense witness, runs prepopulation, and initializes execution
    /// scratch for inputs that will be supplied through `apply_inputs`.
    /// A new call starts a new witness and resets the supplied scratch.
    ///
    /// # Safety
    /// Requires the complete, stable generator access declarations documented
    /// by `generate_parallel`.
    #[cfg(feature = "std")]
    pub unsafe fn start_generation(
        &self,
        scratch: &mut GenerationScratch<F>,
        num_threads: usize,
    ) -> Result<PartitionWitness<'a, F>> {
        ensure!(
            num_threads > 0,
            "witness generation requires at least one worker"
        );
        let mut witness = self.new_witness();
        scratch.reset(
            &self.prepopulation.pending,
            &self.prepopulation.initial_ready,
            num_threads,
        );
        // SAFETY: The caller supplies the access contract; prepopulation has
        // no caller-input dependencies.
        unsafe { self.run_schedule_parallel(&self.prepopulation, &mut witness, scratch)? };
        ensure!(
            scratch.is_complete(),
            "witness prepopulation did not complete"
        );
        scratch.reset(
            &self.generation.pending,
            &self.generation.initial_ready,
            num_threads,
        );
        Ok(witness)
    }

    /// Applies a batch of caller values and records ready generators whose last
    /// dependency was supplied. Aliases and repeated assignments release an
    /// input class only once. Call only between executions for this witness.
    /// All caller assignments during this generation must pass through here.
    #[cfg(feature = "std")]
    pub fn apply_inputs(
        &self,
        witness: &mut PartitionWitness<'_, F>,
        scratch: &mut GenerationScratch<F>,
        inputs: PartialWitness<F>,
    ) -> Result<()> {
        self.check_execution(witness, scratch)?;
        scratch.failed = true;
        Self::write_inputs(witness, inputs, |slot| {
            if let Some(input) = self.generation.input_index(slot) {
                self.generation
                    .release_input(input, &mut scratch.pending, |consumer| {
                        scratch.ready.push(consumer);
                    });
            }
        })?;
        scratch.failed = false;
        Ok(())
    }

    /// Runs currently available generators serially, preserving pending counts
    /// for subsequent input batches. An incomplete witness is not an error.
    #[cfg(feature = "std")]
    pub fn run_available(
        &self,
        witness: &mut PartitionWitness<'_, F>,
        scratch: &mut GenerationScratch<F>,
    ) -> Result<()> {
        self.check_execution(witness, scratch)?;
        scratch.failed = true;
        scratch.distribute_ready(&self.generation.order);
        while let Some(local) = scratch.workers[..scratch.num_workers]
            .iter()
            .find_map(|worker| worker.ready.pop())
        {
            let worker = &mut scratch.workers[0];
            self.run_generator::<true>(self.generation.order[local], witness, &mut worker.outputs)?;
            self.generation
                .release_successors(local, &scratch.pending, |consumer| {
                    worker.ready.push(consumer);
                });
            scratch.remaining -= 1;
        }
        scratch.failed = false;
        Ok(())
    }

    /// Runs currently available generators using the initialized worker count.
    /// Returns when no queued or running work remains, even if inputs are still
    /// missing. `scratch.is_complete()` reports whether every generator finished.
    /// Errors or panics invalidate scratch until `start_generation` is called again.
    ///
    /// # Safety
    /// Requires `generate_parallel`'s generator access contract. The witness and
    /// scratch must be the unchanged pair from this generator's `start_generation`,
    /// except for values supplied through `apply_inputs` and preceding executions.
    #[cfg(feature = "std")]
    pub unsafe fn run_available_parallel(
        &self,
        witness: &mut PartitionWitness<'_, F>,
        scratch: &mut GenerationScratch<F>,
    ) -> Result<()> {
        self.check_execution(witness, scratch)?;
        // SAFETY: The caller supplies the access contract. Input assignments
        // precede thread creation and runtime counts gate every generator.
        unsafe { self.run_schedule_parallel(&self.generation, witness, scratch) }
    }

    #[cfg(feature = "std")]
    fn check_execution(
        &self,
        witness: &PartitionWitness<'_, F>,
        scratch: &GenerationScratch<F>,
    ) -> Result<()> {
        ensure!(scratch.initialized, "witness execution is not initialized");
        ensure!(
            !scratch.failed,
            "witness execution failed; start a new generation"
        );
        ensure!(
            scratch.pending.len() == self.generation.order.len(),
            "execution scratch does not match the generation schedule"
        );
        ensure!(
            core::ptr::eq(
                witness.representative_map,
                self.circuit.prover_only.representative_map.as_slice()
            ) && witness.num_wires == self.circuit.common.config.num_wires
                && witness.degree == self.circuit.common.degree(),
            "witness does not match the prepared circuit"
        );
        Ok(())
    }

    fn run_schedule(
        &self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        for &generator in &schedule.order {
            self.run_generator::<true>(generator, witness, buffer)?;
        }
        Ok(())
    }

    /// # Safety
    /// Requires generate_parallel's complete, stable access declarations. Inputs
    /// not represented by pending input counts must already be assigned.
    /// Scratch must be initialized for this schedule and witness.
    #[cfg(feature = "std")]
    unsafe fn run_schedule_parallel(
        &self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        scratch: &mut GenerationScratch<F>,
    ) -> Result<()> {
        let num_threads = scratch.num_workers;
        scratch.distribute_ready(&schedule.order);
        if scratch.workers[..num_threads]
            .iter()
            .all(|worker| worker.ready.is_empty())
        {
            return Ok(());
        }
        scratch.failed = true;
        let pending = &scratch.pending;
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
                                let generator = schedule.order[local];
                                // SAFETY: The caller guarantees stable, complete access
                                // sets. Counter AcqRel publication and deque handoff
                                // order every read/comparison after the selected writer.
                                let result = unsafe {
                                    self.run_generator_parallel::<true>(
                                        generator,
                                        witness,
                                        &mut worker.outputs,
                                    )
                                };
                                if let Err(error) = result {
                                    pool.fail(error);
                                    return;
                                }
                                let pushed =
                                    schedule.release_successors(local, pending, |consumer| {
                                        worker.ready.push(consumer);
                                    });
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
        scratch.failed = false;
        Ok(())
    }

    fn new_witness(&self) -> PartitionWitness<'a, F> {
        let common = &self.circuit.common;
        PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &self.circuit.prover_only.representative_map,
        )
    }

    /// Assigns caller values and reports each newly assigned representative once.
    fn write_inputs(
        witness: &mut PartitionWitness<'_, F>,
        inputs: PartialWitness<F>,
        mut on_assignment: impl FnMut(usize),
    ) -> Result<()> {
        for (target, value) in inputs.target_values {
            if let Some(slot) = witness.set_target_returning_rep(target, value)? {
                on_assignment(slot);
            }
        }
        Ok(())
    }

    fn check_required_inputs(&self, witness: &mut PartitionWitness<'_, F>) -> Result<()> {
        let dense = witness.dense_mut();
        for &slot in &self.generation.input_slots {
            ensure!(dense.is_set(slot), "required input slot {slot} is missing");
        }
        Ok(())
    }

    fn run_generator<const READY: bool>(
        &self,
        index: usize,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<bool> {
        if READY {
            self.run_generator_outputs_ready(index, witness, buffer)?;
        } else if !self.run_generator_outputs(index, witness, buffer)? {
            return Ok(false);
        }
        let generator = &self.circuit.prover_only.generators[index].0;
        for (target, value) in buffer.target_values.drain(..) {
            witness
                .set_target(target, value)
                .map_err(|error| anyhow!("generator {index} ({}): {error}", generator.id()))?;
        }
        Ok(true)
    }

    #[cfg(feature = "std")]
    unsafe fn run_generator_parallel<const READY: bool>(
        &self,
        index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<bool> {
        if READY {
            self.run_generator_outputs_ready(index, witness, buffer)?;
        } else if !self.run_generator_outputs(index, witness, buffer)? {
            return Ok(false);
        }
        // SAFETY: The prepared graph and generate_parallel's access contract
        // guarantee the required ordering of reads and writes.
        unsafe { self.write_generator_outputs(index, witness, buffer)? };
        Ok(true)
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

    /// Returns false without running when a dependency is missing.
    fn run_generator_outputs(
        &self,
        index: usize,
        witness: &PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<bool> {
        let generator = &self.circuit.prover_only.generators[index].0;
        if !witness.contains_all(&generator.watch_list()) {
            return Ok(false);
        }
        self.run_generator_outputs_ready(index, witness, buffer)?;
        Ok(true)
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
