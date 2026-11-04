#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{anyhow, ensure, Result};
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
#[cfg(feature = "std")]
use std::sync::atomic::Ordering;

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
use crate::iop::witness::{PartialWitness, PartitionWitness, WitnessWrite};
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
    /// Read classes with no generator writer, checked before the main schedule.
    input_slots: Vec<usize>,
    #[cfg(debug_assertions)]
    outputs: Vec<Vec<Target>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PreparedWitnessGenerator<'a, F, C, D>
{
    /// Discovers outputs and builds two deterministic FIFO readiness schedules.
    /// Prepopulation includes every generator runnable without caller inputs.
    /// The remaining schedule treats its outputs and caller inputs as available.
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
        let mut input_slots: Vec<_> = classes
            .iter()
            .filter(|(_, class)| !class.has_writer)
            .map(|(&slot, _)| slot)
            .collect();
        input_slots.sort_unstable();
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
            input_slots,
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
        self.apply_inputs(&mut witness, inputs)?;
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
        ensure!(
            num_threads > 0,
            "witness generation requires at least one worker"
        );
        let mut witness = self.new_witness();
        // SAFETY: The caller's access contract covers every generator in both
        // schedules. Joining prepopulation publishes its values to the next one.
        unsafe {
            self.run_schedule_parallel(&self.prepopulation, &mut witness, scratch, num_threads)?;
        }
        self.apply_inputs(&mut witness, inputs)?;
        // SAFETY: As above; caller inputs are applied before workers are spawned.
        unsafe {
            self.run_schedule_parallel(&self.generation, &mut witness, scratch, num_threads)?;
        }
        Ok(witness)
    }

    fn run_schedule(
        &self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        for &generator in &schedule.order {
            self.run_generator(generator, witness, buffer)?;
        }
        Ok(())
    }

    /// # Safety
    /// Requires generate_parallel's complete, stable access declarations. Inputs
    /// external to this schedule must be assigned before its workers start.
    #[cfg(feature = "std")]
    unsafe fn run_schedule_parallel(
        &self,
        schedule: &GenerationSchedule,
        witness: &mut PartitionWitness<'_, F>,
        scratch: &mut GenerationScratch<F>,
        num_threads: usize,
    ) -> Result<()> {
        if schedule.order.is_empty() {
            return Ok(());
        }
        scratch.reset(&schedule.pending, &schedule.initial_ready, num_threads);
        let pending = &scratch.pending;
        let mut workers = core::mem::take(&mut scratch.workers);
        let inactive = workers.split_off(num_threads);
        let pool = GenerationPool::new(
            workers
                .iter()
                .map(|worker| worker.ready.stealer())
                .collect(),
            schedule.order.len(),
        );
        witness.dense_mut().set_parallel(true);
        scratch.workers = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_threads);
            for (index, mut worker) in workers.into_iter().enumerate() {
                let witness = &*witness;
                let pool = &pool;
                // Each deque moves with its owning worker; only its Stealer is shared.
                let handle = std::thread::Builder::new().spawn_scoped(scope, move || {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let mut victim = (index + 1) % num_threads;
                        while let Some(local) = pool.next(&worker.ready, index, &mut victim) {
                            let generator = schedule.order[local];
                            // SAFETY: The caller guarantees stable, complete access
                            // sets. Counter AcqRel publication and deque handoff
                            // order every read/comparison after the selected writer.
                            let result = unsafe {
                                self.run_generator_parallel(generator, witness, &mut worker.outputs)
                            };
                            if let Err(error) = result {
                                pool.fail(error);
                                return;
                            }
                            let mut pushed = false;
                            for &consumer in schedule.successors(local) {
                                let previous = pending[consumer].fetch_sub(1, Ordering::AcqRel);
                                assert!(
                                    previous > 0,
                                    "generator {} dependency released twice",
                                    schedule.order[consumer]
                                );
                                if previous == 1 {
                                    worker.ready.push(consumer);
                                    pushed = true;
                                }
                            }
                            if pushed {
                                pool.notify_work();
                            }
                            pool.complete();
                        }
                    }));
                    if let Err(panic) = result {
                        pool.abort();
                        resume_unwind(panic);
                    }
                    worker
                });
                match handle {
                    Ok(handle) => handles.push(handle),
                    Err(error) => {
                        pool.fail(anyhow!("could not spawn witness worker {index}: {error}"));
                        break;
                    }
                }
            }
            let mut completed = Vec::with_capacity(num_threads);
            let mut panic = None;
            for handle in handles {
                match handle.join() {
                    Ok(worker) => completed.push(worker),
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
            completed
        });
        scratch.workers.extend(inactive);
        witness.dense_mut().set_parallel(false);
        pool.finish()
    }

    fn new_witness(&self) -> PartitionWitness<'a, F> {
        let common = &self.circuit.common;
        PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &self.circuit.prover_only.representative_map,
        )
    }

    fn apply_inputs(
        &self,
        witness: &mut PartitionWitness<'_, F>,
        inputs: PartialWitness<F>,
    ) -> Result<()> {
        for (target, value) in inputs.target_values {
            witness.set_target(target, value)?;
        }
        let dense = witness.dense_mut();
        for &slot in &self.input_slots {
            ensure!(dense.is_set(slot), "required input slot {slot} is missing");
        }
        Ok(())
    }

    fn run_generator(
        &self,
        index: usize,
        witness: &mut PartitionWitness<'_, F>,
        buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        self.run_generator_outputs(index, witness, buffer)?;
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
        self.run_generator_outputs(index, witness, buffer)?;
        for (target, value) in buffer.target_values.drain(..) {
            // SAFETY: generate_parallel's access contract and the prepared graph
            // grant exclusive assignment, then concurrent read-only comparisons.
            unsafe { witness.set_target_shared(target, value) }.map_err(|error| {
                anyhow!(
                    "generator {index} ({}): {error}",
                    self.circuit.prover_only.generators[index].0.id()
                )
            })?;
        }
        Ok(())
    }

    fn run_generator_outputs(
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
