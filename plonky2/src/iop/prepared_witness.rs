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
use crate::iop::generation_scratch::GenerationScratch;
use crate::iop::generator::{GeneratedValues, WitnessGeneratorRef};
use crate::iop::generator_outputs::discover_outputs;
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness, WitnessWrite};
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// A reusable execution order and dependency graph for built-in generators.
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
    order: Vec<usize>,
    /// Read classes with no generator writer, checked once before execution.
    input_slots: Vec<usize>,
    /// Number of distinct predecessors required for reads and duplicate-write checks.
    pending: Vec<usize>,
    /// Generator `g` unlocks `successors[successor_offsets[g]..successor_offsets[g + 1]]`.
    successor_offsets: Vec<usize>,
    successors: Vec<usize>,
    #[cfg(debug_assertions)]
    outputs: Vec<Vec<Target>>,
}

impl<'a, F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PreparedWitnessGenerator<'a, F, C, D>
{
    /// Discovers outputs and builds a deterministic FIFO readiness order.
    /// Read classes with no writer are external inputs, not generator dependencies.
    /// Caller prefills do not affect the plan: produced classes still wait for a writer.
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
        let order = execution_order(generators, &writes, &mut classes)?;
        let mut edges = Vec::new();
        for class in classes.values() {
            if let Some(producer) = class.producer {
                edges.extend(
                    class
                        .readers
                        .iter()
                        .filter_map(|&reader| (reader != producer).then_some((producer, reader))),
                );
            }
        }
        // The first writer in the serial order owns assignment. Other writers
        // wait for it even if the class has no readers. A dense temporary avoids
        // a hash-map entry and a separate writer Vec for every output class.
        let mut selected_writers = vec![usize::MAX; prover.representative_map.len()];
        for &writer in &order {
            for &slot in &writes[writer] {
                let producer = &mut selected_writers[slot];
                if *producer == usize::MAX {
                    *producer = writer;
                } else if *producer != writer {
                    edges.push((*producer, writer));
                }
            }
        }
        drop(selected_writers);
        // Deduplicate predecessors shared by read and write classes.
        edges.sort_unstable();
        edges.dedup();
        let mut position = vec![0; generators.len()];
        for (i, &generator) in order.iter().enumerate() {
            position[generator] = i;
        }
        let mut pending = vec![0; generators.len()];
        let mut successor_offsets = vec![0; generators.len() + 1];
        let mut successors = Vec::with_capacity(edges.len());
        for (producer, consumer) in edges {
            ensure!(
                position[producer] < position[consumer],
                "generator {consumer} depends on later generator {producer}"
            );
            pending[consumer] += 1;
            successor_offsets[producer + 1] += 1;
            successors.push(consumer);
        }
        for g in 0..generators.len() {
            successor_offsets[g + 1] += successor_offsets[g];
        }
        Ok(Self {
            circuit,
            order,
            input_slots,
            pending,
            successor_offsets,
            successors,
            #[cfg(debug_assertions)]
            outputs,
        })
    }

    /// Initial dependency counts, indexed by the circuit's generator indices.
    /// Each distinct predecessor is counted once, for reads and writer ordering.
    pub fn dependency_counts(&self) -> &[usize] {
        &self.pending
    }

    /// Distinct readers and other writers that wait for `generator` to finish.
    /// Entries are distinct and sorted by the circuit's generator indices.
    pub fn successors(&self, generator: usize) -> &[usize] {
        &self.successors[self.successor_offsets[generator]..self.successor_offsets[generator + 1]]
    }

    /// Generates a dense witness, preserving caller-prefill and duplicate checks.
    /// Access the returned witness through its methods; its stock `values` is empty.
    pub fn generate(&self, inputs: PartialWitness<F>) -> Result<PartitionWitness<'a, F>> {
        let mut witness = self.input_witness(inputs)?;
        let mut buffer = GeneratedValues::empty();
        for &index in &self.order {
            self.run_generator(index, &mut witness, &mut buffer)?;
        }
        Ok(witness)
    }

    /// Runs one generator per work item using scoped threads and reusable scratch.
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
        let mut witness = self.input_witness(inputs)?;
        scratch.reset(&self.pending, &self.order, num_threads);
        let pending = &scratch.pending;
        let queue = &scratch.queue;
        let wake = &scratch.wake;
        witness.dense_mut().set_parallel(true);
        std::thread::scope(|scope| {
            for worker in scratch.workers.iter_mut().take(num_threads) {
                let witness = &witness;
                scope.spawn(move || {
                    let result = catch_unwind(AssertUnwindSafe(|| loop {
                        let index = {
                            let mut state = queue.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if state.failure.is_some() || state.panicked || state.remaining == 0
                                {
                                    return;
                                }
                                if let Some(index) = state.ready.pop_front() {
                                    break index;
                                }
                                state = wake.wait(state).unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        // SAFETY: The caller guarantees complete, stable access
                        // sets. Prepared edges order every read and other writer
                        // after the selected writer; publication uses AcqRel
                        // counters followed by the queue mutex handoff.
                        let result = unsafe {
                            self.run_generator_parallel(index, witness, &mut worker.outputs)
                        };
                        if let Err(error) = result {
                            let mut state = queue.lock().unwrap_or_else(|e| e.into_inner());
                            if state.failure.is_none() {
                                state.failure = Some(error);
                            }
                            wake.notify_all();
                            return;
                        }
                        worker.ready_successors.clear();
                        // Counter updates are independent of the queue mutex.
                        for &consumer in self.successors(index) {
                            let previous = pending[consumer].fetch_sub(1, Ordering::AcqRel);
                            assert!(
                                previous > 0,
                                "generator {consumer} dependency released twice"
                            );
                            if previous == 1 {
                                worker.ready_successors.push(consumer);
                            }
                        }
                        let mut state = queue.lock().unwrap_or_else(|e| e.into_inner());
                        state.ready.extend(worker.ready_successors.drain(..));
                        state.remaining -= 1;
                        wake.notify_all();
                    }));
                    if let Err(panic) = result {
                        // Stop parked workers before resuming the panic, or
                        // scope could wait forever for workers with no work.
                        let mut state = queue.lock().unwrap_or_else(|e| e.into_inner());
                        state.panicked = true;
                        wake.notify_all();
                        drop(state);
                        resume_unwind(panic);
                    }
                });
            }
        });
        witness.dense_mut().set_parallel(false);
        let mut state = queue.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(error) = state.failure.take() {
            return Err(error);
        }
        ensure!(state.remaining == 0, "witness generation did not complete");
        Ok(witness)
    }

    fn input_witness(&self, inputs: PartialWitness<F>) -> Result<PartitionWitness<'a, F>> {
        let common = &self.circuit.common;
        let prover = &self.circuit.prover_only;
        let mut witness = PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &prover.representative_map,
        );
        for (target, value) in inputs.target_values {
            witness.set_target(target, value)?;
        }
        let dense = witness.dense_mut();
        for &slot in &self.input_slots {
            ensure!(dense.is_set(slot), "required input slot {slot} is missing");
        }
        Ok(witness)
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

#[derive(Default)]
struct ReadClass {
    // Appended in builder order, with each generator appearing once per class.
    readers: Vec<usize>,
    has_writer: bool,
    producer: Option<usize>,
}

fn execution_order<F: RichField + Extendable<D>, const D: usize>(
    generators: &[WitnessGeneratorRef<F, D>],
    writes: &[Vec<usize>],
    classes: &mut HashMap<usize, ReadClass>,
) -> Result<Vec<usize>> {
    // Count distinct read classes with a writer, not producer generators.
    let mut remaining = vec![0usize; generators.len()];
    for class in classes.values().filter(|class| class.has_writer) {
        for &reader in &class.readers {
            remaining[reader] += 1;
        }
    }
    // The order doubles as a FIFO queue: processed entries remain in place.
    let mut order: Vec<_> = (0..generators.len())
        .filter(|&g| remaining[g] == 0)
        .collect();
    let mut head = 0;
    let mut unlocked = Vec::new();
    while head < order.len() {
        let generator = order[head];
        head += 1;
        unlocked.clear();
        for slot in &writes[generator] {
            let Some(class) = classes.get_mut(slot) else {
                continue;
            };
            if class.producer.is_some() {
                continue;
            }
            class.producer = Some(generator);
            for &reader in &class.readers {
                remaining[reader] = remaining[reader].checked_sub(1).ok_or_else(|| {
                    anyhow!("readiness count released twice for generator {reader}")
                })?;
                if remaining[reader] == 0 {
                    unlocked.push(reader);
                }
            }
        }
        // Several output classes may unlock readers; merge them in builder order.
        unlocked.sort_unstable();
        order.extend_from_slice(&unlocked);
    }
    if order.len() != generators.len() {
        let generator = remaining
            .iter()
            .position(|&count| count != 0)
            .expect("unscheduled generator has an unresolved read");
        let mut slots: Vec<_> = classes
            .iter()
            .filter(|(_, class)| {
                class.has_writer
                    && class.producer.is_none()
                    && class.readers.binary_search(&generator).is_ok()
            })
            .map(|(&slot, _)| slot)
            .collect();
        slots.sort_unstable();
        let blocked_classes = slots.len();
        slots.truncate(8);
        return Err(anyhow!(
            "witness preparation blocked: {} generators could not be ordered; generator {generator} ({}) waits on {blocked_classes} read classes (first eight: {slots:?})",
            generators.len() - order.len(), generators[generator].0.id()
        ));
    }
    Ok(order)
}
