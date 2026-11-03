#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::{anyhow, ensure, Result};
use hashbrown::HashMap;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::{GeneratedValues, WitnessGeneratorRef};
use crate::iop::generator_outputs::discover_outputs;
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness, WitnessWrite};
use crate::plonk::circuit_data::CircuitData;
use crate::plonk::config::GenericConfig;

/// A reusable serial execution order for the circuit's built-in generators.
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

        // All read classes must be known before filtering outputs: later
        // generators may read a class written by an earlier one.
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
                // Outputs without readers cannot unlock any generators.
                if let Some(class) = classes.get_mut(&slot) {
                    class.has_writer = true;
                    slots.push(slot);
                }
            }
            slots.sort_unstable();
            slots.dedup();
            writes.push(slots);
            #[cfg(debug_assertions)]
            outputs.push(discovered);
        }
        let order = execution_order(generators, &writes, classes)?;
        Ok(Self {
            circuit,
            order,
            #[cfg(debug_assertions)]
            outputs,
        })
    }

    /// Generates a dense witness, preserving caller-prefill and duplicate checks.
    /// Access the returned witness through its methods; its stock `values` is empty.
    pub fn generate(&self, inputs: PartialWitness<F>) -> Result<PartitionWitness<'a, F>> {
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
        let mut buffer = GeneratedValues::empty();
        for &index in &self.order {
            let generator = &prover.generators[index].0;
            ensure!(
                generator.run(&witness, &mut buffer),
                "scheduled generator {index} ({}) did not finish",
                generator.id()
            );
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
            for (target, value) in buffer.target_values.drain(..) {
                witness
                    .set_target(target, value)
                    .map_err(|error| anyhow!("generator {index} ({}): {error}", generator.id()))?;
            }
        }
        Ok(witness)
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
    mut classes: HashMap<usize, ReadClass>,
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
            let class = classes.get_mut(slot).expect("write class has readers");
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
