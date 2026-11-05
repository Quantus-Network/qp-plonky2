#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use hashbrown::HashMap;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generation_schedule::{make_schedule, GenerationSchedule};
use crate::iop::generator::{GeneratedValues, WitnessGenerator};
use crate::iop::scheduling_state::SchedulingState;
use crate::iop::target::Target;
use crate::iop::witness::PartitionWitness;
use crate::iop::witness_generator_executor::ExecutionData;
use crate::plonk::circuit_data::{CircuitData, CommonCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, Read};

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
    execution: Arc<ExecutionData<'a, F, C, D>>,
    reads: Vec<Vec<usize>>,
    writes: Vec<Vec<usize>>,
    consumers: HashMap<usize, Vec<usize>>,
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
            execution: Arc::new(ExecutionData::new(
                circuit,
                #[cfg(debug_assertions)]
                outputs,
            )),
            reads,
            writes,
            consumers,
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
            Arc::clone(self.execution.identity()),
            self.execution
                .circuit()
                .prover_only
                .representative_map
                .len(),
            self.reads.len(),
        )
    }

    /// Appends a runnable depth-first closure to the supplied planned state.
    /// Nothing is executed. Blocked generators remain unscheduled; an empty
    /// schedule is valid. All nonzero state bytes mean assumed available.
    pub fn make_schedule(&self, state: &mut SchedulingState) -> Result<GenerationSchedule> {
        ensure!(
            state.chain().matches_scheduler(self.execution.identity()),
            "scheduling state belongs to another scheduler"
        );
        Ok(make_schedule(
            &self.reads,
            &self.writes,
            &self.consumers,
            state,
        ))
    }

    /// Builds a one-shot schedule assuming the flagged representatives supplied.
    /// Returns an error if any generator remains blocked. Planning state is
    /// local and discarded; caller buffers are never modified.
    pub fn make_complete_schedule(&self, classes_set: &[u8]) -> Result<GenerationSchedule> {
        ensure!(
            classes_set.len()
                == self
                    .execution
                    .circuit()
                    .prover_only
                    .representative_map
                    .len(),
            "supplied representative count does not match circuit"
        );
        let mut state = SchedulingState::with_supplied(
            Arc::clone(self.execution.identity()),
            classes_set,
            self.reads.len(),
        );
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
        let circuit = self.execution.circuit();
        let common = &circuit.common;
        PartitionWitness::new_dense(
            common.config.num_wires,
            common.degree(),
            &circuit.prover_only.representative_map,
        )
    }

    pub(super) fn execution_data(&self) -> Arc<ExecutionData<'a, F, C, D>> {
        Arc::clone(&self.execution)
    }
}

/// Discovers ordered output targets by executing a built-in with dummy inputs.
/// Duplicate emissions are preserved; generated values are never applied to scratch.
///
/// `scratch` must be a dedicated dense witness for the same circuit, initially
/// unassigned. Only its declared read slots are populated and then cleared, on
/// success or a returned error. Stored values need not be zeroed between probes.
///
/// Recognition uses built-in IDs, not concrete type identification. Callers must
/// supply genuine built-ins; custom implementations reusing those IDs are unsupported.
/// This relies on built-in output targets and their order being input-independent.
fn discover_outputs<F: RichField + Extendable<D>, const D: usize>(
    generator: &dyn WitnessGenerator<F, D>,
    common: &CommonCircuitData<F, D>,
    scratch: &mut PartitionWitness<'_, F>,
) -> Result<Vec<Target>> {
    let id = generator.id();
    let rule = dummy_rule(&id)?;
    ensure!(
        scratch.num_wires == common.config.num_wires && scratch.degree == common.degree(),
        "output probe scratch layout does not match the circuit"
    );
    let reads = generator
        .reads()
        .ok_or_else(|| anyhow!("generator {id} does not declare its reads"))?;
    let override_value = match rule {
        DummyRule::Zero => None,
        DummyRule::Quotient => {
            ensure!(
                D > 0 && D.checked_mul(2) == Some(reads.len()),
                "unexpected quotient read declaration for {id}"
            );
            Some((reads[D], F::ONE))
        }
        DummyRule::Interpolation => Some((
            *reads
                .first()
                .ok_or_else(|| anyhow!("{id} has no shift input"))?,
            F::ONE,
        )),
        DummyRule::Lookup => {
            ensure!(
                reads.len() == 1,
                "unexpected lookup read declaration for {id}"
            );
            let mut parameters = Vec::new();
            generator
                .serialize(&mut parameters, common)
                .map_err(|_| anyhow!("could not serialize {id} for output discovery"))?;
            let mut reader = Buffer::new(&parameters);
            let table = (|| {
                reader.read_usize()?; // row
                reader.read_usize()?; // slot within the row
                reader.read_usize() // common lookup-table index
            })()
            .map_err(|_| anyhow!("invalid serialized lookup parameters for {id}"))?;
            let &(key, _) = common
                .luts
                .get(table)
                .and_then(|lut| lut.first())
                .ok_or_else(|| anyhow!("{id} references missing or empty lookup table {table}"))?;
            Some((reads[0], F::from_canonical_u16(key)))
        }
    };
    let slot_of = |target: Target| -> Result<usize> {
        let index = target.index(scratch.num_wires, scratch.degree);
        let slot = *scratch
            .representative_map
            .get(index)
            .ok_or_else(|| anyhow!("{id} reads target {target:?} outside scratch"))?;
        ensure!(
            slot < scratch.representative_map.len(),
            "{id} has an invalid representative slot {slot}"
        );
        Ok(slot)
    };
    let override_value = override_value
        .map(|(target, value)| slot_of(target).map(|slot| (slot, value)))
        .transpose()?;
    let mut slots = reads.into_iter().map(slot_of).collect::<Result<Vec<_>>>()?;
    slots.sort_unstable();
    slots.dedup();

    let mut buffer = GeneratedValues::empty();
    let result = (|| -> Result<()> {
        let dense = scratch.dense_mut();
        for &slot in &slots {
            dense.clear(slot);
        }
        for &slot in &slots {
            // Apply overrides by representative so aliased read targets agree.
            let value = override_value
                .filter(|&(special, _)| special == slot)
                .map_or(F::ZERO, |(_, value)| value);
            dense.assign(slot, value)?;
        }
        ensure!(
            generator.run(scratch, &mut buffer),
            "built-in output probe failed for {id}; check the dummy-input rule and instance parameters"
        );
        Ok(())
    })();
    let dense = scratch.dense_mut();
    for slot in slots {
        dense.clear(slot);
    }
    result?;
    Ok(buffer
        .target_values
        .into_iter()
        .map(|(target, _)| target)
        .collect())
}

enum DummyRule {
    Zero,
    Quotient,
    Interpolation,
    Lookup,
}

fn dummy_rule(id: &str) -> Result<DummyRule> {
    match id {
        "QuotientGeneratorExtension" => Ok(DummyRule::Quotient),
        "InterpolationGenerator" => Ok(DummyRule::Interpolation),
        "LookupGenerator" => Ok(DummyRule::Lookup),
        "ArithmeticBaseGenerator"
        | "ArithmeticExtensionGenerator"
        | "MulExtensionGenerator"
        | "ReducingGenerator"
        | "ReducingExtensionGenerator"
        | "RandomAccessGenerator"
        | "ExponentiationGenerator"
        | "PoseidonGenerator"
        | "PoseidonMdsGenerator"
        | "Poseidon2FullGen"
        | "Poseidon2MdsGenerator"
        | "Poseidon2IntMixGenerator"
        | "LookupTableGenerator"
        | "DummyProofGenerator"
        | "SplitGenerator"
        | "WireSplitGenerator"
        | "EqualityGenerator"
        | "LowHighGenerator"
        | "CopyGenerator"
        | "RandomValueGenerator"
        | "NonzeroTestGenerator"
        | "ConstantGenerator" => Ok(DummyRule::Zero),
        _ => {
            if let Some(base) = id.strip_prefix("BaseSplitGenerator + Base: ") {
                let base = parse_base(id, base)?;
                ensure!(base != 0, "invalid zero base for {id}");
                return Ok(DummyRule::Zero);
            }
            if let Some(base) = id.strip_prefix("BaseSumGenerator + Base: ") {
                parse_base(id, base)?;
                return Ok(DummyRule::Zero);
            }
            Err(anyhow!(
                "unsupported generator ID for output discovery: {id}"
            ))
        }
    }
}

fn parse_base(id: &str, base: &str) -> Result<usize> {
    base.parse()
        .map_err(|_| anyhow!("invalid base in built-in generator ID: {id}"))
}
