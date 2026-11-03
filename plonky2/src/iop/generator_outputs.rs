#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::{anyhow, ensure, Result};

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::iop::generator::{GeneratedValues, WitnessGenerator};
use crate::iop::target::Target;
use crate::iop::witness::PartitionWitness;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::util::serialization::{Buffer, Read};

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
pub(crate) fn discover_outputs<F: RichField + Extendable<D>, const D: usize>(
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

#[derive(Clone, Copy)]
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
