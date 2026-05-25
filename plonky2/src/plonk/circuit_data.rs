//! Circuit data specific to the prover and the verifier.
//!
//! This module also defines a [`CircuitConfig`] to be customized
//! when building circuits for arbitrary statements.
//!
//! After building a circuit, one obtains an instance of [`CircuitData`].
//! This contains both prover and verifier data, allowing to generate
//! proofs for the given circuit and verify them.
//!
//! Most of the [`CircuitData`] is actually prover-specific, and can be
//! extracted by calling [`CircuitData::prover_data`] method.
//! The verifier data can similarly be extracted by calling [`CircuitData::verifier_data`].
//! This is useful to allow even small devices to verify plonky2 proofs.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, vec, vec::Vec};
use core::ops::{Range, RangeFrom};
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use anyhow::{ensure, Result};
pub use qp_plonky2_core::CircuitConfig;
use qp_plonky2_core::ZkMode;
use serde::Serialize;

use super::circuit_builder::LookupWire;
use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::structure::{
    FriBatchInfo, FriBatchInfoTarget, FriInstanceInfo, FriInstanceInfoTarget, FriOpeningExpression,
    FriOracleInfo, FriOracleLayout, FriPolynomialInfo,
};
use crate::fri::FriParams;
// Re-export CircuitConfig from core
use crate::gates::gate::{check_gate_id_collisions, GateRef};
use crate::gates::lookup::{Lookup, LookupGate};
use crate::gates::lookup_table::{LookupTable, LookupTableGate};
use crate::gates::selectors::SelectorsInfo;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{generate_partial_witness, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use crate::plonk::prover::prove;
use crate::plonk::verifier::verify;
use crate::util::log2_ceil;
use crate::util::serialization::{
    Buffer, GateSerializer, IoResult, Read, WitnessGeneratorSerializer, Write,
};
use crate::util::timing::TimingTree;

/// Mock circuit data to only do witness generation without generating a proof.
#[derive(Eq, PartialEq, Debug)]
pub struct MockCircuitData<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    MockCircuitData<F, C, D>
{
    pub fn generate_witness(&self, inputs: PartialWitness<F>) -> PartitionWitness<'_, F> {
        generate_partial_witness::<F, C, D>(inputs, &self.prover_only, &self.common).unwrap()
    }
}

/// Circuit data required by the prover or the verifier.
#[derive(Eq, PartialEq, Debug)]
pub struct CircuitData<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub verifier_only: VerifierOnlyCircuitData<C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_circuit_data(self, gate_serializer, generator_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_circuit_data(gate_serializer, generator_serializer)
    }

    pub fn prove(&self, inputs: PartialWitness<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        prove::<F, C, D>(
            &self.prover_only,
            &self.common,
            inputs,
            &mut TimingTree::default(),
        )
    }

    /// Verify a proof for this circuit.
    ///
    /// **IMPORTANT**: For cyclic recursive circuits (those using
    /// [`conditionally_verify_cyclic_proof`](crate::recursion::cyclic_recursion) or
    /// [`conditionally_verify_cyclic_proof_or_dummy`](crate::recursion::cyclic_recursion)),
    /// you MUST use [`verify_cyclic`](Self::verify_cyclic) instead. This method does not
    /// verify that the verifier data embedded in the proof's public inputs matches the
    /// actual circuit, which is required for cyclic recursion security.
    pub fn verify(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()> {
        verify::<F, C, D>(proof_with_pis, &self.verifier_only, &self.common)
    }

    /// Verify a cyclic recursive proof.
    ///
    /// This method MUST be used instead of [`verify`](Self::verify) for circuits that use
    /// cyclic recursion ([`conditionally_verify_cyclic_proof`](crate::recursion::cyclic_recursion)
    /// or [`conditionally_verify_cyclic_proof_or_dummy`](crate::recursion::cyclic_recursion)).
    ///
    /// In addition to standard proof verification, this checks that the verifier data
    /// embedded in the proof's public inputs matches the actual verifier data for this
    /// circuit. This prevents an attacker from substituting a valid proof chain built
    /// with a different (but structurally identical) circuit.
    ///
    /// # Security
    ///
    /// Without this check, an attacker could:
    /// 1. Build a malicious circuit with the same structure as the legitimate one
    /// 2. Create a valid proof chain using their circuit
    /// 3. Present it as a proof for the legitimate circuit
    ///
    /// The embedded verifier data check ensures the proof was actually generated for
    /// this specific circuit.
    pub fn verify_cyclic(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        self.verify(proof_with_pis.clone())?;
        crate::recursion::cyclic_recursion::check_cyclic_proof_verifier_data(
            &proof_with_pis,
            &self.verifier_only,
            &self.common,
        )
    }

    pub fn verify_compressed(
        &self,
        compressed_proof_with_pis: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<()> {
        compressed_proof_with_pis.verify(&self.verifier_only, &self.common)
    }

    pub fn compress(
        &self,
        proof: ProofWithPublicInputs<F, C, D>,
    ) -> Result<CompressedProofWithPublicInputs<F, C, D>> {
        proof.compress(&self.verifier_only.circuit_digest, &self.common)
    }

    pub fn decompress(
        &self,
        proof: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        proof.decompress(&self.verifier_only.circuit_digest, &self.common)
    }

    pub fn verifier_data(&self) -> VerifierCircuitData<F, C, D> {
        let CircuitData {
            verifier_only,
            common,
            ..
        } = self;
        VerifierCircuitData {
            verifier_only: verifier_only.clone(),
            common: common.clone(),
        }
    }

    pub fn prover_data(self) -> ProverCircuitData<F, C, D> {
        let CircuitData {
            prover_only,
            common,
            ..
        } = self;
        ProverCircuitData {
            prover_only,
            common,
        }
    }
}

/// Circuit data required by the prover. This may be thought of as a proving key, although it
/// includes code for witness generation.
///
/// The goal here is to make proof generation as fast as we can, rather than making this prover
/// structure as succinct as we can. Thus we include various precomputed data which isn't strictly
/// required, like LDEs of preprocessed polynomials. If more succinctness was desired, we could
/// construct a more minimal prover structure and convert back and forth.
#[derive(Debug)]
pub struct ProverCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub prover_only: ProverOnlyCircuitData<F, C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverCircuitData<F, C, D>
{
    pub fn to_bytes(
        &self,
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_prover_circuit_data(self, gate_serializer, generator_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        gate_serializer: &dyn GateSerializer<F, D>,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_prover_circuit_data(gate_serializer, generator_serializer)
    }

    pub fn prove(&self, inputs: PartialWitness<F>) -> Result<ProofWithPublicInputs<F, C, D>> {
        prove::<F, C, D>(
            &self.prover_only,
            &self.common,
            inputs,
            &mut TimingTree::default(),
        )
    }
}

/// Circuit data required by the verifier (a subset of the full circuit data).
///
/// This can be extracted from [`CircuitData`] via [`CircuitData::verifier_data`] and
/// distributed to verifiers who don't need the prover-specific data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub verifier_only: VerifierOnlyCircuitData<C, D>,
    pub common: CommonCircuitData<F, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    VerifierCircuitData<F, C, D>
{
    pub fn to_bytes(&self, gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_verifier_circuit_data(self, gate_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        gate_serializer: &dyn GateSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(&bytes);
        buffer.read_verifier_circuit_data(gate_serializer)
    }

    /// Verify a proof for this circuit.
    ///
    /// **IMPORTANT**: For cyclic recursive circuits (those using
    /// [`conditionally_verify_cyclic_proof`](crate::recursion::cyclic_recursion) or
    /// [`conditionally_verify_cyclic_proof_or_dummy`](crate::recursion::cyclic_recursion)),
    /// you MUST use [`verify_cyclic`](Self::verify_cyclic) instead. This method does not
    /// verify that the verifier data embedded in the proof's public inputs matches the
    /// actual circuit, which is required for cyclic recursion security.
    pub fn verify(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()> {
        verify::<F, C, D>(proof_with_pis, &self.verifier_only, &self.common)
    }

    /// Verify a cyclic recursive proof.
    ///
    /// This method MUST be used instead of [`verify`](Self::verify) for circuits that use
    /// cyclic recursion ([`conditionally_verify_cyclic_proof`](crate::recursion::cyclic_recursion)
    /// or [`conditionally_verify_cyclic_proof_or_dummy`](crate::recursion::cyclic_recursion)).
    ///
    /// In addition to standard proof verification, this checks that the verifier data
    /// embedded in the proof's public inputs matches the actual verifier data for this
    /// circuit. This prevents an attacker from substituting a valid proof chain built
    /// with a different (but structurally identical) circuit.
    ///
    /// # Security
    ///
    /// Without this check, an attacker could:
    /// 1. Build a malicious circuit with the same structure as the legitimate one
    /// 2. Create a valid proof chain using their circuit
    /// 3. Present it as a proof for the legitimate circuit
    ///
    /// The embedded verifier data check ensures the proof was actually generated for
    /// this specific circuit.
    pub fn verify_cyclic(&self, proof_with_pis: ProofWithPublicInputs<F, C, D>) -> Result<()>
    where
        C::Hasher: AlgebraicHasher<F>,
    {
        self.verify(proof_with_pis.clone())?;
        crate::recursion::cyclic_recursion::check_cyclic_proof_verifier_data(
            &proof_with_pis,
            &self.verifier_only,
            &self.common,
        )
    }

    pub fn verify_compressed(
        &self,
        compressed_proof_with_pis: CompressedProofWithPublicInputs<F, C, D>,
    ) -> Result<()> {
        compressed_proof_with_pis.verify(&self.verifier_only, &self.common)
    }
}

/// Circuit data required by the prover, but not the verifier.
#[derive(Eq, PartialEq, Debug)]
pub struct ProverOnlyCircuitData<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub generators: Vec<WitnessGeneratorRef<F, D>>,
    /// Generator indices (within the `Vec` above), indexed by the representative of each target
    /// they watch.
    pub generator_indices_by_watches: BTreeMap<usize, Vec<usize>>,
    /// Commitments to the constants polynomials and sigma polynomials.
    pub constants_sigmas_commitment: PolynomialBatch<F, C, D>,
    /// The transpose of the list of sigma polynomials.
    pub sigmas: Vec<Vec<F>>,
    /// Subgroup of order `degree`.
    pub subgroup: Vec<F>,
    /// Targets to be made public.
    pub public_inputs: Vec<Target>,
    /// A map from each `Target`'s index to the index of its representative in the disjoint-set
    /// forest.
    pub representative_map: Vec<usize>,
    /// Pre-computed roots for faster FFT.
    pub fft_root_table: Option<FftRootTable<F>>,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash,
    ///The concrete placement of the lookup gates for each lookup table index.
    pub lookup_rows: Vec<LookupWire>,
    /// A vector of (looking_in, looking_out) pairs for each lookup table index.
    pub lut_to_lookups: Vec<Lookup>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProverOnlyCircuitData<F, C, D>
{
    pub fn check_lookup_metadata(&self, common_data: &CommonCircuitData<F, D>) -> Result<()> {
        let degree = 1usize
            .checked_shl(common_data.degree_bits().try_into().unwrap_or(usize::BITS))
            .ok_or_else(|| anyhow::anyhow!("lookup metadata degree bits exceed usize width"))?;
        let num_wires = common_data.config.num_wires;
        let num_luts = common_data.luts.len();

        if num_luts == 0 {
            ensure!(
                self.lookup_rows.is_empty(),
                "lookup rows present but common data has no lookup tables"
            );
            ensure!(
                self.lut_to_lookups.is_empty(),
                "lookup targets present but common data has no lookup tables"
            );
            return Ok(());
        }

        ensure!(
            common_data.num_lookup_polys != 0,
            "lookup tables require lookup polynomials"
        );
        ensure!(
            common_data.num_lookup_selectors != 0,
            "lookup tables require lookup selectors"
        );
        ensure!(
            self.lookup_rows.len() == num_luts,
            "lookup row count {} does not match lookup table count {}",
            self.lookup_rows.len(),
            num_luts
        );
        ensure!(
            self.lut_to_lookups.len() == num_luts,
            "lookup target count {} does not match lookup table count {}",
            self.lut_to_lookups.len(),
            num_luts
        );

        let num_lookup_slots = LookupGate::num_slots(&common_data.config);
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
        ensure!(
            num_lookup_slots != 0,
            "lookup gate capacity must be non-zero"
        );
        ensure!(
            num_lut_slots != 0,
            "lookup table gate capacity must be non-zero"
        );

        for (lut_index, lookup_wire) in self.lookup_rows.iter().enumerate() {
            let LookupWire {
                last_lu_gate,
                last_lut_gate,
                first_lut_gate,
            } = *lookup_wire;
            let lut = &common_data.luts[lut_index];
            let lookups = &self.lut_to_lookups[lut_index];

            ensure!(!lut.is_empty(), "lookup table {lut_index} is empty");
            ensure!(
                !lookups.is_empty(),
                "lookup table {lut_index} has no lookup targets"
            );
            ensure!(
                last_lu_gate < last_lut_gate,
                "lookup table {lut_index} has empty or inverted lookup rows"
            );
            ensure!(
                last_lut_gate <= first_lut_gate,
                "lookup table {lut_index} has empty or inverted table rows"
            );
            ensure!(
                first_lut_gate
                    .checked_add(1)
                    .is_some_and(|next_row| next_row < degree),
                "lookup table {lut_index} table rows exceed trace degree"
            );

            let actual_lookup_rows = last_lut_gate - last_lu_gate;
            let expected_lookup_rows = lookups.len().div_ceil(num_lookup_slots);
            ensure!(
                actual_lookup_rows == expected_lookup_rows,
                "lookup table {lut_index} row count {} does not match {} lookup targets",
                actual_lookup_rows,
                lookups.len()
            );

            let actual_lut_rows = first_lut_gate - last_lut_gate + 1;
            let expected_lut_rows = lut.len().div_ceil(num_lut_slots);
            ensure!(
                actual_lut_rows == expected_lut_rows,
                "lookup table {lut_index} table row count {} does not match {} table entries",
                actual_lut_rows,
                lut.len()
            );

            for &(input, output) in lookups {
                validate_prover_target(input, num_wires, degree, self.representative_map.len())?;
                validate_prover_target(output, num_wires, degree, self.representative_map.len())?;
            }
        }

        Ok(())
    }

    pub fn to_bytes(
        &self,
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_prover_only_circuit_data(self, generator_serializer, common_data)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: &[u8],
        generator_serializer: &dyn WitnessGeneratorSerializer<F, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(bytes);
        buffer.read_prover_only_circuit_data(generator_serializer, common_data)
    }
}

fn validate_prover_target(
    target: Target,
    num_wires: usize,
    degree: usize,
    representative_len: usize,
) -> Result<()> {
    let index = match target {
        Target::Wire(wire) => {
            ensure!(wire.row < degree, "wire target row is outside the trace");
            ensure!(
                wire.column < num_wires,
                "wire target column is outside the trace"
            );
            wire.row
                .checked_mul(num_wires)
                .and_then(|row_offset| row_offset.checked_add(wire.column))
        }
        Target::VirtualTarget { index } => degree
            .checked_mul(num_wires)
            .and_then(|wire_targets| wire_targets.checked_add(index)),
    }
    .ok_or_else(|| anyhow::anyhow!("prover target index overflow"))?;

    ensure!(
        index < representative_len,
        "prover target index is outside the representative map"
    );

    Ok(())
}

/// Circuit data required by the verifier, but not the prover.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct VerifierOnlyCircuitData<C: GenericConfig<D>, const D: usize> {
    /// A commitment to each constant polynomial and each permutation polynomial.
    pub constants_sigmas_cap: MerkleCap<C::F, C::Hasher>,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
}

impl<C: GenericConfig<D>, const D: usize> VerifierOnlyCircuitData<C, D> {
    pub fn to_bytes(&self) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_verifier_only_circuit_data(self)?;
        Ok(buffer)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> IoResult<Self> {
        let mut buffer = Buffer::new(&bytes);
        buffer.read_verifier_only_circuit_data()
    }
}

/// Circuit data required by both the prover and the verifier.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommonCircuitData<F: RichField + Extendable<D>, const D: usize> {
    pub config: CircuitConfig,

    /// Trace degree bits of the underlying PLONK circuit.
    pub trace_degree_bits: usize,

    pub fri_params: FriParams,

    /// Shared public degree bits for the initial phase-1 FRI oracle commitments in PolyFri mode.
    /// This may be larger than the trace degree if logical masking lifts the committed polynomial
    /// degree above the native trace domain.
    pub public_initial_degree_bits: usize,

    /// Raw-vs-logical oracle layout metadata used by prover-private helpers and tests.
    ///
    /// Public FRI instance generation now emits only logical masked openings, so this metadata is
    /// no longer consulted when building the public proof shape.
    pub fri_oracle_layouts: Vec<FriOracleLayout>,

    /// The types of gates used in this circuit, along with their prefixes.
    pub gates: Vec<GateRef<F, D>>,

    /// Information on the circuit's selector polynomials.
    pub selectors_info: SelectorsInfo,

    /// The degree of the PLONK quotient polynomial.
    pub quotient_degree_factor: usize,

    /// The largest number of constraints imposed by any gate.
    pub num_gate_constraints: usize,

    /// The number of constant wires.
    pub num_constants: usize,

    pub num_public_inputs: usize,

    /// The `{k_i}` valued used in `S_ID_i` in Plonk's permutation argument.
    pub k_is: Vec<F>,

    /// The number of partial products needed to compute the `Z` polynomials.
    pub num_partial_products: usize,

    /// The number of lookup polynomials.
    pub num_lookup_polys: usize,

    /// The number of lookup selectors.
    pub num_lookup_selectors: usize,

    /// The stored lookup tables.
    pub luts: Vec<LookupTable>,
}

impl<F: RichField + Extendable<D>, const D: usize> CommonCircuitData<F, D> {
    /// Validate invariants required by the prover.
    ///
    /// This checks that degree parameters are consistent and within bounds.
    pub fn check_valid(&self) -> Result<(), &'static str> {
        self.config.check_valid()?;
        self.config.check_reducing_widths::<D>()?;
        self.config.check_extension_gate_widths::<D>()?;
        if !self.luts.is_empty() || self.num_lookup_polys != 0 || self.num_lookup_selectors != 0 {
            self.config.check_lookup_widths()?;
        }
        self.fri_params
            .check_valid()
            .map_err(|_| "invalid FRI params")?;
        self.config
            .fri_config
            .required_proof_of_work_leading_zeros::<F>()
            .map_err(|_| "invalid FRI proof-of-work bits")?;

        // Quotient degree must fit within FRI rate.
        let quotient_degree_bits = log2_ceil(self.quotient_degree_factor);
        if quotient_degree_bits > self.config.fri_config.rate_bits {
            return Err("quotient_degree_factor exceeds FRI rate_bits");
        }

        // Public initial degree must be at least as large as trace degree.
        if self.public_initial_degree_bits < self.trace_degree_bits {
            return Err("public_initial_degree_bits must be >= trace_degree_bits");
        }
        if self.public_initial_degree_bits != self.fri_params.degree_bits {
            return Err("public_initial_degree_bits must match FRI degree_bits");
        }
        self.check_poly_fri_proving_params()?;

        // All lookup tables must be non-empty.
        if self.luts.iter().any(|lut| lut.is_empty()) {
            return Err("lookup table is empty");
        }

        check_gate_id_collisions(&self.gates)?;

        Ok(())
    }

    fn checked_degree_from_bits(bits: usize) -> Result<usize, &'static str> {
        1usize
            .checked_shl(bits.try_into().unwrap_or(usize::BITS))
            .ok_or("degree bits exceed usize width")
    }

    fn checked_public_initial_degree_bits(
        trace_degree: usize,
        max_mask_degree: usize,
    ) -> Result<usize, &'static str> {
        let required_degree = trace_degree
            .checked_add(max_mask_degree)
            .and_then(|degree| degree.checked_add(1))
            .ok_or("PolyFri public initial degree overflow")?;
        let rounded = required_degree
            .checked_next_power_of_two()
            .ok_or("PolyFri public initial degree overflow")?;
        Ok(rounded.trailing_zeros() as usize)
    }

    fn check_mask_degree(
        knob_name: &'static str,
        mask_degree: usize,
        split_degree: usize,
    ) -> Result<(), &'static str> {
        if mask_degree >= split_degree {
            return match knob_name {
                "wire_mask_degree" => Err("wire_mask_degree must be less than trace degree"),
                "z_mask_degree" => Err("z_mask_degree must be less than trace degree"),
                "fri_batch_mask_degree" => {
                    Err("fri_batch_mask_degree must be less than FRI batch-mask chunk degree")
                }
                _ => Err("PolyFri mask degree exceeds split degree"),
            };
        }
        Ok(())
    }

    fn check_poly_fri_proving_params(&self) -> Result<(), &'static str> {
        let ZkMode::PolyFri(poly_fri) = &self.config.zk_config.mode else {
            return Ok(());
        };

        let trace_degree = Self::checked_degree_from_bits(self.trace_degree_bits)?;
        Self::check_mask_degree("wire_mask_degree", poly_fri.wire_mask_degree, trace_degree)?;
        Self::check_mask_degree("z_mask_degree", poly_fri.z_mask_degree, trace_degree)?;

        let max_phase1_mask_degree = poly_fri.wire_mask_degree.max(poly_fri.z_mask_degree);
        let expected_public_initial_degree_bits =
            Self::checked_public_initial_degree_bits(trace_degree, max_phase1_mask_degree)?;
        if self.public_initial_degree_bits != expected_public_initial_degree_bits {
            return Err("invalid PolyFri public_initial_degree_bits");
        }

        let batch_mask_chunk_degree = match self.fri_params.batch_mask_layout() {
            crate::fri::FriFinalPolyLayout::Single => {
                Self::checked_degree_from_bits(self.fri_params.degree_bits)?
            }
            crate::fri::FriFinalPolyLayout::Split {
                chunk_degree_bits, ..
            } => Self::checked_degree_from_bits(chunk_degree_bits)?,
        };
        Self::check_mask_degree(
            "fri_batch_mask_degree",
            poly_fri.fri_batch_mask_degree,
            batch_mask_chunk_degree,
        )?;

        if self.config.max_quotient_degree_factor < 2 {
            return Err("max_quotient_degree_factor too small for PolyFri");
        }
        if self.num_lookup_polys != 0 && self.config.max_quotient_degree_factor < 3 {
            return Err("max_quotient_degree_factor too small for PolyFri lookups");
        }

        Ok(())
    }

    pub fn to_bytes(&self, gate_serializer: &dyn GateSerializer<F, D>) -> IoResult<Vec<u8>> {
        let mut buffer = Vec::new();
        buffer.write_common_circuit_data(self, gate_serializer)?;
        Ok(buffer)
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        gate_serializer: &dyn GateSerializer<F, D>,
    ) -> IoResult<Self> {
        let mut buffer = Buffer::new(&bytes);
        buffer.read_common_circuit_data(gate_serializer)
    }

    /// Trace degree bits used by the PLONK identity checks and subgroup arithmetic.
    pub const fn degree_bits(&self) -> usize {
        self.trace_degree_bits
    }

    /// Degree bits for the public initial FRI codeword used by masked phase-1 oracles.
    pub const fn public_initial_degree_bits(&self) -> usize {
        self.public_initial_degree_bits
    }

    /// Degree of the public initial FRI codeword used by masked phase-1 oracles.
    pub const fn public_initial_degree(&self) -> usize {
        1 << self.public_initial_degree_bits()
    }

    /// LDE size of the public initial codeword used by masked phase-1 oracles.
    pub const fn public_initial_lde_size(&self) -> usize {
        self.public_initial_degree() << self.config.fri_config.rate_bits
    }

    pub const fn degree(&self) -> usize {
        1 << self.degree_bits()
    }

    pub const fn lde_size(&self) -> usize {
        self.fri_params.lde_size()
    }

    pub fn lde_generator(&self) -> F {
        F::primitive_root_of_unity(self.degree_bits() + self.config.fri_config.rate_bits)
    }

    pub fn constraint_degree(&self) -> usize {
        self.gates
            .iter()
            .map(|g| g.0.degree())
            .max()
            .expect("No gates?")
    }

    pub const fn quotient_degree(&self) -> usize {
        self.quotient_degree_factor * self.degree()
    }

    /// PolyFri masks the permutation accumulator family, so the partial-product chunk size drops
    /// by one to keep the masked accumulator checks within the existing quotient degree bound.
    pub fn permutation_partial_product_degree(&self) -> usize {
        if self.config.uses_poly_fri_zk() {
            self.quotient_degree_factor - 1
        } else {
            self.quotient_degree_factor
        }
    }

    /// Lookup running-sum accumulators already consume one degree in the filtered constraints.
    /// PolyFri masking adds one more effective degree, so their chunk size drops by one more to
    /// stay within the unchanged quotient bound.
    pub fn lookup_accumulator_degree(&self) -> usize {
        if self.config.uses_poly_fri_zk() {
            self.quotient_degree_factor - 2
        } else {
            self.quotient_degree_factor - 1
        }
    }

    /// Range of the constants polynomials in the `constants_sigmas_commitment`.
    pub const fn constants_range(&self) -> Range<usize> {
        0..self.num_constants
    }

    /// Range of the sigma polynomials in the `constants_sigmas_commitment`.
    pub const fn sigmas_range(&self) -> Range<usize> {
        self.num_constants..self.num_constants + self.config.num_routed_wires
    }

    /// Range of the `z`s polynomials in the `zs_partial_products_commitment`.
    pub const fn zs_range(&self) -> Range<usize> {
        0..self.config.num_challenges
    }

    /// Range of the partial products polynomials in the `zs_partial_products_lookup_commitment`.
    pub const fn partial_products_range(&self) -> Range<usize> {
        self.config.num_challenges..(self.num_partial_products + 1) * self.config.num_challenges
    }

    /// Range of lookup polynomials in the `zs_partial_products_lookup_commitment`.
    pub const fn lookup_range(&self) -> RangeFrom<usize> {
        self.num_zs_partial_products_polys()..
    }

    /// Range of lookup polynomials needed for evaluation at `g * zeta`.
    pub const fn next_lookup_range(&self, i: usize) -> Range<usize> {
        self.num_zs_partial_products_polys() + i * self.num_lookup_polys
            ..self.num_zs_partial_products_polys() + i * self.num_lookup_polys + 2
    }

    pub(crate) fn get_fri_instance(&self, zeta: F::Extension) -> FriInstanceInfo<F, D> {
        // All polynomials are opened at zeta.
        let zeta_batch = FriBatchInfo {
            point: zeta,
            openings: self.fri_all_openings(),
        };

        // The Z polynomials are also opened at g * zeta.
        let g = F::Extension::primitive_root_of_unity(self.degree_bits());
        let zeta_next = g * zeta;
        let zeta_next_batch = FriBatchInfo {
            point: zeta_next,
            openings: self.fri_next_batch_openings(),
        };

        let openings = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfo {
            oracles: self.fri_oracles(),
            batches: openings,
        }
    }

    pub(crate) fn get_fri_instance_target(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        zeta: ExtensionTarget<D>,
    ) -> FriInstanceInfoTarget<F, D> {
        // All polynomials are opened at zeta.
        let zeta_batch = FriBatchInfoTarget {
            point: zeta,
            openings: self.fri_all_openings(),
        };

        // The Z polynomials are also opened at g * zeta.
        let g = F::primitive_root_of_unity(self.degree_bits());
        let zeta_next = builder.mul_const_extension(g, zeta);
        let zeta_next_batch = FriBatchInfoTarget {
            point: zeta_next,
            openings: self.fri_next_batch_openings(),
        };

        let openings = vec![zeta_batch, zeta_next_batch];
        FriInstanceInfoTarget {
            oracles: self.fri_oracles(),
            batches: openings,
        }
    }

    fn fri_oracles(&self) -> Vec<FriOracleInfo> {
        [
            PlonkOracle::CONSTANTS_SIGMAS,
            PlonkOracle::WIRES,
            PlonkOracle::ZS_PARTIAL_PRODUCTS,
            PlonkOracle::QUOTIENT,
        ]
        .into_iter()
        .map(|oracle| FriOracleInfo {
            num_polys: self.fri_oracle_layouts[oracle.index].logical_polys,
            blinding: oracle.blinding,
        })
        .collect()
    }

    pub(crate) const fn num_preprocessed_polys(&self) -> usize {
        self.sigmas_range().end
    }

    fn fri_oracle_openings<I>(
        &self,
        oracle: PlonkOracle,
        logical_indices: I,
    ) -> Vec<FriOpeningExpression<F, D>>
    where
        I: IntoIterator<Item = usize>,
    {
        logical_indices
            .into_iter()
            .map(|logical_index| {
                FriOpeningExpression::raw(FriPolynomialInfo {
                    oracle_index: oracle.index,
                    polynomial_index: logical_index,
                })
            })
            .collect()
    }

    fn fri_preprocessed_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        self.fri_oracle_openings(
            PlonkOracle::CONSTANTS_SIGMAS,
            0..self.num_preprocessed_polys(),
        )
    }

    fn fri_wire_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        self.fri_oracle_openings(PlonkOracle::WIRES, 0..self.config.num_wires)
    }

    fn fri_zs_partial_products_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        self.fri_oracle_openings(
            PlonkOracle::ZS_PARTIAL_PRODUCTS,
            0..self.num_zs_partial_products_polys(),
        )
    }

    pub(crate) const fn num_zs_partial_products_polys(&self) -> usize {
        self.config.num_challenges * (1 + self.num_partial_products)
    }

    /// Returns the total number of lookup polynomials.
    pub(crate) const fn num_all_lookup_polys(&self) -> usize {
        self.config.num_challenges * self.num_lookup_polys
    }

    fn fri_zs_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        self.fri_oracle_openings(PlonkOracle::ZS_PARTIAL_PRODUCTS, self.zs_range())
    }

    /// Returns polynomials that require evaluation at `zeta` and `g * zeta`.
    fn fri_next_batch_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        [self.fri_zs_openings(), self.fri_lookup_openings()].concat()
    }

    fn fri_quotient_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        self.fri_oracle_openings(PlonkOracle::QUOTIENT, 0..self.num_quotient_polys())
    }

    /// Returns the information for lookup polynomials, i.e. the index within the oracle and the indices of the polynomials within the commitment.
    fn fri_lookup_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        self.fri_oracle_openings(
            PlonkOracle::ZS_PARTIAL_PRODUCTS,
            self.num_zs_partial_products_polys()
                ..self.num_zs_partial_products_polys() + self.num_all_lookup_polys(),
        )
    }

    pub(crate) const fn num_quotient_polys(&self) -> usize {
        self.config.num_challenges * self.quotient_degree_factor
    }

    fn fri_all_openings(&self) -> Vec<FriOpeningExpression<F, D>> {
        [
            self.fri_preprocessed_openings(),
            self.fri_wire_openings(),
            self.fri_zs_partial_products_openings(),
            self.fri_quotient_openings(),
            self.fri_lookup_openings(),
        ]
        .concat()
    }
}

/// The `Target` version of `VerifierCircuitData`, for use inside recursive circuits. Note that this
/// is intentionally missing certain fields, such as `CircuitConfig`, because we support only a
/// limited form of dynamic inner circuits. We can't practically make things like the wire count
/// dynamic, at least not without setting a maximum wire count and paying for the worst case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifierCircuitTarget {
    /// A commitment to each constant polynomial and each permutation polynomial.
    pub constants_sigmas_cap: MerkleCapTarget,
    /// A digest of the "circuit" (i.e. the instance, minus public inputs), which can be used to
    /// seed Fiat-Shamir.
    pub circuit_digest: HashOutTarget,
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::sync::Arc;
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use itertools::Itertools;
    use qp_plonky2_core::ZkMode;

    use super::{CircuitConfig, CommonCircuitData};
    use crate::field::extension::Extendable;
    use crate::field::types::Field;
    use crate::fri::structure::{FriCoefficient, FriOpeningExpression, FriOracleRepresentation};
    use crate::fri::FriFinalPolyLayout;
    use crate::gates::lookup::LookupGate;
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::hash::hash_types::RichField;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::util::partial_products::num_partial_products;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    fn permutation_effective_degree(uses_poly_fri: bool, chunk_degree: usize) -> usize {
        chunk_degree + usize::from(uses_poly_fri)
    }

    fn lookup_effective_degree(uses_poly_fri: bool, chunk_degree: usize) -> usize {
        chunk_degree + 1 + usize::from(uses_poly_fri)
    }

    fn build_common(config: CircuitConfig) -> CommonCircuitData<F, D> {
        let mut builder = CircuitBuilder::<F, D>::new(config);
        builder.add_gate(NoopGate, vec![]);
        builder.build::<C>().common
    }

    fn build_lookup_common(config: CircuitConfig) -> CommonCircuitData<F, D> {
        let table: LookupTable = Arc::new((0..4).zip_eq(1..5).collect());
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let input = builder.constant(F::ONE);
        let table_index = builder.add_lookup_table_from_pairs(table);
        let _ = builder.add_lookup_from_index(input, table_index);
        builder.build::<C>().common
    }

    fn assert_raw_opening_expression<F: RichField + Extendable<D>, const D: usize>(
        expression: &FriOpeningExpression<F, D>,
    ) {
        assert_eq!(
            expression.terms.len(),
            1,
            "PolyFri public openings must stay in logical form"
        );
        assert!(matches!(
            &expression.terms[0].coefficient,
            FriCoefficient::One
        ));
    }

    #[test]
    fn permutation_partial_product_degree_disabled_boundary() {
        let common = build_common(CircuitConfig::standard_recursion_config());
        let degree = common.permutation_partial_product_degree();

        assert_eq!(degree, common.quotient_degree_factor);
        assert_eq!(
            common.num_partial_products,
            num_partial_products(common.config.num_routed_wires, degree)
        );
        assert_eq!(
            permutation_effective_degree(false, degree),
            common.quotient_degree_factor,
        );
        assert!(
            permutation_effective_degree(false, degree + 1) > common.quotient_degree_factor,
            "raising the permutation chunk degree by one would exceed the quotient degree bound",
        );
    }

    #[test]
    fn permutation_partial_product_degree_polyfri_boundary() {
        let common = build_common(CircuitConfig::standard_recursion_polyfri_zk_config());
        let degree = common.permutation_partial_product_degree();

        assert_eq!(degree, common.quotient_degree_factor - 1);
        assert_eq!(
            common.num_partial_products,
            num_partial_products(common.config.num_routed_wires, degree)
        );
        assert_eq!(
            permutation_effective_degree(true, degree),
            common.quotient_degree_factor,
        );
        assert!(
            permutation_effective_degree(true, degree + 1) > common.quotient_degree_factor,
            "raising the masked permutation chunk degree by one would exceed the quotient degree bound",
        );
    }

    #[test]
    fn lookup_accumulator_degree_disabled_boundary() {
        let common = build_lookup_common(CircuitConfig::standard_recursion_config());
        let degree = common.lookup_accumulator_degree();

        assert!(common.num_lookup_polys > 0);
        assert_eq!(degree, common.quotient_degree_factor - 1);
        assert_eq!(
            common.num_lookup_polys,
            LookupGate::num_slots(&common.config).div_ceil(degree) + 1,
        );
        assert_eq!(
            lookup_effective_degree(false, degree),
            common.quotient_degree_factor
        );
        assert!(
            lookup_effective_degree(false, degree + 1) > common.quotient_degree_factor,
            "raising the lookup accumulator degree by one would exceed the quotient degree bound",
        );
    }

    #[test]
    fn lookup_accumulator_degree_polyfri_boundary() {
        let common = build_lookup_common(CircuitConfig::standard_recursion_polyfri_zk_config());
        let degree = common.lookup_accumulator_degree();

        assert!(common.num_lookup_polys > 0);
        assert_eq!(degree, common.quotient_degree_factor - 2);
        assert_eq!(
            common.num_lookup_polys,
            LookupGate::num_slots(&common.config).div_ceil(degree) + 1,
        );
        assert_eq!(
            lookup_effective_degree(true, degree),
            common.quotient_degree_factor
        );
        assert!(
            lookup_effective_degree(true, degree + 1) > common.quotient_degree_factor,
            "raising the masked lookup accumulator degree by one would exceed the quotient degree bound",
        );
    }

    #[test]
    #[should_panic(expected = "Invalid PolyFri config: `wire_mask_degree`")]
    fn polyfri_wire_mask_degree_is_validated_up_front() {
        let mut config = CircuitConfig::standard_recursion_polyfri_zk_config();
        if let ZkMode::PolyFri(poly_fri) = &mut config.zk_config.mode {
            poly_fri.wire_mask_degree = usize::MAX;
        }
        let _ = build_common(config);
    }

    #[test]
    #[should_panic(expected = "Invalid PolyFri config: `fri_batch_mask_degree`")]
    fn polyfri_batch_mask_degree_is_validated_up_front() {
        let mut config = CircuitConfig::standard_recursion_polyfri_zk_config();
        if let ZkMode::PolyFri(poly_fri) = &mut config.zk_config.mode {
            poly_fri.fri_batch_mask_degree = usize::MAX;
        }
        let _ = build_common(config);
    }

    #[test]
    #[should_panic(
        expected = "Invalid PolyFri config: `max_quotient_degree_factor` must be at least 2"
    )]
    fn polyfri_permutation_budget_is_validated_up_front() {
        let mut config = CircuitConfig::standard_recursion_polyfri_zk_config();
        config.max_quotient_degree_factor = 1;
        let _ = build_common(config);
    }

    #[test]
    #[should_panic(
        expected = "Invalid PolyFri config: `max_quotient_degree_factor` must be at least 3 when lookups are enabled"
    )]
    fn polyfri_lookup_budget_is_validated_up_front() {
        let mut config = CircuitConfig::standard_recursion_polyfri_zk_config();
        config.max_quotient_degree_factor = 2;
        let _ = build_lookup_common(config);
    }

    #[test]
    fn row_blinding_uses_legacy_degree_budgets() {
        let common = build_lookup_common(CircuitConfig::standard_recursion_zk_config());

        assert_eq!(
            common.permutation_partial_product_degree(),
            common.quotient_degree_factor
        );
        assert_eq!(
            common.lookup_accumulator_degree(),
            common.quotient_degree_factor - 1
        );
    }

    #[test]
    fn row_blinding_keeps_raw_layouts_but_adds_builder_rows() {
        let disabled = build_common(CircuitConfig::standard_recursion_config());
        let row_blinding = build_common(CircuitConfig::standard_recursion_zk_config());
        let polyfri = build_common(CircuitConfig::standard_recursion_polyfri_zk_config());

        assert_eq!(disabled.degree(), polyfri.degree());
        assert!(
            row_blinding.degree() > disabled.degree(),
            "legacy row blinding should append witness rows before final padding",
        );

        assert_eq!(
            row_blinding.fri_oracle_layouts[1].representation,
            FriOracleRepresentation::Raw,
        );
        assert_eq!(
            row_blinding.fri_oracle_layouts[2].representation,
            FriOracleRepresentation::Raw,
        );
        assert_eq!(row_blinding.fri_params.batch_masking, None);
        assert_eq!(
            row_blinding.fri_params.final_poly_layout,
            FriFinalPolyLayout::Single
        );

        assert!(matches!(
            polyfri.fri_oracle_layouts[1].representation,
            FriOracleRepresentation::Raw
        ));
        assert!(matches!(
            polyfri.fri_oracle_layouts[2].representation,
            FriOracleRepresentation::Raw
        ));
        assert_eq!(
            polyfri.fri_oracle_layouts[1].raw_polys, polyfri.fri_oracle_layouts[1].logical_polys,
            "PolyFri wires must expose only logical polynomials"
        );
        assert_eq!(
            polyfri.fri_oracle_layouts[2].raw_polys, polyfri.fri_oracle_layouts[2].logical_polys,
            "PolyFri permutation/lookup oracles must expose only logical polynomials"
        );
        assert!(polyfri.fri_params.batch_masking.is_some());
        assert!(matches!(
            polyfri.fri_params.final_poly_layout,
            FriFinalPolyLayout::Split { .. }
        ));
    }

    #[test]
    fn polyfri_fri_instance_uses_raw_logical_openings() {
        let common = build_lookup_common(CircuitConfig::standard_recursion_polyfri_zk_config());
        let all_openings = common.fri_all_openings();
        let wire_openings = common.fri_wire_openings();
        let zs_openings = common.fri_zs_partial_products_openings();

        assert!(matches!(
            common.fri_oracle_layouts[1].representation,
            FriOracleRepresentation::Raw
        ));
        assert!(matches!(
            common.fri_oracle_layouts[2].representation,
            FriOracleRepresentation::Raw
        ));

        for opening in all_openings
            .iter()
            .chain(wire_openings.iter())
            .chain(zs_openings.iter())
        {
            assert_raw_opening_expression(opening);
        }
    }
}
