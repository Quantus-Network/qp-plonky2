#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

use anyhow::{anyhow, ensure, Result};
use hashbrown::HashMap;
use plonky2_field::extension::Extendable;
use plonky2_field::polynomial::PolynomialCoeffs;

use crate::fri::proof::{FriFinalPolys, FriFinalPolysTarget, FriProof, FriProofTarget};
use crate::fri::{FriConfig, FriFinalPolyLayout, FriParams, FriReductionStrategy};
use crate::gates::noop::NoopGate;
use crate::gates::selectors::SelectorsInfo;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::generator::{GeneratedValues, SimpleGenerator};
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitData, VerifierCircuitTarget,
    VerifierOnlyCircuitData,
};
use crate::plonk::config::{AlgebraicHasher, GenericConfig, GenericHashOut, Hasher};
use crate::plonk::proof::{
    OpeningSet, OpeningSetTarget, Proof, ProofTarget, ProofWithPublicInputs,
    ProofWithPublicInputsTarget,
};
use crate::util::serialization::{Buffer, DefaultGateSerializer, IoResult, Read, Write};

/// Creates a dummy proof which is suitable for use as a base proof in a cyclic recursion tree.
/// Such a base proof will not actually be verified, so most of its data is arbitrary. However, its
/// public inputs which encode the cyclic verification key must be set properly, and this method
/// takes care of that. It also allows the user to specify any other public inputs which should be
/// set in this base proof.
pub fn cyclic_base_proof<F, C, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    verifier_data: &VerifierOnlyCircuitData<C, D>,
    nonzero_public_inputs: HashMap<usize, F>,
) -> ProofWithPublicInputs<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: AlgebraicHasher<C::F>,
    C::InnerHasher: AlgebraicHasher<F>,
{
    try_cyclic_base_proof(common_data, verifier_data, nonzero_public_inputs)
        .expect("invalid cyclic recursion metadata")
}

/// Fallible variant of [`cyclic_base_proof`] for untrusted or deserialized recursion metadata.
pub fn try_cyclic_base_proof<F, C, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    verifier_data: &VerifierOnlyCircuitData<C, D>,
    mut nonzero_public_inputs: HashMap<usize, F>,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: AlgebraicHasher<C::F>,
    C::InnerHasher: AlgebraicHasher<F>,
{
    let (start_vk_pis, cap_elements) = cyclic_verifier_public_input_layout(common_data)?;
    ensure!(
        verifier_data.constants_sigmas_cap.0.len() == cap_elements,
        "verifier cap length does not match cyclic metadata"
    );

    // Add the cyclic verifier data public inputs.
    nonzero_public_inputs.extend((start_vk_pis..).zip(verifier_data.circuit_digest.elements));
    for i in 0..cap_elements {
        let start = start_vk_pis + 4 + 4 * i;
        nonzero_public_inputs
            .extend((start..).zip(verifier_data.constants_sigmas_cap.0[i].elements));
    }

    // TODO: A bit wasteful to build a dummy circuit here. We could potentially use a proof that
    // just consists of zeros, apart from public inputs.
    let dummy = try_dummy_circuit::<F, C, D>(common_data)?;
    dummy_proof::<F, C, D>(&dummy, nonzero_public_inputs)
}

/// Generate a proof for a dummy circuit. The `public_inputs` parameter let the caller specify
/// certain public inputs (identified by their indices) which should be given specific values.
/// The rest will default to zero.
pub fn dummy_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    circuit: &CircuitData<F, C, D>,
    nonzero_public_inputs: HashMap<usize, F>,
) -> anyhow::Result<ProofWithPublicInputs<F, C, D>>
where
{
    let mut pw = PartialWitness::new();
    for i in 0..circuit.common.num_public_inputs {
        let pi = nonzero_public_inputs.get(&i).copied().unwrap_or_default();
        pw.set_target(circuit.prover_only.public_inputs[i], pi)?;
    }
    circuit.prove(pw)
}

/// Generate a circuit matching a given `CommonCircuitData`.
pub fn dummy_circuit<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
) -> CircuitData<F, C, D>
where
    C::InnerHasher: crate::plonk::config::AlgebraicHasher<F>,
{
    try_dummy_circuit::<F, C, D>(common_data).expect("invalid dummy circuit metadata")
}

/// Fallible variant of [`dummy_circuit`] for untrusted or deserialized recursion metadata.
pub fn try_dummy_circuit<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
) -> Result<CircuitData<F, C, D>>
where
    C::InnerHasher: crate::plonk::config::AlgebraicHasher<F>,
{
    common_data
        .check_valid()
        .map_err(|err| anyhow!("invalid common circuit data: {err}"))?;
    let config = common_data.config.clone();

    // Number of `NoopGate`s to add to get a circuit of size `degree` in the end.
    // Need to account for public input hashing, a `PublicInputGate` and a `ConstantGate`.
    let degree = checked_trace_degree(common_data)?;
    let public_input_gate_count = common_data.num_public_inputs.div_ceil(8);
    let non_noop_gate_count = public_input_gate_count
        .checked_add(2)
        .ok_or_else(|| anyhow!("dummy circuit gate count overflow"))?;
    let num_noop_gate = degree
        .checked_sub(non_noop_gate_count)
        .ok_or_else(|| anyhow!("dummy circuit degree is too small for public-input metadata"))?;

    let mut builder = CircuitBuilder::<F, D>::new(config);
    for _ in 0..num_noop_gate {
        builder.add_gate(NoopGate, vec![]);
    }
    for gate in &common_data.gates {
        builder.add_gate_to_gate_set(gate.clone());
    }
    for _ in 0..common_data.num_public_inputs {
        builder.add_virtual_public_input();
    }

    let circuit = builder.build::<C>();
    ensure!(
        &circuit.common == common_data,
        "dummy circuit metadata did not round-trip"
    );
    Ok(circuit)
}

fn checked_trace_degree<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
) -> Result<usize> {
    ensure!(
        common_data.trace_degree_bits < usize::BITS as usize,
        "trace degree bits exceed usize capacity"
    );
    1usize
        .checked_shl(common_data.trace_degree_bits as u32)
        .ok_or_else(|| anyhow!("trace degree overflow"))
}

fn checked_cap_elements(cap_height: usize) -> Result<usize> {
    ensure!(
        cap_height < usize::BITS as usize,
        "FRI cap height exceeds usize capacity"
    );
    1usize
        .checked_shl(cap_height as u32)
        .ok_or_else(|| anyhow!("FRI cap length overflow"))
}

fn cyclic_verifier_public_input_layout<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
) -> Result<(usize, usize)> {
    let cap_elements = checked_cap_elements(common_data.config.fri_config.cap_height)?;
    let cap_public_inputs = cap_elements
        .checked_mul(4)
        .ok_or_else(|| anyhow!("cyclic verifier cap public-input length overflow"))?;
    let verifier_data_inputs = 4usize
        .checked_add(cap_public_inputs)
        .ok_or_else(|| anyhow!("cyclic verifier public-input length overflow"))?;
    let start = common_data
        .num_public_inputs
        .checked_sub(verifier_data_inputs)
        .ok_or_else(|| anyhow!("not enough public inputs for cyclic verifier data"))?;
    Ok((start, cap_elements))
}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub(crate) fn dummy_proof_and_vk<C: GenericConfig<D, F = F> + 'static>(
        &mut self,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<(ProofWithPublicInputsTarget<D>, VerifierCircuitTarget)>
    where
        C::Hasher: AlgebraicHasher<F>,
        C::InnerHasher: AlgebraicHasher<F>,
    {
        let dummy_circuit = try_dummy_circuit::<F, C, D>(common_data)?;
        let dummy_proof_with_pis = dummy_proof::<F, C, D>(&dummy_circuit, HashMap::new())?;
        let dummy_proof_with_pis_target = self.add_virtual_proof_with_pis(common_data);
        let dummy_verifier_data_target =
            self.add_virtual_verifier_data(self.config.fri_config.cap_height);

        self.add_simple_generator(DummyProofGenerator {
            proof_with_pis_target: dummy_proof_with_pis_target.clone(),
            proof_with_pis: dummy_proof_with_pis,
            verifier_data_target: dummy_verifier_data_target.clone(),
            verifier_data: dummy_circuit.verifier_data(),
        });

        Ok((dummy_proof_with_pis_target, dummy_verifier_data_target))
    }

    pub fn dummy_proof_and_constant_vk_no_generator<C: GenericConfig<D, F = F> + 'static>(
        &mut self,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<(ProofWithPublicInputsTarget<D>, VerifierCircuitTarget)>
    where
        C::Hasher: AlgebraicHasher<F>,
        C::InnerHasher: AlgebraicHasher<F>,
    {
        let dummy_circuit = try_dummy_circuit::<F, C, D>(common_data)?;
        let dummy_proof_with_pis_target = self.add_virtual_proof_with_pis(common_data);
        let dummy_verifier_data_target = self.constant_verifier_data(&dummy_circuit.verifier_only);

        Ok((dummy_proof_with_pis_target, dummy_verifier_data_target))
    }
}

#[derive(Debug)]
pub struct DummyProofGenerator<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    pub(crate) proof_with_pis_target: ProofWithPublicInputsTarget<D>,
    pub(crate) proof_with_pis: ProofWithPublicInputs<F, C, D>,
    pub(crate) verifier_data_target: VerifierCircuitTarget,
    pub(crate) verifier_data: VerifierCircuitData<F, C, D>,
}

impl<F, C, const D: usize> Default for DummyProofGenerator<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    fn default() -> Self {
        let proof_with_pis_target = ProofWithPublicInputsTarget {
            proof: ProofTarget {
                wires_cap: MerkleCapTarget(vec![]),
                plonk_zs_partial_products_cap: MerkleCapTarget(vec![]),
                quotient_polys_cap: MerkleCapTarget(vec![]),
                openings: OpeningSetTarget::default(),
                opening_proof: FriProofTarget {
                    commit_phase_merkle_caps: vec![],
                    batch_mask_proof: None,
                    query_round_proofs: vec![],
                    final_polys: FriFinalPolysTarget { chunks: vec![] },
                    pow_witness: Target::default(),
                },
            },
            public_inputs: vec![],
        };

        let proof_with_pis = ProofWithPublicInputs {
            proof: Proof {
                wires_cap: MerkleCap(vec![]),
                plonk_zs_partial_products_cap: MerkleCap(vec![]),
                quotient_polys_cap: MerkleCap(vec![]),
                openings: OpeningSet::default(),
                opening_proof: FriProof {
                    commit_phase_merkle_caps: vec![],
                    batch_mask_proof: None,
                    query_round_proofs: vec![],
                    final_polys: FriFinalPolys::from_single(PolynomialCoeffs { coeffs: vec![] }),
                    pow_witness: F::ZERO,
                },
            },
            public_inputs: vec![],
        };

        let verifier_data_target = VerifierCircuitTarget {
            constants_sigmas_cap: MerkleCapTarget(vec![]),
            circuit_digest: HashOutTarget {
                elements: [Target::default(); 4],
            },
        };

        let verifier_data = VerifierCircuitData {
            common: CommonCircuitData {
                config: CircuitConfig::default(),
                trace_degree_bits: 0,
                fri_params: FriParams {
                    config: FriConfig {
                        rate_bits: 0,
                        cap_height: 0,
                        proof_of_work_bits: 0,
                        reduction_strategy: FriReductionStrategy::MinSize(None),
                        num_query_rounds: 0,
                    },
                    leaf_hiding: false,
                    batch_masking: None,
                    degree_bits: 0,
                    reduction_arity_bits: vec![],
                    final_poly_layout: FriFinalPolyLayout::Single,
                },
                public_initial_degree_bits: 0,
                fri_oracle_layouts: vec![],
                gates: vec![],
                selectors_info: SelectorsInfo {
                    selector_indices: vec![],
                    groups: vec![],
                },
                quotient_degree_factor: 0,
                num_gate_constraints: 0,
                num_constants: 0,
                num_public_inputs: 0,
                k_is: vec![],
                num_partial_products: 0,
                num_lookup_polys: 0,
                num_lookup_selectors: 0,
                luts: vec![],
            },
            verifier_only: VerifierOnlyCircuitData {
                constants_sigmas_cap: MerkleCap(vec![]),
                circuit_digest: <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash::from_bytes(
                    &vec![0; <<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::HASH_SIZE],
                ),
            },
        };

        Self {
            proof_with_pis_target,
            proof_with_pis,
            verifier_data_target,
            verifier_data,
        }
    }
}

impl<F, C, const D: usize> SimpleGenerator<F, D> for DummyProofGenerator<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F> + 'static,
    C::Hasher: AlgebraicHasher<F>,
{
    fn id(&self) -> String {
        "DummyProofGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        out_buffer.set_proof_with_pis_target(&self.proof_with_pis_target, &self.proof_with_pis)?;
        out_buffer.set_verifier_data_target(
            &self.verifier_data_target,
            &self.verifier_data.verifier_only,
        )
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target_verifier_circuit(&self.verifier_data_target)?;
        dst.write_verifier_circuit_data(&self.verifier_data, &DefaultGateSerializer)?;
        dst.write_target_proof_with_public_inputs(&self.proof_with_pis_target)?;
        dst.write_proof_with_public_inputs(&self.proof_with_pis)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let verifier_data_target = src.read_target_verifier_circuit()?;
        let verifier_data = src.read_verifier_circuit_data(&DefaultGateSerializer)?;
        let proof_with_pis_target = src.read_target_proof_with_public_inputs()?;
        let proof_with_pis = src.read_proof_with_public_inputs(&verifier_data.common)?;
        Ok(Self {
            proof_with_pis_target,
            proof_with_pis,
            verifier_data_target,
            verifier_data,
        })
    }
}
