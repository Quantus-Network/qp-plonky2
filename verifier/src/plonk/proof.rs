//! plonky2 proof definition.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use qp_plonky2_core::fri_proof::FriProof;
use qp_plonky2_core::merkle_tree::MerkleCap;
use serde::{Deserialize, Serialize};

use crate::field::extension::Extendable;
use crate::fri::structure::{FriOpeningBatch, FriOpenings};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::util::serialization::{Buffer, Read, Write};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct Proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A batch FRI argument for all openings.
    pub opening_proof: FriProof<F, C::Hasher, D>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct ProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: Proof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProofWithPublicInputs<F, C, D>
{
    pub fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        Buffer::decode_all(&bytes, |buffer| {
            buffer.read_proof_with_public_inputs(common_data)
        })
        .map_err(anyhow::Error::msg)
    }
}

// Re-export proof challenge types from core
pub use qp_plonky2_core::proof::ProofChallenges;

#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
/// The purported values of each polynomial at a single point.
pub struct OpeningSet<F: RichField + Extendable<D>, const D: usize> {
    pub constants: Vec<F::Extension>,
    pub plonk_sigmas: Vec<F::Extension>,
    pub wires: Vec<F::Extension>,
    pub plonk_zs: Vec<F::Extension>,
    pub plonk_zs_next: Vec<F::Extension>,
    pub partial_products: Vec<F::Extension>,
    pub quotient_polys: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> OpeningSet<F, D> {
    pub(crate) fn to_fri_openings(&self) -> FriOpenings<F, D> {
        let zeta_batch = FriOpeningBatch {
            values: [
                self.constants.as_slice(),
                self.plonk_sigmas.as_slice(),
                self.wires.as_slice(),
                self.plonk_zs.as_slice(),
                self.partial_products.as_slice(),
                self.quotient_polys.as_slice(),
            ]
            .concat(),
        };
        let zeta_next_batch = FriOpeningBatch {
            values: self.plonk_zs_next.clone(),
        };
        FriOpenings {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

// Tests removed - they require prover code (CircuitBuilder, PartialWitness) which is not
// available in the verifier-only crate. End-to-end verification tests should be in the
// main plonky2 crate or in integration tests that have access to both prover and verifier.
