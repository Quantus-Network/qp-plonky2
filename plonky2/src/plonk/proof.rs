//! plonky2 proof definition.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::field::extension::Extendable;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::proof::{FriChallengesTarget, FriProof, FriProofTarget};
use crate::fri::structure::{
    FriOpeningBatch, FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget,
};
use crate::hash::hash_types::{MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofTarget<const D: usize> {
    pub wires_cap: MerkleCapTarget,
    pub plonk_zs_partial_products_cap: MerkleCapTarget,
    pub quotient_polys_cap: MerkleCapTarget,
    pub openings: OpeningSetTarget<D>,
    pub opening_proof: FriProofTarget<D>,
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
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

// Re-export proof challenge types from core
pub use qp_plonky2_core::proof::ProofChallenges;

pub(crate) struct ProofChallengesTarget<const D: usize> {
    pub plonk_betas: Vec<Target>,
    pub plonk_gammas: Vec<Target>,
    pub plonk_alphas: Vec<Target>,
    pub plonk_deltas: Vec<Target>,
    pub plonk_zeta: ExtensionTarget<D>,
    pub fri_challenges: FriChallengesTarget<D>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofWithPublicInputsTarget<const D: usize> {
    pub proof: ProofTarget<D>,
    pub public_inputs: Vec<Target>,
}

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
    pub lookup_zs: Vec<F::Extension>,
    pub lookup_zs_next: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> OpeningSet<F, D> {
    pub fn new<C: GenericConfig<D, F = F>>(
        zeta: F::Extension,
        g: F::Extension,
        constants_sigmas_commitment: &PolynomialBatch<F, C, D>,
        wires_commitment: &PolynomialBatch<F, C, D>,
        zs_partial_products_lookup_commitment: &PolynomialBatch<F, C, D>,
        quotient_polys_commitment: &PolynomialBatch<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        let eval_commitment = |z: F::Extension, c: &PolynomialBatch<F, C, D>| {
            c.polynomials
                .par_iter()
                .map(|p| p.to_extension().eval(z))
                .collect::<Vec<_>>()
        };
        let constants_sigmas_eval = eval_commitment(zeta, constants_sigmas_commitment);

        // `zs_partial_products_lookup_eval` contains the permutation argument polynomials as well as lookup polynomials.
        let zs_partial_products_lookup_eval =
            eval_commitment(zeta, zs_partial_products_lookup_commitment);
        let zs_partial_products_lookup_next_eval =
            eval_commitment(g * zeta, zs_partial_products_lookup_commitment);
        let quotient_polys = eval_commitment(zeta, quotient_polys_commitment);

        Self {
            constants: constants_sigmas_eval[common_data.constants_range()].to_vec(),
            plonk_sigmas: constants_sigmas_eval[common_data.sigmas_range()].to_vec(),
            wires: eval_commitment(zeta, wires_commitment),
            plonk_zs: zs_partial_products_lookup_eval[common_data.zs_range()].to_vec(),
            plonk_zs_next: zs_partial_products_lookup_next_eval[common_data.zs_range()].to_vec(),
            partial_products: zs_partial_products_lookup_eval[common_data.partial_products_range()]
                .to_vec(),
            quotient_polys,
            lookup_zs: zs_partial_products_lookup_eval[common_data.lookup_range()].to_vec(),
            lookup_zs_next: zs_partial_products_lookup_next_eval[common_data.lookup_range()]
                .to_vec(),
        }
    }
    pub(crate) fn to_fri_openings(&self) -> FriOpenings<F, D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                .concat(),
            }
        } else {
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatch {
                values: [self.plonk_zs_next.clone(), self.lookup_zs_next.clone()].concat(),
            }
        } else {
            FriOpeningBatch {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpenings {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

/// The purported values of each polynomial at a single point.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpeningSetTarget<const D: usize> {
    pub constants: Vec<ExtensionTarget<D>>,
    pub plonk_sigmas: Vec<ExtensionTarget<D>>,
    pub wires: Vec<ExtensionTarget<D>>,
    pub plonk_zs: Vec<ExtensionTarget<D>>,
    pub plonk_zs_next: Vec<ExtensionTarget<D>>,
    pub lookup_zs: Vec<ExtensionTarget<D>>,
    pub next_lookup_zs: Vec<ExtensionTarget<D>>,
    pub partial_products: Vec<ExtensionTarget<D>>,
    pub quotient_polys: Vec<ExtensionTarget<D>>,
}

impl<const D: usize> OpeningSetTarget<D> {
    pub(crate) fn to_fri_openings(&self) -> FriOpeningsTarget<D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                .concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [self.plonk_zs_next.clone(), self.next_lookup_zs.clone()].concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpeningsTarget {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}
