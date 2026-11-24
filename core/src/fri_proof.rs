//! FRI proof types shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::config::Hasher;
use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialCoeffs;
use crate::hash_types::RichField;
use crate::merkle_proofs::MerkleProof;
use crate::merkle_tree::MerkleCap;
use crate::plonk_common::salt_size;

/// Evaluations and Merkle proof produced by the prover in a FRI query step.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriQueryStep<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub evals: Vec<F::Extension>,
    pub merkle_proof: MerkleProof<F, H>,
}

/// Evaluations and Merkle proofs of the original set of polynomials,
/// before they are combined into a composition polynomial.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriInitialTreeProof<F: RichField, H: Hasher<F>> {
    pub evals_proofs: Vec<(Vec<F>, MerkleProof<F, H>)>,
}

impl<F: RichField, H: Hasher<F>> FriInitialTreeProof<F, H> {
    pub fn unsalted_eval(&self, oracle_index: usize, poly_index: usize, salted: bool) -> F {
        self.unsalted_evals(oracle_index, salted)[poly_index]
    }

    pub fn unsalted_evals(&self, oracle_index: usize, salted: bool) -> &[F] {
        let evals = &self.evals_proofs[oracle_index].0;
        &evals[..evals.len() - salt_size(salted)]
    }
}

/// Proof for a FRI query round.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriQueryRound<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub initial_trees_proof: FriInitialTreeProof<F, H>,
    pub steps: Vec<FriQueryStep<F, H, D>>,
}

/// The full FRI proof.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriProof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// A Merkle cap for each reduced polynomial in the commit phase.
    pub commit_phase_merkle_caps: Vec<MerkleCap<F, H>>,
    /// Query rounds proofs
    pub query_round_proofs: Vec<FriQueryRound<F, H, D>>,
    /// The final reduced polynomial.
    pub final_poly: PolynomialCoeffs<F::Extension>,
    /// Witness showing that the prover did PoW.
    pub pow_witness: F,
}
