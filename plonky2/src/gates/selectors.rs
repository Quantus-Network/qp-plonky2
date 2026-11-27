//! Selector types and functions for gate constraints.
//!
//! Core types are re-exported from qp-plonky2-core.
//! Prover-specific functions are defined locally.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

// Re-export selector types from core
pub use qp_plonky2_core::selectors::{SelectorsInfo, UNUSED_SELECTOR};

use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialValues;
use crate::gates::gate::{GateInstance, GateRef};
use crate::hash::hash_types::RichField;

/// Returns the selector polynomials.
///
/// Selector polynomials are computed as follows:
/// Build a selector polynomial `S_i` for each gate group `G_i` of `selectors_info` (see
/// `SelectorsInfo::for_gate_degrees`), with
/// S_i\[j\] =
///     if j-th row gate=g_k in G_i
///         k
///     else
///         UNUSED_SELECTOR
pub(crate) fn selector_polynomials<F: RichField + Extendable<D>, const D: usize>(
    gates: &[GateRef<F, D>],
    instances: &[GateInstance<F, D>],
    selectors_info: &SelectorsInfo,
) -> Vec<PolynomialValues<F>> {
    let index = |id| gates.iter().position(|gate| gate.0.id() == id).unwrap();
    // Placeholder value to indicate that a gate doesn't use a selector polynomial.
    let unused = F::from_canonical_usize(UNUSED_SELECTOR);

    let mut polynomials =
        vec![PolynomialValues::zero(instances.len()); selectors_info.num_selectors()];
    for (row, instance) in instances.iter().enumerate() {
        let gate_index = index(instance.gate_ref.0.id());
        let gate_selector = selectors_info.selector_indices[gate_index];
        for (selector, polynomial) in polynomials.iter_mut().enumerate() {
            polynomial.values[row] = if selector == gate_selector {
                F::from_canonical_usize(gate_index)
            } else {
                unused
            };
        }
    }

    polynomials
}
