//! Selector types and functions for gate constraints.
//!
//! Core types are re-exported from qp-plonky2-core.
//! Prover-specific functions are defined locally.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

// Re-export selector types from core
pub use qp_plonky2_core::selectors::{LookupSelectors, SelectorsInfo, UNUSED_SELECTOR};

use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialValues;
use crate::gates::gate::{GateInstance, GateRef};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_builder::LookupWire;

/// Returns selector polynomials for each LUT. We have two constraint domains (remember that gates are stored upside down):
/// - [last_lut_row, first_lut_row] (Sum and RE transition constraints),
/// - [last_lu_row, last_lut_row - 1] (LDC column transition constraints).
///
/// We also add two more:
/// - {first_lut_row + 1} where we check the initial values of sum and RE (which are 0),
/// - {last_lu_row} where we check that the last value of LDC is 0.
///
/// Conceptually they're part of the selector ends lookups, but since we can have one polynomial for *all* LUTs it's here.
pub(crate) fn selectors_lookup<F: RichField + Extendable<D>, const D: usize>(
    _gates: &[GateRef<F, D>],
    instances: &[GateInstance<F, D>],
    lookup_rows: &[LookupWire],
) -> Vec<PolynomialValues<F>> {
    let n = instances.len();
    let mut lookup_selectors = Vec::with_capacity(LookupSelectors::StartEnd as usize);
    for _ in 0..LookupSelectors::StartEnd as usize {
        lookup_selectors.push(PolynomialValues::<F>::new(vec![F::ZERO; n]));
    }

    for &LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in lookup_rows
    {
        for row in last_lut_row..first_lut_row + 1 {
            lookup_selectors[LookupSelectors::TransSre as usize].values[row] = F::ONE;
        }
        for row in last_lu_row..last_lut_row {
            lookup_selectors[LookupSelectors::TransLdc as usize].values[row] = F::ONE;
        }
        lookup_selectors[LookupSelectors::InitSre as usize].values[first_lut_row + 1] = F::ONE;
        lookup_selectors[LookupSelectors::LastLdc as usize].values[last_lu_row] = F::ONE;
    }
    lookup_selectors
}

/// Returns selectors for checking the validity of the LUTs.
/// Each selector equals one on its respective LUT's `last_lut_row`, and 0 elsewhere.
pub(crate) fn selector_ends_lookups<F: RichField + Extendable<D>, const D: usize>(
    lookup_rows: &[LookupWire],
    instances: &[GateInstance<F, D>],
) -> Vec<PolynomialValues<F>> {
    let n = instances.len();
    let mut lookups_ends = Vec::with_capacity(lookup_rows.len());
    for &LookupWire {
        last_lu_gate: _,
        last_lut_gate: last_lut_row,
        first_lut_gate: _,
    } in lookup_rows
    {
        let mut lookup_ends = PolynomialValues::<F>::new(vec![F::ZERO; n]);
        lookup_ends.values[last_lut_row] = F::ONE;
        lookups_ends.push(lookup_ends);
    }
    lookups_ends
}

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
