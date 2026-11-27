#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::iter;

use itertools::Itertools;

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::plonk_common;
use crate::plonk::vars::EvaluationVars;

/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
) -> Vec<F> {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    numerators
        .chunks(chunk_size)
        .zip_eq(denominators.chunks(chunk_size))
        .zip_eq(product_accs.tuple_windows())
        .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
            let num_chunk_product = nume_chunk.iter().copied().product();
            let den_chunk_product = deno_chunk.iter().copied().product();
            // Assert that next_acc * deno_product = prev_acc * nume_product.
            prev_acc * num_chunk_product - next_acc * den_chunk_product
        })
        .collect()
}

/// Evaluate the vanishing polynomial at `x`. In this context, the vanishing polynomial is a random
/// linear combination of gate constraints, plus some other terms relating to the permutation
/// argument. All such terms should vanish on `H`.
pub(crate) fn eval_vanishing_poly<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    x: F::Extension,
    vars: EvaluationVars<F, D>,
    local_zs: &[F::Extension],
    next_zs: &[F::Extension],
    partial_products: &[F::Extension],
    s_sigmas: &[F::Extension],
    betas: &[F],
    gammas: &[F],
    alphas: &[F],
) -> Vec<F::Extension> {
    let max_degree = common_data.permutation_partial_product_degree();
    let num_prods = common_data.num_partial_products;

    let constraint_terms = evaluate_gate_constraints::<F, D>(common_data, vars);

    // The L_0(x) (Z(x) - 1) vanishing terms.
    let mut vanishing_z_1_terms = Vec::new();

    // The terms checking the partial products.
    let mut vanishing_partial_products_terms = Vec::new();

    let l_0_x = plonk_common::eval_l_0(common_data.degree(), x);

    for i in 0..common_data.config.num_challenges {
        let z_x = local_zs[i];
        let z_gx = next_zs[i];
        vanishing_z_1_terms.push(l_0_x * (z_x - F::Extension::ONE));

        let numerator_values = (0..common_data.config.num_routed_wires)
            .map(|j| {
                let wire_value = vars.local_wires[j];
                let k_i = common_data.k_is[j];
                let s_id = x.scalar_mul(k_i);
                wire_value + s_id.scalar_mul(betas[i]) + gammas[i].into()
            })
            .collect::<Vec<_>>();
        let denominator_values = (0..common_data.config.num_routed_wires)
            .map(|j| {
                let wire_value = vars.local_wires[j];
                let s_sigma = s_sigmas[j];
                wire_value + s_sigma.scalar_mul(betas[i]) + gammas[i].into()
            })
            .collect::<Vec<_>>();

        // The partial products considered for this iteration of `i`.
        let current_partial_products = &partial_products[i * num_prods..(i + 1) * num_prods];
        // Check the quotient partial products.
        let partial_product_checks = check_partial_products(
            &numerator_values,
            &denominator_values,
            current_partial_products,
            z_x,
            z_gx,
            max_degree,
        );
        vanishing_partial_products_terms.extend(partial_product_checks);
    }

    let vanishing_terms = [
        vanishing_z_1_terms,
        vanishing_partial_products_terms,
        constraint_terms,
    ]
    .concat();

    let alphas = &alphas.iter().map(|&a| a.into()).collect::<Vec<_>>();
    plonk_common::reduce_with_powers_multi(&vanishing_terms, alphas)
}

/// Evaluates all gate constraints.
///
/// `num_gate_constraints` is the largest number of constraints imposed by any gate. It is not
/// strictly necessary, but it helps performance by ensuring that we allocate a vector with exactly
/// the capacity that we need.
pub fn evaluate_gate_constraints<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    vars: EvaluationVars<F, D>,
) -> Vec<F::Extension> {
    let mut constraints = vec![F::Extension::ZERO; common_data.num_gate_constraints];
    for (i, gate) in common_data.gates.iter().enumerate() {
        let selector_index = common_data.selectors_info.selector_indices[i];
        let gate_constraints = gate.0.eval_filtered(
            vars,
            i,
            selector_index,
            common_data.selectors_info.groups[selector_index].clone(),
            common_data.selectors_info.num_selectors(),
        );
        for (i, c) in gate_constraints.into_iter().enumerate() {
            debug_assert!(
                i < common_data.num_gate_constraints,
                "num_constraints() gave too low of a number"
            );
            constraints[i] += c;
        }
    }
    constraints
}
