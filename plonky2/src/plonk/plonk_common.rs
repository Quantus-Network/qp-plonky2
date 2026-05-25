//! Utility methods and constants for Plonk.
//!
//! Core types and functions are re-exported from qp-plonky2-core.
//! Circuit-specific functions are defined locally.

// Re-export common types and functions from core
pub use qp_plonky2_core::plonk_common::{
    eval_l_0, eval_zero_poly, reduce_with_powers, reduce_with_powers_multi, salt_size, PlonkOracle,
    SALT_SIZE,
};

use crate::field::extension::Extendable;
use crate::gates::arithmetic_base::ArithmeticGate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::util::reducing::ReducingFactorTarget;

/// Evaluates the Lagrange basis L_0(x), which has L_0(1) = 1 and vanishes at all other points in
/// the order-`n` subgroup.
pub(crate) fn eval_l_0_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    n: usize,
    x: ExtensionTarget<D>,
    _x_pow_n: ExtensionTarget<D>,
) -> ExtensionTarget<D> {
    debug_assert!(n.is_power_of_two());

    // For n = 2^k, L_0(x) = n^{-1} * product_j (1 + x^{2^j}).
    // This is equivalent to (x^n - 1) / (n * (x - 1)) away from x = 1,
    // and remains well-defined at every subgroup point.
    let one = builder.one_extension();
    let mut x_power = x;
    let mut product = one;
    for _ in 0..n.trailing_zeros() {
        let factor = builder.add_extension(one, x_power);
        product = builder.mul_extension(product, factor);
        x_power = builder.square_extension(x_power);
    }

    let inv_n = F::from_canonical_usize(n).inverse();
    builder.mul_const_extension(inv_n, product)
}

pub fn reduce_with_powers_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    terms: &[Target],
    alpha: Target,
) -> Target {
    if terms.len() <= ArithmeticGate::new_from_config(&builder.config).num_ops + 1 {
        terms
            .iter()
            .rev()
            .fold(builder.zero(), |acc, &t| builder.mul_add(alpha, acc, t))
    } else {
        let alpha = builder.convert_to_ext(alpha);
        let mut alpha = ReducingFactorTarget::new(alpha);
        alpha.reduce_base(terms, builder).0[0]
    }
}

pub fn reduce_with_powers_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    terms: &[ExtensionTarget<D>],
    alpha: Target,
) -> ExtensionTarget<D> {
    let alpha = builder.convert_to_ext(alpha);
    let mut alpha = ReducingFactorTarget::new(alpha);
    alpha.reduce(terms, builder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::types::Field;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type FE = <C as GenericConfig<D>>::FE;

    #[test]
    fn eval_l_0_circuit_handles_subgroup_points() -> anyhow::Result<()> {
        let n = 8usize;
        let mut points = FE::two_adic_subgroup(n.trailing_zeros() as usize);
        points.extend([FE::from_canonical_u64(7), FE::from_canonical_u64(123)]);

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        for x in points {
            let x_target = builder.constant_extension(x);
            let x_pow_n = builder.constant_extension(x.exp_u64(n as u64));
            let actual = eval_l_0_circuit(&mut builder, n, x_target, x_pow_n);
            let expected = builder.constant_extension(eval_l_0(n, x));
            builder.connect_extension(actual, expected);
        }

        let data = builder.build::<C>();
        let proof = data.prove(PartialWitness::new())?;
        data.verify(proof)
    }
}
