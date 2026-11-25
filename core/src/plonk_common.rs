//! Utility methods and constants for Plonk shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::packed::PackedField;
use crate::field::types::Field;

/// Number of random values added to each polynomial for salting.
pub const SALT_SIZE: usize = 4;

/// Holds the Merkle tree index and blinding flag of a set of polynomials used in FRI.
#[derive(Debug, Copy, Clone)]
pub struct PlonkOracle {
    pub index: usize,
    pub blinding: bool,
}

impl PlonkOracle {
    pub const CONSTANTS_SIGMAS: PlonkOracle = PlonkOracle {
        index: 0,
        blinding: false,
    };
    pub const WIRES: PlonkOracle = PlonkOracle {
        index: 1,
        blinding: true,
    };
    pub const ZS_PARTIAL_PRODUCTS: PlonkOracle = PlonkOracle {
        index: 2,
        blinding: true,
    };
    pub const QUOTIENT: PlonkOracle = PlonkOracle {
        index: 3,
        blinding: true,
    };
}

/// Number of partial-product polynomials the permutation argument needs over `num_routed_wires`
/// wires multiplied `max_degree` at a time: one per chunk, the last chunk being `Z(gx)` itself.
pub const fn num_partial_products(num_routed_wires: usize, max_degree: usize) -> usize {
    num_routed_wires.div_ceil(max_degree) - 1
}

/// `LookupGate` holds one (input, output) pair per two routed wires.
pub const fn lookup_gate_slots(num_routed_wires: usize) -> usize {
    num_routed_wires / 2
}

/// One RE polynomial plus the partial Sum/LDC polynomials, each covering `lookup_degree` lookup
/// slots; none when the circuit has no lookup tables.
pub const fn num_lookup_polys(
    num_routed_wires: usize,
    lookup_degree: usize,
    num_luts: usize,
) -> usize {
    if num_luts == 0 {
        0
    } else {
        lookup_gate_slots(num_routed_wires).div_ceil(lookup_degree) + 1
    }
}

pub const fn salt_size(salted: bool) -> usize {
    if salted {
        SALT_SIZE
    } else {
        0
    }
}

/// Evaluate the polynomial which vanishes on any multiplicative subgroup of a given order `n`.
pub fn eval_zero_poly<F: Field>(n: usize, x: F) -> F {
    // Z(x) = x^n - 1
    x.exp_u64(n as u64) - F::ONE
}

/// Evaluate the Lagrange basis `L_0` with `L_0(1) = 1`, and `L_0(x) = 0` for other members of the
/// order `n` multiplicative subgroup.
pub fn eval_l_0<F: Field>(n: usize, x: F) -> F {
    if x.is_one() {
        // The code below would divide by zero, since we have (x - 1) in both the numerator and
        // denominator.
        return F::ONE;
    }

    // L_0(x) = (x^n - 1) / (n * (x - 1))
    //        = Z(x) / (n * (x - 1))
    eval_zero_poly(n, x) / (F::from_canonical_usize(n) * (x - F::ONE))
}

/// For each alpha in alphas, compute a reduction of the given terms using powers of alpha. T can
/// be any type convertible to a double-ended iterator.
pub fn reduce_with_powers_multi<
    'a,
    F: Field,
    I: DoubleEndedIterator<Item = &'a F>,
    T: IntoIterator<IntoIter = I>,
>(
    terms: T,
    alphas: &[F],
) -> Vec<F> {
    let mut cumul = vec![F::ZERO; alphas.len()];
    for &term in terms.into_iter().rev() {
        cumul
            .iter_mut()
            .zip(alphas)
            .for_each(|(c, &alpha)| *c = term.multiply_accumulate(*c, alpha));
    }
    cumul
}

pub fn reduce_with_powers<'a, P: PackedField, T: IntoIterator<Item = &'a P>>(
    terms: T,
    alpha: P::Scalar,
) -> P
where
    T::IntoIter: DoubleEndedIterator,
{
    let mut sum = P::ZEROS;
    for &term in terms.into_iter().rev() {
        sum = sum * alpha + term;
    }
    sum
}
