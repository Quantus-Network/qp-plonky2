use alloc::vec::Vec;

use anyhow::{ensure, Result};
use plonky2_util::log2_ceil;

use crate::fft::ifft;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

/// Computes the unique degree < n interpolant of an arbitrary list of n (point, value) pairs.
///
/// Note that the implementation assumes that `F` is two-adic, in particular that
/// `2^{F::TWO_ADICITY} >= points.len()`. This leads to a simple FFT-based implementation.
pub fn interpolant<F: Field>(points: &[(F, F)]) -> PolynomialCoeffs<F> {
    try_interpolant(points).expect("invalid interpolation points")
}

pub fn try_interpolant<F: Field>(points: &[(F, F)]) -> Result<PolynomialCoeffs<F>> {
    validate_interpolation_points(points)?;
    let n = points.len();
    let n_log = log2_ceil(n);

    let subgroup = F::two_adic_subgroup(n_log);
    let barycentric_weights = try_barycentric_weights(points)?;
    let subgroup_evals = subgroup
        .into_iter()
        .map(|x| try_interpolate(points, x, &barycentric_weights))
        .collect::<Result<Vec<_>>>()?;

    let mut coeffs = ifft(PolynomialValues::new(subgroup_evals));
    coeffs.trim();
    Ok(coeffs)
}

/// Interpolate the polynomial defined by an arbitrary set of (point, value) pairs at the given
/// point `x`.
pub fn interpolate<F: Field>(points: &[(F, F)], x: F, barycentric_weights: &[F]) -> F {
    try_interpolate(points, x, barycentric_weights).expect("invalid interpolation inputs")
}

pub fn try_interpolate<F: Field>(points: &[(F, F)], x: F, barycentric_weights: &[F]) -> Result<F> {
    validate_interpolation_points(points)?;
    ensure!(
        barycentric_weights.len() == points.len(),
        "barycentric weight count must match point count"
    );

    // If x is in the list of points, the Lagrange formula would divide by zero.
    for &(x_i, y_i) in points {
        if x_i == x {
            return Ok(y_i);
        }
    }

    let l_x: F = points.iter().map(|&(x_i, _y_i)| x - x_i).product();

    let mut sum = F::ZERO;
    for i in 0..points.len() {
        let x_i = points[i].0;
        let y_i = points[i].1;
        let w_i = barycentric_weights[i];
        let denominator = x - x_i;
        let denominator_inv = denominator
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("interpolation denominator is zero"))?;
        sum += w_i * denominator_inv * y_i;
    }

    Ok(l_x * sum)
}

pub fn barycentric_weights<F: Field>(points: &[(F, F)]) -> Vec<F> {
    try_barycentric_weights(points).expect("invalid interpolation points")
}

pub fn try_barycentric_weights<F: Field>(points: &[(F, F)]) -> Result<Vec<F>> {
    validate_interpolation_points(points)?;
    let n = points.len();
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let denominator = (0..n)
            .filter(|&j| j != i)
            .map(|j| points[i].0 - points[j].0)
            .product::<F>();
        let inverse = denominator
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("barycentric denominator is zero"))?;
        weights.push(inverse);
    }
    Ok(weights)
}

/// Interpolate the linear polynomial passing through `points` on `x`.
pub fn interpolate2<F: Field>(points: [(F, F); 2], x: F) -> F {
    try_interpolate2(points, x).expect("interpolate2 requires distinct x-coordinates")
}

pub fn try_interpolate2<F: Field>(points: [(F, F); 2], x: F) -> Result<F> {
    // a0 -> a1
    // b0 -> b1
    // x  -> a1 + (x-a0)*(b1-a1)/(b0-a0)
    let (a0, a1) = points[0];
    let (b0, b1) = points[1];
    let denominator_inv = (b0 - a0)
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("interpolate2 requires distinct x-coordinates"))?;
    Ok(a1 + (x - a0) * (b1 - a1) * denominator_inv)
}

fn validate_interpolation_points<F: Field>(points: &[(F, F)]) -> Result<()> {
    ensure!(!points.is_empty(), "interpolation point set is empty");
    let n_log = log2_ceil(points.len());
    ensure!(
        n_log <= F::TWO_ADICITY,
        "interpolation point set exceeds field two-adicity"
    );
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            ensure!(
                points[i].0 != points[j].0,
                "interpolation x-coordinates must be distinct"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "rand")]
mod tests {
    use super::*;
    use crate::extension::quartic::QuarticExtension;
    use crate::goldilocks_field::GoldilocksField;
    use crate::polynomial::PolynomialCoeffs;
    use crate::types::{Field, Sample};

    #[test]
    fn interpolant_random() {
        type F = GoldilocksField;

        for deg in 1..10 {
            let domain = F::rand_vec(deg);
            let coeffs = F::rand_vec(deg);
            let coeffs = PolynomialCoeffs { coeffs };

            let points = eval_naive(&coeffs, &domain);
            assert_eq!(interpolant(&points), coeffs);
        }
    }

    #[test]
    fn interpolant_random_roots_of_unity() {
        type F = GoldilocksField;

        for deg_log in 0..4 {
            let deg = 1 << deg_log;
            let domain = F::two_adic_subgroup(deg_log);
            let coeffs = F::rand_vec(deg);
            let coeffs = PolynomialCoeffs { coeffs };

            let points = eval_naive(&coeffs, &domain);
            assert_eq!(interpolant(&points), coeffs);
        }
    }

    #[test]
    fn interpolant_random_overspecified() {
        type F = GoldilocksField;

        for deg in 0..10 {
            let points = deg + 5;
            let domain = F::rand_vec(points);
            let coeffs = F::rand_vec(deg);
            let coeffs = PolynomialCoeffs { coeffs };

            let points = eval_naive(&coeffs, &domain);
            assert_eq!(interpolant(&points), coeffs);
        }
    }

    fn eval_naive<F: Field>(coeffs: &PolynomialCoeffs<F>, domain: &[F]) -> Vec<(F, F)> {
        domain.iter().map(|&x| (x, coeffs.eval(x))).collect()
    }

    #[test]
    fn test_interpolate2() {
        type F = QuarticExtension<GoldilocksField>;
        let points = [(F::rand(), F::rand()), (F::rand(), F::rand())];
        let x = F::rand();

        let ev0 = interpolant(&points).eval(x);
        let ev1 = interpolate(&points, x, &barycentric_weights(&points));
        let ev2 = interpolate2(points, x);

        assert_eq!(ev0, ev1);
        assert_eq!(ev0, ev2);
    }

    #[test]
    fn malformed_interpolation_inputs_return_err() {
        type F = GoldilocksField;
        let duplicate = vec![(F::ONE, F::TWO), (F::ONE, F::ZERO)];
        assert!(try_barycentric_weights(&duplicate).is_err());
        assert!(try_interpolant(&duplicate).is_err());
        assert!(try_interpolate2([(F::ONE, F::ZERO), (F::ONE, F::TWO)], F::ZERO).is_err());

        let empty: Vec<(F, F)> = Vec::new();
        assert!(try_barycentric_weights(&empty).is_err());
        assert!(try_interpolant(&empty).is_err());

        let valid = vec![
            (F::ZERO, F::from_canonical_u64(3)),
            (F::ONE, F::from_canonical_u64(5)),
        ];
        let weights = try_barycentric_weights(&valid).unwrap();
        assert_eq!(
            try_interpolate(&valid, F::TWO, &weights).unwrap(),
            interpolate(&valid, F::TWO, &weights)
        );
        assert_eq!(try_interpolant(&valid).unwrap(), interpolant(&valid));
    }
}
