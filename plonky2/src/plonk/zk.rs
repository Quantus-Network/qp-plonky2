//! Masked polynomial helpers for PLONK commitments.
//!
//! PolyFri phase 1 commits the logical masked polynomial directly,
//! `M(X) = f(X) + (X^n - 1) * r(X)`, so public proof objects never expose the raw split pieces.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::hash::hash_types::RichField;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::cached_point_power;
use crate::util::timing::TimingTree;

/// How a logical polynomial is represented inside the raw committed batch.
///
/// `SplitMask` remains only for prover-private helper paths and tests. Public PolyFri
/// commitments now use `Raw` for the masked logical polynomial itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogicalPolynomialLayout {
    Raw {
        raw_index: usize,
    },
    SplitMask {
        low_index: usize,
        high_index: usize,
        /// Trace-domain size `n` used in `low(X) + X^n * high(X)`.
        split_power: usize,
    },
}

/// A polynomial commitment together with the mapping from raw committed pieces to logical
/// polynomials used by the PLONK protocol.
#[derive(Debug)]
pub struct LogicalPolynomialBatch<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub raw: PolynomialBatch<F, C, D>,
    pub logical_layouts: Vec<LogicalPolynomialLayout>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for LogicalPolynomialBatch<F, C, D>
{
    fn default() -> Self {
        Self {
            raw: PolynomialBatch::default(),
            logical_layouts: Vec::new(),
        }
    }
}

/// Prover-side plan for logical masking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitMaskPlan {
    /// Trace-domain size `n` used in the vanishing-term mask `f(X) + (X^n - 1) * r(X)`.
    pub split_power: usize,
    /// Degree bound for the sampled mask polynomial `r(X)`.
    pub mask_degree: usize,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    LogicalPolynomialBatch<F, C, D>
{
    fn logical_eval_with_point_powers(
        &self,
        logical_index: usize,
        point: F::Extension,
        point_power_cache: &mut Vec<(usize, F::Extension)>,
    ) -> F::Extension {
        match &self.logical_layouts[logical_index] {
            LogicalPolynomialLayout::Raw { raw_index } => self.raw.polynomials[*raw_index]
                .to_extension::<D>()
                .eval(point),
            LogicalPolynomialLayout::SplitMask {
                low_index,
                high_index,
                split_power,
            } => {
                let low = self.raw.polynomials[*low_index]
                    .to_extension::<D>()
                    .eval(point);
                let high = self.raw.polynomials[*high_index]
                    .to_extension::<D>()
                    .eval(point);
                low + cached_point_power(point, *split_power, point_power_cache) * high
            }
        }
    }

    pub fn logical_eval(&self, logical_index: usize, point: F::Extension) -> F::Extension {
        let mut point_power_cache = Vec::new();
        self.logical_eval_with_point_powers(logical_index, point, &mut point_power_cache)
    }

    pub fn logical_evals(&self, point: F::Extension) -> Vec<F::Extension> {
        let mut point_power_cache = Vec::new();
        (0..self.logical_layouts.len())
            .map(|i| self.logical_eval_with_point_powers(i, point, &mut point_power_cache))
            .collect()
    }

    pub fn raw_polys_len(&self) -> usize {
        self.raw.polynomials.len()
    }

    /// Reconstruct the logical LDE values at a queried coset point.
    ///
    /// PolyFri now commits masked logical polynomials directly, but the helper still accepts the
    /// legacy split layout for prover-private code paths and tests.
    pub fn get_lde_values(&self, index: usize, step: usize, point: F) -> Vec<F> {
        let raw_values = self.raw.get_lde_values(index, step);
        let mut point_power_cache = Vec::new();
        self.logical_layouts
            .iter()
            .map(|layout| match layout {
                LogicalPolynomialLayout::Raw { raw_index } => raw_values[*raw_index],
                LogicalPolynomialLayout::SplitMask {
                    low_index,
                    high_index,
                    split_power,
                } => {
                    raw_values[*low_index]
                        + cached_point_power(point, *split_power, &mut point_power_cache)
                            * raw_values[*high_index]
                }
            })
            .collect()
    }
}

pub fn commit_values_with_split_mask<F, C, const D: usize>(
    values: Vec<PolynomialValues<F>>,
    mask_plan: Option<&SplitMaskPlan>,
    target_len: Option<usize>,
    rate_bits: usize,
    leaf_hiding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    fft_root_table: Option<&FftRootTable<F>>,
) -> LogicalPolynomialBatch<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let coeffs = timed!(
        timing,
        "IFFT for masked commit",
        values
            .into_iter()
            .map(PolynomialValues::ifft)
            .collect::<Vec<_>>()
    );

    commit_coeffs_with_split_mask(
        coeffs,
        mask_plan,
        target_len,
        rate_bits,
        leaf_hiding,
        cap_height,
        timing,
        fft_root_table,
    )
}

pub fn commit_coeffs_with_split_mask<F, C, const D: usize>(
    coeffs: Vec<PolynomialCoeffs<F>>,
    mask_plan: Option<&SplitMaskPlan>,
    target_len: Option<usize>,
    rate_bits: usize,
    leaf_hiding: bool,
    cap_height: usize,
    timing: &mut TimingTree,
    fft_root_table: Option<&FftRootTable<F>>,
) -> LogicalPolynomialBatch<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    match mask_plan {
        None => {
            let logical_layouts = (0..coeffs.len())
                .map(|raw_index| LogicalPolynomialLayout::Raw { raw_index })
                .collect();
            let target_len = target_len.unwrap_or_else(|| {
                coeffs
                    .iter()
                    .map(|poly| poly.len())
                    .max()
                    .expect("masked commitment requires at least one polynomial")
                    .next_power_of_two()
            });
            let mut coeffs = coeffs;
            for poly in &mut coeffs {
                poly.pad(target_len).unwrap();
            }
            let raw = PolynomialBatch::from_coeffs(
                coeffs,
                rate_bits,
                leaf_hiding,
                cap_height,
                timing,
                fft_root_table,
            );
            LogicalPolynomialBatch {
                raw,
                logical_layouts,
            }
        }
        Some(mask_plan) => {
            let masks = timed!(
                timing,
                "sample phase-1 masks",
                sample_mask_polys(coeffs.len(), mask_plan.mask_degree)
            );

            let mut masked_coeffs = Vec::with_capacity(coeffs.len());
            let mut logical_layouts = Vec::with_capacity(coeffs.len());
            for (raw_index, (f, r)) in coeffs.into_iter().zip(masks).enumerate() {
                masked_coeffs.push(mask_with_vanishing(f, r, mask_plan.split_power));
                logical_layouts.push(LogicalPolynomialLayout::Raw { raw_index });
            }

            let target_len = target_len.unwrap_or_else(|| {
                masked_coeffs
                    .iter()
                    .map(|poly| poly.len())
                    .max()
                    .expect("masked commitment requires at least one polynomial")
                    .next_power_of_two()
            });
            for poly in &mut masked_coeffs {
                poly.pad(target_len).unwrap();
            }

            let raw = PolynomialBatch::from_coeffs(
                masked_coeffs,
                rate_bits,
                leaf_hiding,
                cap_height,
                timing,
                fft_root_table,
            );

            LogicalPolynomialBatch {
                raw,
                logical_layouts,
            }
        }
    }
}

pub fn sample_mask_polys<F: RichField>(
    count: usize,
    mask_degree: usize,
) -> Vec<PolynomialCoeffs<F>> {
    #[cfg(feature = "rand")]
    {
        (0..count)
            .map(|_| PolynomialCoeffs::new(F::rand_vec(mask_degree + 1)))
            .collect()
    }

    #[cfg(not(feature = "rand"))]
    {
        let _ = (count, mask_degree);
        panic!("Cannot enable PolyFri split masking without rand feature");
    }
}

/// Return the masked logical polynomial `f(X) + (X^n - 1) * r(X)`.
pub fn mask_with_vanishing<F: Field>(
    f: PolynomialCoeffs<F>,
    r: PolynomialCoeffs<F>,
    split_power: usize,
) -> PolynomialCoeffs<F> {
    assert!(
        f.len() <= split_power,
        "Masked logical polynomial exceeds the trace-degree bound"
    );
    assert!(
        r.len() <= split_power,
        "Mask polynomial exceeds the trace-degree bound"
    );

    let mut coeffs = vec![F::ZERO; split_power + r.len()];
    for (i, coeff) in f.coeffs.into_iter().enumerate() {
        coeffs[i] += coeff;
    }
    for (i, coeff) in r.coeffs.into_iter().enumerate() {
        coeffs[i] -= coeff;
        coeffs[i + split_power] += coeff;
    }

    PolynomialCoeffs::new(coeffs)
}

pub fn split_mask_coeffs<F: Field>(
    mut f: PolynomialCoeffs<F>,
    mut r: PolynomialCoeffs<F>,
    split_power: usize,
) -> (PolynomialCoeffs<F>, PolynomialCoeffs<F>) {
    assert!(
        f.len() <= split_power,
        "Split-mask low polynomial exceeds the trace-degree bound"
    );
    assert!(
        r.len() <= split_power,
        "Split-mask mask polynomial exceeds the trace-degree bound"
    );

    let target_len = f.len().max(r.len());
    f.pad(target_len).unwrap();
    r.pad(target_len).unwrap();

    let low = &f - &r;
    (low, r)
}

#[cfg(test)]
mod tests {
    use super::{
        mask_with_vanishing, split_mask_coeffs, LogicalPolynomialBatch, LogicalPolynomialLayout,
        SplitMaskPlan,
    };
    use crate::field::extension::Extendable;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::polynomial::PolynomialCoeffs;
    use crate::field::types::Field;
    use crate::fri::oracle::PolynomialBatch;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::util::timing::TimingTree;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;

    #[test]
    fn split_mask_coeffs_reconstructs_masked_polynomial() {
        let n = 8;
        let f = PolynomialCoeffs::new(vec![F::ONE, F::TWO, F::from_canonical_u64(3)]);
        let r = PolynomialCoeffs::new(vec![F::from_canonical_u64(5), F::from_canonical_u64(7)]);

        let (low, high) = split_mask_coeffs(f.clone(), r.clone(), n);

        for i in 0..f.coeffs.len().max(r.coeffs.len()) {
            let original = f.coeffs.get(i).copied().unwrap_or(F::ZERO);
            let mask = r.coeffs.get(i).copied().unwrap_or(F::ZERO);
            let low_coeff = low.coeffs.get(i).copied().unwrap_or(F::ZERO);
            let high_coeff = high.coeffs.get(i).copied().unwrap_or(F::ZERO);
            assert_eq!(low_coeff + high_coeff, original);
            assert_eq!(high_coeff, mask);
        }
    }

    #[test]
    fn mask_with_vanishing_matches_expected_identity() {
        let n = 8;
        let f = PolynomialCoeffs::new(vec![F::from_canonical_u64(4), F::ONE, F::TWO]);
        let r = PolynomialCoeffs::new(vec![F::from_canonical_u64(3), F::from_canonical_u64(5)]);

        let masked = mask_with_vanishing(f.clone(), r.clone(), n);
        let off_subgroup = F::coset_shift().into();
        let expected = f.to_extension::<D>().eval(off_subgroup)
            + (off_subgroup.exp_u64(n as u64) - <F as Extendable<D>>::Extension::ONE)
                * r.to_extension::<D>().eval(off_subgroup);

        assert_eq!(masked.to_extension::<D>().eval(off_subgroup), expected);
    }

    #[test]
    fn logical_eval_reconstructs_split_mask_on_and_off_subgroup() {
        let n = 8;
        let f = PolynomialCoeffs::new(vec![F::from_canonical_u64(4), F::ONE, F::TWO]);
        let r = PolynomialCoeffs::new(vec![F::from_canonical_u64(3), F::from_canonical_u64(5)]);
        let (low, high) = split_mask_coeffs(f.clone(), r.clone(), n);

        let logical_batch = LogicalPolynomialBatch::<F, C, D> {
            raw: PolynomialBatch {
                polynomials: vec![low, high.clone()],
                merkle_tree: Default::default(),
                degree_log: 0,
                rate_bits: 0,
                blinding: false,
            },
            logical_layouts: vec![LogicalPolynomialLayout::SplitMask {
                low_index: 0,
                high_index: 1,
                split_power: n,
            }],
        };

        let subgroup_gen = F::primitive_root_of_unity(3);
        for i in 0..n {
            let point = subgroup_gen.exp_u64(i as u64).into();
            assert_eq!(
                logical_batch.logical_eval(0, point),
                f.to_extension::<D>().eval(point)
            );
        }

        let off_subgroup = F::coset_shift().into();
        let expected = f.to_extension::<D>().eval(off_subgroup)
            + (off_subgroup.exp_u64(n as u64) - <F as Extendable<D>>::Extension::ONE)
                * high.to_extension::<D>().eval(off_subgroup);
        assert_eq!(logical_batch.logical_eval(0, off_subgroup), expected);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn commit_coeffs_split_mask_exposes_masked_logical_polynomials() {
        let n = 8;
        let coeffs = vec![
            PolynomialCoeffs::new(vec![F::ONE, F::TWO]),
            PolynomialCoeffs::new(vec![F::from_canonical_u64(9), F::from_canonical_u64(11)]),
        ];
        let originals = coeffs.clone();
        let plan = SplitMaskPlan {
            split_power: n,
            mask_degree: 1,
        };

        let batch = super::commit_coeffs_with_split_mask::<F, C, D>(
            coeffs,
            Some(&plan),
            None,
            1,
            false,
            0,
            &mut TimingTree::default(),
            None,
        );

        assert_eq!(batch.logical_layouts.len(), originals.len());
        assert_eq!(batch.raw_polys_len(), originals.len());
        assert!(batch
            .logical_layouts
            .iter()
            .all(|layout| matches!(layout, LogicalPolynomialLayout::Raw { .. })));

        let subgroup_gen = F::primitive_root_of_unity(3);
        for i in 0..n {
            let point = subgroup_gen.exp_u64(i as u64).into();
            let evals = batch.logical_evals(point);
            let expected = originals
                .iter()
                .map(|poly| poly.to_extension::<D>().eval(point))
                .collect::<Vec<_>>();
            assert_eq!(evals, expected);
        }
    }
}
