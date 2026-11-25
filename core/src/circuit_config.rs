//! Circuit configuration types shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use serde::{Deserialize, Serialize};

use crate::field::cosets::get_unique_coset_shifts;
use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::fri::{FriConfig, FriParams, FriReductionStrategy};
use crate::plonk_common::{
    num_lookup_polys, num_lookup_selectors, num_partial_products, LookupTable,
};
use crate::selectors::SelectorsInfo;

/// Configuration to be used when building a circuit. This defines the shape of the circuit
/// as well as its targeted security level and sub-protocol (e.g. FRI) parameters.
///
/// It supports a [`Default`] implementation tailored for recursion with Poseidon hash (of width 12)
/// as internal hash function and FRI rate of 1/8.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CircuitConfig {
    /// The number of wires available at each row. This corresponds to the "width" of the circuit,
    /// and consists in the sum of routed wires and advice wires.
    pub num_wires: usize,
    /// The number of routed wires, i.e. wires that will be involved in Plonk's permutation argument.
    /// This allows copy constraints, i.e. enforcing that two distant values in a circuit are equal.
    /// Non-routed wires are called advice wires.
    pub num_routed_wires: usize,
    /// The number of constants that can be used per gate.
    pub num_constants: usize,
    /// Whether to use a dedicated gate for base field arithmetic, rather than using a single gate
    /// for both base field and extension field arithmetic.
    pub use_base_arithmetic_gate: bool,
    pub security_bits: usize,
    /// The number of challenge points to generate, for IOPs that have soundness errors of (roughly)
    /// `degree / |F|`.
    pub num_challenges: usize,
    /// A boolean to activate the zero-knowledge property. When this is set to `false`, proofs *may*
    /// leak additional information.
    pub zero_knowledge: bool,
    /// A cap on the quotient polynomial's degree factor. The actual degree factor is derived
    /// systematically, but will never exceed this value.
    pub max_quotient_degree_factor: usize,
    pub fri_config: FriConfig,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::standard_recursion_config()
    }
}

impl CircuitConfig {
    /// The widest row `check_valid` accepts, for `num_wires` and `num_constants`.
    pub const MAX_WIDTH: usize = 1 << 16;

    pub const fn num_advice_wires(&self) -> usize {
        self.num_wires - self.num_routed_wires
    }

    /// A typical recursion config, without zero-knowledge, targeting ~100 bit security.
    pub const fn standard_recursion_config() -> Self {
        Self {
            num_wires: 143,
            num_routed_wires: 80,
            num_constants: 2,
            use_base_arithmetic_gate: true,
            security_bits: 100,
            num_challenges: 2,
            zero_knowledge: false,
            max_quotient_degree_factor: 8,
            fri_config: FriConfig {
                rate_bits: 3,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 28,
            },
        }
    }

    pub fn standard_ecc_config() -> Self {
        Self {
            num_wires: 144,
            ..Self::standard_recursion_config()
        }
    }

    pub fn wide_ecc_config() -> Self {
        Self {
            num_wires: 234,
            ..Self::standard_recursion_config()
        }
    }

    pub fn standard_recursion_zk_config() -> Self {
        Self {
            zero_knowledge: true,
            ..Self::standard_recursion_config()
        }
    }

    /// Validate that the circuit config has valid parameters.
    ///
    /// This checks invariants that the PLONK protocol relies on for soundness.
    /// Returns an error message if validation fails.
    pub fn check_valid(&self) -> Result<(), &'static str> {
        if self.num_challenges == 0 {
            return Err("num_challenges must not be 0 (zero challenges means no verification)");
        }

        if self.num_constants == 0 {
            return Err("num_constants must not be 0 (causes infinite loop in circuit build)");
        }

        // num_routed_wires must be at least 3 for LookupTableGate (which uses 3 wires per slot)
        // and at least 2 for LookupGate (which uses 2 wires per slot).
        // We check >= 3 as the stricter bound.
        if self.num_routed_wires < 3 {
            return Err("num_routed_wires must be >= 3 (required for lookup gates)");
        }

        // Zero queries test nothing, leaving FRI with no soundness regardless of the other
        // parameters. (A rate of 1 is rejected below, since no quotient factor fits in it.)
        if self.fri_config.num_query_rounds == 0 {
            return Err("fri_config.num_query_rounds must not be 0 (FRI is never queried)");
        }

        if self.num_routed_wires > self.num_wires {
            return Err("num_routed_wires must not exceed num_wires");
        }
        // Every row is opened at `num_wires` points and loading allocates per routed wire, so
        // wider rows are not circuits but attempts to exhaust the loader. Standard configs use
        // a few hundred.
        if self.num_wires > Self::MAX_WIDTH || self.num_constants > Self::MAX_WIDTH {
            return Err("num_wires and num_constants must not exceed 2^16");
        }

        // The lookup argument batches `max_quotient_degree_factor - 1` slots per partial polynomial
        // and the permutation argument `max_quotient_degree_factor` wires, so neither can be zero,
        // and the prover computes no partial products at all unless the factor is below the
        // routed width.
        if self.max_quotient_degree_factor < 2 {
            return Err("max_quotient_degree_factor must be at least 2");
        }
        if self.max_quotient_degree_factor >= self.num_routed_wires {
            return Err("max_quotient_degree_factor must be below num_routed_wires");
        }
        // The quotient is split into `max_quotient_degree_factor` chunks, each committed at the
        // trace degree, so the LDE must be at least that many times larger than the trace.
        if crate::util::log2_ceil(self.max_quotient_degree_factor) > self.fri_config.rate_bits {
            return Err("max_quotient_degree_factor exceeds FRI rate_bits");
        }

        if self.fri_config.reduction_strategy.has_zero_arity() {
            return Err("FRI reduction arity bits must be non-zero");
        }

        Ok(())
    }

    /// Checks the parameters that only make sense inside `F`: the rate must leave room for a
    /// trace in the field's largest two-adic domain, and a `Fixed` schedule must fold a degree
    /// that domain can hold. The builder's blinding estimates shift and multiply by these before
    /// any trace degree is known, so they are bounded up front rather than per degree.
    pub fn check_fits_field<F: Field>(&self) -> Result<(), &'static str> {
        let largest_degree_bits = F::TWO_ADICITY
            .checked_sub(self.fri_config.rate_bits)
            .ok_or("FRI rate_bits exceed the field's two-adicity")?;
        if let FriReductionStrategy::Fixed(arity_bits) = &self.fri_config.reduction_strategy {
            arity_bits
                .iter()
                .try_fold(0usize, |total, &bits| total.checked_add(bits))
                .filter(|&total| total <= largest_degree_bits)
                .ok_or("FRI total reduction arity exceeds the field's largest domain")?;
        }
        Ok(())
    }

    /// Validate that the circuit config has valid parameters. Panics on failure.
    ///
    /// Use this at circuit build time where invalid config is a programmer error.
    /// Use `check_valid()` for deserialization where invalid data should return an error.
    pub fn validate(&self) {
        self.check_valid().expect("Invalid circuit config");
    }

    /// Checks the conjectured FRI security level (see the ethSTARK paper) against
    /// `security_bits`: each query contributes `rate_bits`, grinding adds `proof_of_work_bits`,
    /// and the challenges live in `F`'s degree-`D` extension, which caps the total.
    pub fn check_fri_security<F: Field + Extendable<D>, const D: usize>(
        &self,
    ) -> Result<(), &'static str> {
        let FriConfig {
            rate_bits,
            proof_of_work_bits,
            num_query_rounds,
            ..
        } = self.fri_config;
        let field_bits = F::Extension::order().bits() as usize;
        let query_bits = num_query_rounds
            .saturating_mul(rate_bits)
            .saturating_add(proof_of_work_bits as usize);
        if field_bits.min(query_bits) < self.security_bits {
            return Err("FRI params fall short of target security");
        }
        Ok(())
    }
}

/// The constructors of a `CommonCircuitData`. Both crates declare the struct with the same
/// fields over their own `GateRef`, so this expands inside each `impl` block and keeps one copy
/// of the load path. The expansion site must have `GateRef<F, D>` and its `Gate` trait in scope.
#[macro_export]
macro_rules! common_circuit_data_constructors {
    () => {
        /// Common data from the primitives `CircuitBuilder::build` fixes; every other field is
        /// derived from them, so these are all that is serialized.
        pub fn new(
            config: $crate::circuit_config::CircuitConfig,
            degree_bits: usize,
            gates: Vec<GateRef<F, D>>,
            num_public_inputs: usize,
            luts: Vec<$crate::plonk_common::LookupTable>,
        ) -> Result<Self, &'static str> {
            // The fit checks take logarithms of the config's widths, so the config comes first.
            config.check_valid()?;
            if !gates.iter().all(|gate| gate.0.parameters_fit(&config)) {
                return Err("gate parameters do not fit within the circuit config");
            }
            let gate_shapes: Vec<$crate::circuit_config::GateShape> = gates
                .iter()
                .map(|gate| $crate::circuit_config::GateShape {
                    id: gate.0.id(),
                    degree: gate.0.degree(),
                    num_wires: gate.0.num_wires(),
                    num_constants: gate.0.num_constants(),
                    num_constraints: gate.0.num_constraints(),
                })
                .collect();
            let derived = $crate::circuit_config::derive_common_data::<F, D>(
                &config,
                degree_bits,
                &gate_shapes,
                &luts,
            )?;
            Ok(Self {
                config,
                trace_degree_bits: degree_bits,
                fri_params: derived.fri_params,
                gates,
                selectors_info: derived.selectors_info,
                num_gate_constraints: derived.num_gate_constraints,
                num_constants: derived.num_constants,
                num_public_inputs,
                k_is: derived.k_is,
                num_partial_products: derived.num_partial_products,
                num_lookup_polys: derived.num_lookup_polys,
                num_lookup_selectors: derived.num_lookup_selectors,
                luts,
            })
        }

        /// Holds the config and lookup tables and nothing else, so gate readers can check their
        /// parameters and resolve table indices before the gates, and with them the derived
        /// fields, are known.
        pub(crate) fn tables_only(
            config: $crate::circuit_config::CircuitConfig,
            luts: Vec<$crate::plonk_common::LookupTable>,
        ) -> Self {
            Self {
                fri_params: $crate::fri::FriParams {
                    config: config.fri_config.clone(),
                    leaf_hiding: false,
                    degree_bits: 0,
                    reduction_arity_bits: Vec::new(),
                },
                config,
                trace_degree_bits: 0,
                gates: Vec::new(),
                selectors_info: $crate::selectors::SelectorsInfo {
                    selector_indices: Vec::new(),
                    groups: Vec::new(),
                },
                num_gate_constraints: 0,
                num_constants: 0,
                num_public_inputs: 0,
                k_is: Vec::new(),
                num_partial_products: 0,
                num_lookup_polys: 0,
                num_lookup_selectors: 0,
                luts,
            }
        }
    };
}

/// The gate-derived quantities the builder folds into `CommonCircuitData`, for one gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateShape {
    pub id: String,
    pub degree: usize,
    pub num_wires: usize,
    pub num_constants: usize,
    pub num_constraints: usize,
}

/// The fields of a `CommonCircuitData` that follow from its config, degree, gates and lookup
/// tables. Only those primitives are serialized; this is recomputed on load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedCommonData<F: Field> {
    pub fri_params: FriParams,
    pub selectors_info: SelectorsInfo,
    pub num_gate_constraints: usize,
    pub num_constants: usize,
    pub k_is: Vec<F>,
    pub num_partial_products: usize,
    pub num_lookup_polys: usize,
    pub num_lookup_selectors: usize,
}

/// Derives the rest of a `CommonCircuitData` the way `CircuitBuilder::build` does, after
/// checking that the primitives are ones the builder could have fixed: the config passes the
/// builder's checks, the FRI domain and schedule exist for the degree, the gates are sorted by
/// `(degree, id)` without repeats and fit the config, and no lookup table is empty.
///
/// The gate methods behind [`GateShape`] are only meaningful on parameters the builder could
/// have produced, so callers check every gate's `parameters_fit` before building the shapes.
///
/// This cannot tell which circuit the primitives describe, so a verifier must pin the whole
/// common data, not only the verifier key.
pub fn derive_common_data<F: Field + Extendable<D>, const D: usize>(
    config: &CircuitConfig,
    degree_bits: usize,
    gates: &[GateShape],
    luts: &[LookupTable],
) -> Result<DerivedCommonData<F>, &'static str> {
    config.check_valid()?;
    config.check_fri_security::<F, D>()?;
    let fri_params = config
        .fri_config
        .checked_fri_params::<F>(degree_bits, config.zero_knowledge)?;

    let sorted_without_duplicates = gates
        .windows(2)
        .all(|pair| (pair[0].degree, &pair[0].id) < (pair[1].degree, &pair[1].id));
    if gates.is_empty() || !sorted_without_duplicates {
        return Err("gates must be sorted by degree and id without duplicates");
    }
    if gates
        .iter()
        .any(|gate| gate.num_wires > config.num_wires || gate.num_constants > config.num_constants)
    {
        return Err("gate does not fit within the circuit config");
    }
    if luts.iter().any(|lut| lut.is_empty()) {
        return Err("lookup table is empty");
    }
    // The builder stores each table once and gates refer to it by index, so a repeated table
    // would give the same gate two encodings.
    if (1..luts.len()).any(|i| luts[..i].contains(&luts[i])) {
        return Err("lookup tables must be distinct");
    }

    let quotient_degree_factor = config.max_quotient_degree_factor;
    let degrees: Vec<usize> = gates.iter().map(|gate| gate.degree).collect();
    let selectors_info = SelectorsInfo::for_gate_degrees(&degrees, quotient_degree_factor + 1)?;
    let num_lookup_selectors = num_lookup_selectors(luts.len());
    let max_gate_constants = gates.iter().map(|gate| gate.num_constants).max().unwrap();
    let num_constants = selectors_info.num_selectors() + num_lookup_selectors + max_gate_constants;

    Ok(DerivedCommonData {
        fri_params,
        selectors_info,
        num_gate_constraints: gates.iter().map(|gate| gate.num_constraints).max().unwrap(),
        num_constants,
        k_is: get_unique_coset_shifts::<F>(1 << degree_bits, config.num_routed_wires),
        num_partial_products: num_partial_products(config.num_routed_wires, quotient_degree_factor),
        num_lookup_polys: num_lookup_polys(
            config.num_routed_wires,
            quotient_degree_factor - 1,
            luts.len(),
        ),
        num_lookup_selectors,
    })
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{string::ToString, sync::Arc, vec, vec::Vec};
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use plonky2_field::goldilocks_field::GoldilocksField;

    use super::*;

    #[test]
    fn standard_helpers_select_expected_zk_modes() {
        let disabled = CircuitConfig::standard_recursion_config();
        assert!(!disabled.zero_knowledge);

        let zk = CircuitConfig::standard_recursion_zk_config();
        assert!(zk.zero_knowledge);
    }

    #[test]
    fn circuit_config_validate_accepts_valid_num_challenges() {
        let config = CircuitConfig::standard_recursion_config();
        config.validate(); // Should not panic (num_challenges = 2)
    }

    #[test]
    #[should_panic(expected = "num_challenges must not be 0")]
    fn circuit_config_validate_rejects_zero_num_challenges() {
        let config = CircuitConfig {
            num_challenges: 0, // Invalid - voids soundness
            ..CircuitConfig::standard_recursion_config()
        };
        config.validate();
    }

    #[test]
    #[should_panic(expected = "num_constants must not be 0")]
    fn circuit_config_validate_rejects_zero_num_constants() {
        let config = CircuitConfig {
            num_constants: 0, // Invalid - causes infinite loop
            ..CircuitConfig::standard_recursion_config()
        };
        config.validate();
    }

    #[test]
    fn circuit_config_rejects_degenerate_fri() {
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.rate_bits = 0;
        assert!(config.check_valid().is_err());

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.num_query_rounds = 0;
        assert!(config.check_valid().is_err());

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::ConstantArityBits(0, 5);
        assert_eq!(
            config.check_valid(),
            Err("FRI reduction arity bits must be non-zero")
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![4, 0]);
        assert_eq!(
            config.check_valid(),
            Err("FRI reduction arity bits must be non-zero")
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.max_quotient_degree_factor = 1;
        assert_eq!(
            config.check_valid(),
            Err("max_quotient_degree_factor must be at least 2")
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_wires = CircuitConfig::MAX_WIDTH + 1;
        assert_eq!(
            config.check_valid(),
            Err("num_wires and num_constants must not exceed 2^16")
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.num_routed_wires = config.max_quotient_degree_factor;
        assert_eq!(
            config.check_valid(),
            Err("max_quotient_degree_factor must be below num_routed_wires")
        );

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.rate_bits = 2;
        assert_eq!(
            config.check_valid(),
            Err("max_quotient_degree_factor exceeds FRI rate_bits")
        );
    }

    /// The gate set a small recursion circuit has, as the builder would sort it.
    fn gates() -> Vec<GateShape> {
        [
            ("NoopGate", 0, 0, 0, 0),
            ("ConstantGate", 1, 2, 2, 2),
            ("PublicInputGate", 1, 4, 0, 4),
            ("ArithmeticGate", 3, 80, 2, 20),
            ("PoseidonGate", 7, 135, 0, 123),
            ("CosetInterpolationGate", 8, 64, 0, 24),
        ]
        .map(
            |(id, degree, num_wires, num_constants, num_constraints)| GateShape {
                id: id.to_string(),
                degree,
                num_wires,
                num_constants,
                num_constraints,
            },
        )
        .to_vec()
    }

    fn derive(
        config: &CircuitConfig,
        degree_bits: usize,
        gates: &[GateShape],
        luts: &[LookupTable],
    ) -> Result<DerivedCommonData<GoldilocksField>, &'static str> {
        derive_common_data::<GoldilocksField, 2>(config, degree_bits, gates, luts)
    }

    fn table(pairs: &[(u16, u16)]) -> LookupTable {
        Arc::new(pairs.to_vec())
    }

    #[test]
    fn derive_common_data_matches_the_builder_formulas() {
        let config = CircuitConfig::standard_recursion_config();
        let tables = [table(&[(0, 1), (1, 2)]), table(&[(0, 1)])];
        let derived = derive(&config, 12, &gates(), &tables).unwrap();
        assert_eq!(derived.fri_params, config.fri_config.fri_params(12, false));
        assert_eq!(derived.selectors_info.groups, vec![0..4, 4..5, 5..6]);
        assert_eq!(derived.num_gate_constraints, 123);
        assert_eq!(derived.num_lookup_selectors, 4 + 2);
        assert_eq!(derived.num_constants, 3 + 6 + 2);
        assert_eq!(derived.k_is.len(), config.num_routed_wires);
        assert_eq!(derived.num_partial_products, 9);
        assert_eq!(derived.num_lookup_polys, 7);
        assert_eq!(
            derive(&config, 12, &gates(), &[]).unwrap().num_constants,
            3 + 2
        );
    }

    /// The loader applies the config rules the builder applies, so consistent-looking data with
    /// too few queries is rejected the same way `CircuitBuilder::new` rejects its config.
    #[test]
    fn derive_common_data_rejects_configs_the_builder_rejects() {
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.num_query_rounds = 1;
        config.fri_config.proof_of_work_bits = 0;
        assert_eq!(
            derive(&config, 12, &gates(), &[]),
            Err("FRI params fall short of target security")
        );
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.rate_bits = 0;
        assert!(derive(&config, 12, &gates(), &[]).is_err());
    }

    #[test]
    fn derive_common_data_rejects_oversized_domains() {
        let config = CircuitConfig::standard_recursion_config();
        let too_large = Err("FRI LDE domain exceeds the field's two-adicity");
        assert_eq!(derive(&config, 30, &gates(), &[]), too_large);
        assert_eq!(derive(&config, usize::MAX, &gates(), &[]), too_large);
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.rate_bits = usize::MAX;
        config.fri_config.reduction_strategy = FriReductionStrategy::MinSize(None);
        assert_eq!(derive(&config, 12, &gates(), &[]), too_large);
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.cap_height = 16;
        assert_eq!(
            derive(&config, 12, &gates(), &[]),
            Err("FRI cap height exceeds the LDE size")
        );
    }

    /// A `Fixed` schedule is whatever the config says, so it is bounded after derivation.
    #[test]
    fn checked_fri_params_reject_schedules_that_overrun_the_domain() {
        let fixed = |arity_bits: Vec<usize>, cap_height| {
            FriConfig {
                cap_height,
                reduction_strategy: FriReductionStrategy::Fixed(arity_bits),
                ..CircuitConfig::standard_recursion_config().fri_config
            }
            .checked_fri_params::<GoldilocksField>(12, false)
        };
        assert_eq!(
            fixed(vec![4, 4], 4).map(|params| params.reduction_arity_bits),
            Ok(vec![4, 4])
        );
        assert_eq!(
            fixed(vec![8, 8], 4),
            Err("FRI reductions exceed the degree")
        );
        assert_eq!(
            fixed(vec![6, 6], 4),
            Err("FRI cap height plus reductions exceed the LDE size")
        );
        assert_eq!(
            fixed(vec![usize::MAX, 1], 4),
            Err("FRI reductions exceed the degree")
        );
    }

    #[test]
    fn derive_common_data_rejects_gates_the_builder_never_emits() {
        let config = CircuitConfig::standard_recursion_config();
        let unsorted = Err("gates must be sorted by degree and id without duplicates");
        assert_eq!(derive(&config, 12, &[], &[]), unsorted);
        let mut swapped = gates();
        swapped.swap(0, 1);
        assert_eq!(derive(&config, 12, &swapped, &[]), unsorted);
        let mut repeated = gates();
        repeated.insert(1, repeated[1].clone());
        assert_eq!(derive(&config, 12, &repeated, &[]), unsorted);

        let does_not_fit = Err("gate does not fit within the circuit config");
        let mut wide = gates();
        wide[3].num_wires = config.num_wires + 1;
        assert_eq!(derive(&config, 12, &wide, &[]), does_not_fit);
        let mut many_constants = gates();
        many_constants[3].num_constants = config.num_constants + 1;
        assert_eq!(derive(&config, 12, &many_constants, &[]), does_not_fit);

        let mut high_degree = gates();
        high_degree[5].degree = 9;
        assert_eq!(
            derive(&config, 12, &high_degree, &[]),
            Err("gate degree too high for the quotient degree factor")
        );
    }

    #[test]
    fn derive_common_data_rejects_empty_or_repeated_lookup_tables() {
        let config = CircuitConfig::standard_recursion_config();
        let pairs = [(0, 1), (1, 2)];
        assert_eq!(
            derive(&config, 12, &gates(), &[table(&pairs), table(&[])]),
            Err("lookup table is empty")
        );
        assert_eq!(
            derive(
                &config,
                12,
                &gates(),
                &[table(&pairs), table(&pairs[..1]), table(&pairs)]
            ),
            Err("lookup tables must be distinct")
        );
    }
}
