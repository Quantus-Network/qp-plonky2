//! Circuit configuration types shared between prover and verifier.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use serde::{Deserialize, Serialize};

use crate::field::cosets::get_unique_coset_shifts;
use crate::field::types::Field;
use crate::fri::{FriConfig, FriParams, FriReductionStrategy};
use crate::selectors::{LookupSelectors, SelectorsInfo};

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

        // A rate of 1 makes every codeword low-degree, and zero queries test nothing; either
        // leaves FRI with no soundness regardless of the other parameters.
        if self.fri_config.rate_bits == 0 {
            return Err("fri_config.rate_bits must not be 0 (FRI has no redundancy to test)");
        }
        if self.fri_config.num_query_rounds == 0 {
            return Err("fri_config.num_query_rounds must not be 0 (FRI is never queried)");
        }

        if self.num_routed_wires > self.num_wires {
            return Err("num_routed_wires must not exceed num_wires");
        }
        // Every row is opened at `num_wires` points, so wider rows are not circuits but attempts
        // to overflow the arithmetic gates perform on their widths.
        if self.num_wires > u32::MAX as usize || self.num_constants > u32::MAX as usize {
            return Err("num_wires and num_constants must fit in 32 bits");
        }

        // The lookup argument batches `max_quotient_degree_factor - 1` slots per partial polynomial
        // and the permutation argument `max_quotient_degree_factor` wires, so neither can be zero.
        if self.max_quotient_degree_factor < 2 {
            return Err("max_quotient_degree_factor must be at least 2");
        }

        // A zero arity never folds: `Fixed` would yield a step that reduces nothing and
        // `ConstantArityBits` would loop forever while deriving the schedule.
        let has_zero_arity = match &self.fri_config.reduction_strategy {
            FriReductionStrategy::Fixed(arity_bits) => arity_bits.contains(&0),
            FriReductionStrategy::ConstantArityBits(arity_bits, _) => *arity_bits == 0,
            FriReductionStrategy::MinSize(_) => false,
        };
        if has_zero_arity {
            return Err("FRI reduction arity bits must be non-zero");
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

/// A view of a `CommonCircuitData` (from either crate) for [`check_common_data_valid`].
pub struct CommonDataShape<'a, F: Field> {
    pub config: &'a CircuitConfig,
    pub trace_degree_bits: usize,
    pub fri_params: &'a FriParams,
    pub public_initial_degree_bits: usize,
    pub gates: &'a [GateShape],
    pub selectors_info: &'a SelectorsInfo,
    pub quotient_degree_factor: usize,
    pub num_gate_constraints: usize,
    pub num_constants: usize,
    pub k_is: &'a [F],
    pub num_partial_products: usize,
    pub num_lookup_polys: usize,
    pub num_lookup_selectors: usize,
    pub lut_lengths: &'a [usize],
}

/// Checks that common circuit data is exactly what `CircuitBuilder::build` produces from its
/// primitive inputs: the config, the degree, the gate set and the lookup tables. Every derived
/// field is recomputed and compared, so data that did not come from the builder is rejected
/// whether it was deserialized or constructed in memory.
///
/// The gate methods behind [`GateShape`] are assumed not to panic, which each gate's
/// deserializer guarantees by validating its own parameters.
pub fn check_common_data_valid<F: Field>(data: &CommonDataShape<F>) -> Result<(), &'static str> {
    let config = data.config;
    config.check_valid()?;

    // The builder derives all three degrees from the padded gate count. The verifier evaluates the
    // vanishing polynomial over `trace_degree_bits`, samples FRI query indices from
    // `public_initial_degree_bits` and checks the FRI proof over `fri_params.degree_bits`, so any
    // disagreement lets a proof be checked against a different domain than it was committed to.
    let degree_bits = data.trace_degree_bits;
    if degree_bits != data.public_initial_degree_bits || degree_bits != data.fri_params.degree_bits
    {
        return Err("trace_degree_bits, public_initial_degree_bits and FRI degree_bits must agree");
    }

    // Bound every exponent before anything adds or shifts by it: these come straight from the
    // bytes, and the release profile does not check arithmetic overflow.
    let fri_config = &config.fri_config;
    if degree_bits > F::TWO_ADICITY || fri_config.rate_bits > F::TWO_ADICITY - degree_bits {
        return Err("FRI LDE domain exceeds the field's two-adicity");
    }
    let lde_bits = degree_bits + fri_config.rate_bits;
    if lde_bits >= usize::BITS as usize {
        return Err("FRI LDE domain exceeds the address space");
    }
    if fri_config.cap_height > lde_bits {
        return Err("FRI cap height exceeds the LDE size");
    }

    if *data.fri_params != fri_config.fri_params(degree_bits, config.zero_knowledge) {
        return Err("fri_params must be derived from config");
    }
    // The reduction schedule must fit inside the domain: the folds do not exceed the degree, and
    // the cap still fits beneath them.
    let total_arities = data
        .fri_params
        .reduction_arity_bits
        .iter()
        .try_fold(0usize, |total, &arity_bits| total.checked_add(arity_bits))
        .filter(|&total| total <= degree_bits)
        .ok_or("FRI reductions exceed the degree")?;
    if fri_config.cap_height + total_arities > lde_bits {
        return Err("FRI cap height plus reductions exceed the LDE size");
    }

    let quotient_degree_factor = data.quotient_degree_factor;
    if quotient_degree_factor != config.max_quotient_degree_factor {
        return Err("quotient_degree_factor must equal config.max_quotient_degree_factor");
    }
    if crate::util::log2_ceil(quotient_degree_factor) > fri_config.rate_bits {
        return Err("quotient_degree_factor exceeds FRI rate_bits");
    }

    let gates = data.gates;
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

    let degrees: Vec<usize> = gates.iter().map(|gate| gate.degree).collect();
    if *data.selectors_info
        != SelectorsInfo::for_gate_degrees(&degrees, quotient_degree_factor + 1)?
    {
        return Err("selectors_info must be derived from the gates");
    }

    let num_luts = data.lut_lengths.len();
    if data.lut_lengths.contains(&0) {
        return Err("lookup table is empty");
    }
    let (num_lookup_polys, num_lookup_selectors) = if num_luts == 0 {
        (0, 0)
    } else {
        // One RE polynomial plus the partial Sum/LDC polynomials over `LookupGate`'s two-wire
        // slots; the four shared lookup selectors plus one end selector per table.
        let lookup_slots = config.num_routed_wires / 2;
        (
            lookup_slots.div_ceil(quotient_degree_factor - 1) + 1,
            LookupSelectors::StartEnd as usize + num_luts,
        )
    };
    if data.num_lookup_polys != num_lookup_polys {
        return Err("num_lookup_polys must be derived from the config and lookup tables");
    }
    if data.num_lookup_selectors != num_lookup_selectors {
        return Err("num_lookup_selectors must be derived from the lookup tables");
    }

    let max_gate_constants = gates.iter().map(|gate| gate.num_constants).max().unwrap();
    if data.num_constants
        != data.selectors_info.num_selectors() + num_lookup_selectors + max_gate_constants
    {
        return Err("num_constants must be derived from the selectors and gates");
    }
    if data.num_gate_constraints != gates.iter().map(|gate| gate.num_constraints).max().unwrap() {
        return Err("num_gate_constraints must be the maximum over the gates");
    }
    if data.num_partial_products != config.num_routed_wires.div_ceil(quotient_degree_factor) - 1 {
        return Err("num_partial_products must be derived from the config");
    }
    if data.k_is.len() != config.num_routed_wires
        || data.k_is != get_unique_coset_shifts::<F>(1 << degree_bits, config.num_routed_wires)
    {
        return Err("k_is must be the unique coset shifts for the degree");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{string::ToString, vec, vec::Vec};

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
        config.num_wires = usize::MAX;
        assert_eq!(
            config.check_valid(),
            Err("num_wires and num_constants must fit in 32 bits")
        );
    }

    /// An owned `CommonCircuitData` stand-in derived the way the builder derives it, so tests can
    /// corrupt a single field and check that exactly that corruption is reported.
    struct CommonData {
        config: CircuitConfig,
        degree_bits: usize,
        fri_params: FriParams,
        public_initial_degree_bits: usize,
        gates: Vec<GateShape>,
        selectors_info: SelectorsInfo,
        quotient_degree_factor: usize,
        num_gate_constraints: usize,
        num_constants: usize,
        k_is: Vec<GoldilocksField>,
        num_partial_products: usize,
        num_lookup_polys: usize,
        num_lookup_selectors: usize,
        lut_lengths: Vec<usize>,
    }

    impl CommonData {
        fn new(num_luts: usize) -> Self {
            let config = CircuitConfig::standard_recursion_config();
            let degree_bits = 12;
            let gates = [
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
            .to_vec();
            let quotient_degree_factor = config.max_quotient_degree_factor;
            let degrees: Vec<usize> = gates.iter().map(|gate| gate.degree).collect();
            let selectors_info =
                SelectorsInfo::for_gate_degrees(&degrees, quotient_degree_factor + 1).unwrap();
            let num_lookup_selectors = if num_luts == 0 { 0 } else { 4 + num_luts };
            let num_lookup_polys = if num_luts == 0 {
                0
            } else {
                (config.num_routed_wires / 2).div_ceil(quotient_degree_factor - 1) + 1
            };
            Self {
                fri_params: config
                    .fri_config
                    .fri_params(degree_bits, config.zero_knowledge),
                public_initial_degree_bits: degree_bits,
                num_gate_constraints: gates.iter().map(|gate| gate.num_constraints).max().unwrap(),
                num_constants: selectors_info.num_selectors() + num_lookup_selectors + 2,
                k_is: get_unique_coset_shifts(1 << degree_bits, config.num_routed_wires),
                num_partial_products: config.num_routed_wires.div_ceil(quotient_degree_factor) - 1,
                lut_lengths: vec![16; num_luts],
                config,
                degree_bits,
                gates,
                selectors_info,
                quotient_degree_factor,
                num_lookup_polys,
                num_lookup_selectors,
            }
        }

        fn check(&self) -> Result<(), &'static str> {
            check_common_data_valid(&CommonDataShape {
                config: &self.config,
                trace_degree_bits: self.degree_bits,
                fri_params: &self.fri_params,
                public_initial_degree_bits: self.public_initial_degree_bits,
                gates: &self.gates,
                selectors_info: &self.selectors_info,
                quotient_degree_factor: self.quotient_degree_factor,
                num_gate_constraints: self.num_gate_constraints,
                num_constants: self.num_constants,
                k_is: &self.k_is,
                num_partial_products: self.num_partial_products,
                num_lookup_polys: self.num_lookup_polys,
                num_lookup_selectors: self.num_lookup_selectors,
                lut_lengths: &self.lut_lengths,
            })
        }
    }

    fn corrupted(mutate: impl FnOnce(&mut CommonData)) -> Result<(), &'static str> {
        let mut data = CommonData::new(1);
        mutate(&mut data);
        data.check()
    }

    #[test]
    fn check_common_data_accepts_builder_output() {
        assert_eq!(CommonData::new(0).check(), Ok(()));
        assert_eq!(CommonData::new(3).check(), Ok(()));
    }

    /// #64700: FRI query indices are sampled from `public_initial_degree_bits` and the vanishing
    /// polynomial is evaluated over `trace_degree_bits`, so both must equal the committed FRI
    /// `degree_bits`; otherwise a proof is checked against a different domain than it was
    /// committed to.
    #[test]
    fn check_common_data_rejects_mismatched_degree_bits() {
        let degrees_disagree =
            Err("trace_degree_bits, public_initial_degree_bits and FRI degree_bits must agree");
        assert_eq!(
            corrupted(|data| data.public_initial_degree_bits -= 1),
            degrees_disagree
        );
        assert_eq!(
            corrupted(|data| data.fri_params.degree_bits += 1),
            degrees_disagree
        );
        assert_eq!(
            corrupted(|data| {
                data.degree_bits += 1;
                data.public_initial_degree_bits += 1;
            }),
            degrees_disagree
        );
    }

    #[test]
    fn check_common_data_rejects_oversized_domains() {
        let huge = |degree_bits: usize| {
            corrupted(|data| {
                data.degree_bits = degree_bits;
                data.public_initial_degree_bits = degree_bits;
                data.fri_params.degree_bits = degree_bits;
            })
        };
        assert_eq!(
            huge(30),
            Err("FRI LDE domain exceeds the field's two-adicity")
        );
        assert_eq!(
            huge(usize::MAX),
            Err("FRI LDE domain exceeds the field's two-adicity")
        );
        assert_eq!(
            corrupted(|data| {
                data.config.fri_config.rate_bits = usize::MAX;
                data.fri_params.config.rate_bits = usize::MAX;
            }),
            Err("FRI LDE domain exceeds the field's two-adicity")
        );
        assert_eq!(
            corrupted(|data| {
                data.config.fri_config.cap_height = 16;
                data.fri_params.config.cap_height = 16;
            }),
            Err("FRI cap height exceeds the LDE size")
        );
    }

    /// The verifier runs with `fri_params`, so they must be exactly what the validated config
    /// derives and not a separately supplied copy with, say, zero query rounds or a reduction
    /// schedule the config would never produce.
    #[test]
    fn check_common_data_rejects_divergent_fri_params() {
        let derived = Err("fri_params must be derived from config");
        assert_eq!(
            corrupted(|data| data.fri_params.config.num_query_rounds = 0),
            derived
        );
        assert_eq!(
            corrupted(|data| data.fri_params.leaf_hiding = true),
            derived
        );
        assert_eq!(
            corrupted(|data| data.fri_params.reduction_arity_bits.push(0)),
            derived
        );
        assert_eq!(
            corrupted(|data| data.fri_params.reduction_arity_bits = vec![6, 5]),
            derived
        );
        assert_eq!(
            corrupted(|data| data.fri_params.reduction_arity_bits = vec![4]),
            derived
        );
    }

    #[test]
    fn check_common_data_rejects_divergent_quotient_degree_factor() {
        assert_eq!(
            corrupted(|data| data.quotient_degree_factor = 4),
            Err("quotient_degree_factor must equal config.max_quotient_degree_factor")
        );
        assert_eq!(
            corrupted(|data| {
                data.config.fri_config.rate_bits = 2;
                data.fri_params.config.rate_bits = 2;
                data.fri_params.reduction_arity_bits = vec![4, 4];
            }),
            Err("quotient_degree_factor exceeds FRI rate_bits")
        );
    }

    #[test]
    fn check_common_data_rejects_divergent_gates() {
        let unsorted = Err("gates must be sorted by degree and id without duplicates");
        assert_eq!(corrupted(|data| data.gates.clear()), unsorted);
        assert_eq!(corrupted(|data| data.gates.swap(0, 1)), unsorted);
        assert_eq!(
            corrupted(|data| {
                let duplicate = data.gates[1].clone();
                data.gates.insert(1, duplicate);
            }),
            unsorted
        );
        assert_eq!(
            corrupted(|data| data.gates[3].num_wires = data.config.num_wires + 1),
            Err("gate does not fit within the circuit config")
        );
        assert_eq!(
            corrupted(|data| data.gates[3].num_constants = data.config.num_constants + 1),
            Err("gate does not fit within the circuit config")
        );
        assert_eq!(
            corrupted(|data| data.gates[5].degree = 9),
            Err("gate degree too high for the quotient degree factor")
        );
        assert_eq!(
            corrupted(|data| data.selectors_info.groups.truncate(1)),
            Err("selectors_info must be derived from the gates")
        );
        assert_eq!(
            corrupted(|data| data.num_gate_constraints += 1),
            Err("num_gate_constraints must be the maximum over the gates")
        );
        assert_eq!(
            corrupted(|data| data.num_constants += 1),
            Err("num_constants must be derived from the selectors and gates")
        );
    }

    #[test]
    fn check_common_data_rejects_divergent_lookup_counts() {
        assert_eq!(
            corrupted(|data| data.lut_lengths[0] = 0),
            Err("lookup table is empty")
        );
        assert_eq!(
            corrupted(|data| data.num_lookup_polys = 0),
            Err("num_lookup_polys must be derived from the config and lookup tables")
        );
        assert_eq!(
            corrupted(|data| data.num_lookup_selectors += 1),
            Err("num_lookup_selectors must be derived from the lookup tables")
        );
    }

    #[test]
    fn check_common_data_rejects_divergent_permutation_data() {
        assert_eq!(
            corrupted(|data| data.num_partial_products += 1),
            Err("num_partial_products must be derived from the config")
        );
        assert_eq!(
            corrupted(|data| data.k_is.truncate(10)),
            Err("k_is must be the unique coset shifts for the degree")
        );
        assert_eq!(
            corrupted(|data| data.k_is[3] = data.k_is[2]),
            Err("k_is must be the unique coset shifts for the degree")
        );
    }
}
