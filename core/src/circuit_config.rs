//! Circuit configuration types shared between prover and verifier.

use serde::{Deserialize, Serialize};

use crate::fri::{FriConfig, FriParams, FriReductionStrategy};

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

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::{CircuitConfig, FriParams};

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

    fn fri_params_for(config: &CircuitConfig, degree_bits: usize) -> FriParams {
        config
            .fri_config
            .fri_params(degree_bits, config.zero_knowledge)
    }

    fn check(
        config: &CircuitConfig,
        fri_params: &FriParams,
        public_initial_degree_bits: usize,
    ) -> Result<(), &'static str> {
        check_with_trace(
            config,
            fri_params,
            public_initial_degree_bits,
            public_initial_degree_bits,
        )
    }

    fn check_with_trace(
        config: &CircuitConfig,
        fri_params: &FriParams,
        public_initial_degree_bits: usize,
        trace_degree_bits: usize,
    ) -> Result<(), &'static str> {
        super::check_common_data_valid(
            config,
            1,
            fri_params,
            public_initial_degree_bits,
            trace_degree_bits,
            32,
            || false,
        )
    }

    /// #64700: FRI query indices are sampled from `public_initial_degree_bits`, so it must equal the
    /// transcript-bound FRI `degree_bits`; otherwise queries are drawn from a smaller domain than
    /// the proof is checked against, weakening soundness.
    #[test]
    fn check_common_data_rejects_mismatched_degree_bits() {
        let config = CircuitConfig::standard_recursion_config();
        let fri_params = fri_params_for(&config, 10);
        assert!(check(&config, &fri_params, 10).is_ok());
        assert_eq!(
            check(&config, &fri_params, 9),
            Err("public_initial_degree_bits must match FRI degree_bits")
        );
    }

    /// The vanishing polynomial is evaluated over `trace_degree_bits`; a smaller value than the
    /// committed domain passed deserialization before, so pin all three degrees to each other.
    #[test]
    fn check_common_data_rejects_mismatched_trace_degree_bits() {
        let config = CircuitConfig::standard_recursion_config();
        let fri_params = fri_params_for(&config, 10);
        for trace_degree_bits in [9, 11] {
            assert_eq!(
                check_with_trace(&config, &fri_params, 10, trace_degree_bits),
                Err("trace_degree_bits must match public_initial_degree_bits")
            );
        }
    }

    /// The verifier runs with `fri_params.config`, so it must be the validated `config.fri_config`
    /// and not a separately supplied copy with, say, zero query rounds.
    #[test]
    fn check_common_data_rejects_divergent_fri_params() {
        let config = CircuitConfig::standard_recursion_config();
        let mut fri_params = fri_params_for(&config, 10);
        fri_params.config.num_query_rounds = 0;
        assert_eq!(
            check(&config, &fri_params, 10),
            Err("fri_params.config must match config.fri_config")
        );

        let mut fri_params = fri_params_for(&config, 10);
        fri_params.leaf_hiding = true;
        assert_eq!(
            check(&config, &fri_params, 10),
            Err("fri_params.leaf_hiding must match config.zero_knowledge")
        );
    }

    #[test]
    fn check_common_data_rejects_unfittable_reduction_schedules() {
        let config = CircuitConfig::standard_recursion_config();

        let mut fri_params = fri_params_for(&config, 10);
        fri_params.reduction_arity_bits.push(0);
        assert_eq!(
            check(&config, &fri_params, 10),
            Err("FRI reduction arity bits must be non-zero")
        );

        let mut fri_params = fri_params_for(&config, 10);
        fri_params.reduction_arity_bits = vec![6, 5];
        assert_eq!(
            check(&config, &fri_params, 10),
            Err("FRI reductions exceed the degree")
        );

        let mut fri_params = fri_params_for(&config, 10);
        fri_params.reduction_arity_bits = vec![5, 5];
        assert_eq!(
            check(&config, &fri_params, 10),
            Err("FRI cap height plus reductions exceed the LDE size")
        );

        let fri_params = fri_params_for(&config, 30);
        assert_eq!(
            check(&config, &fri_params, 30),
            Err("FRI LDE domain exceeds the field's two-adicity")
        );
    }

    #[test]
    fn circuit_config_rejects_degenerate_fri() {
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.rate_bits = 0;
        assert!(config.check_valid().is_err());

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.num_query_rounds = 0;
        assert!(config.check_valid().is_err());
    }
}

/// Validate common circuit data fields shared between prover and verifier.
///
/// This is a free function to avoid duplication between `CommonCircuitData::check_valid`
/// (in plonky2) and `CommonVerifierData::check_valid` (in verifier).
///
/// # Arguments
/// * `config` - The circuit configuration
/// * `quotient_degree_factor` - The quotient polynomial degree factor
/// * `fri_params` - The instance-specific FRI parameters the verifier runs with
/// * `public_initial_degree_bits` - Public initial FRI degree bits
/// * `trace_degree_bits` - Trace polynomial degree bits
/// * `two_adicity` - The field's two-adicity, bounding the LDE domain size
/// * `luts_empty_check` - Whether any lookup table is empty
///
/// # Returns
/// `Ok(())` if valid, or an error message describing the validation failure.
pub fn check_common_data_valid(
    config: &CircuitConfig,
    quotient_degree_factor: usize,
    fri_params: &FriParams,
    public_initial_degree_bits: usize,
    trace_degree_bits: usize,
    two_adicity: usize,
    luts_empty_check: impl Fn() -> bool,
) -> Result<(), &'static str> {
    config.check_valid()?;

    // The verifier reads its FRI parameters from `fri_params`, so the validated `config` must be
    // what it actually runs with; the builder derives one from the other.
    if fri_params.config != config.fri_config {
        return Err("fri_params.config must match config.fri_config");
    }
    if fri_params.leaf_hiding != config.zero_knowledge {
        return Err("fri_params.leaf_hiding must match config.zero_knowledge");
    }

    // Quotient degree must fit within FRI rate.
    let quotient_degree_bits = crate::util::log2_ceil(quotient_degree_factor);
    if quotient_degree_bits > config.fri_config.rate_bits {
        return Err("quotient_degree_factor exceeds FRI rate_bits");
    }

    // The builder derives all three degrees from the padded gate count. The verifier evaluates the
    // vanishing polynomial over `trace_degree_bits`, samples FRI query indices from
    // `public_initial_degree_bits` and checks the FRI proof over `fri_params.degree_bits`, so any
    // disagreement lets a proof be checked against a different domain than it was committed to.
    if trace_degree_bits != public_initial_degree_bits {
        return Err("trace_degree_bits must match public_initial_degree_bits");
    }
    if public_initial_degree_bits != fri_params.degree_bits {
        return Err("public_initial_degree_bits must match FRI degree_bits");
    }

    // The LDE domain must exist in the field, and the reduction schedule must fit inside it:
    // every step folds, the folds do not exceed the degree, and the cap still fits beneath them.
    if fri_params.lde_bits() > two_adicity {
        return Err("FRI LDE domain exceeds the field's two-adicity");
    }
    if fri_params.reduction_arity_bits.contains(&0) {
        return Err("FRI reduction arity bits must be non-zero");
    }
    let total_arities = fri_params.total_arities();
    if total_arities > fri_params.degree_bits {
        return Err("FRI reductions exceed the degree");
    }
    if fri_params.config.cap_height + total_arities > fri_params.lde_bits() {
        return Err("FRI cap height plus reductions exceed the LDE size");
    }

    // All lookup tables must be non-empty.
    if luts_empty_check() {
        return Err("lookup table is empty");
    }

    Ok(())
}
