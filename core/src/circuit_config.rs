//! Circuit configuration types shared between prover and verifier.

use serde::Serialize;

use crate::fri::{FriConfig, FriReductionStrategy};

/// Configuration to be used when building a circuit. This defines the shape of the circuit
/// as well as its targeted security level and sub-protocol (e.g. FRI) parameters.
///
/// It supports a [`Default`] implementation tailored for recursion with Poseidon hash (of width 12)
/// as internal hash function and FRI rate of 1/8.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
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
    use super::{check_common_data_valid, check_gate_shape, CircuitConfig};

    #[test]
    fn check_common_data_rejects_zero_quotient_degree() {
        let config = CircuitConfig::standard_recursion_config();
        let nrw = config.num_routed_wires;
        assert!(check_common_data_valid(&config, 0, 10, 4, 4, 0, nrw, || false).is_err());
    }

    #[test]
    fn check_common_data_validates_partial_products() {
        let config = CircuitConfig::standard_recursion_config();
        let nrw = config.num_routed_wires;
        let qdf = 8;
        let expected = nrw.div_ceil(qdf) - 1;
        assert!(check_common_data_valid(&config, qdf, 10, 4, 4, expected, nrw, || false).is_ok());
        assert!(
            check_common_data_valid(&config, qdf, 10, 4, 4, expected + 1, nrw, || false).is_err()
        );
    }

    #[test]
    fn check_common_data_rejects_wrong_k_is_length() {
        let config = CircuitConfig::standard_recursion_config();
        let nrw = config.num_routed_wires;
        let expected = nrw.div_ceil(8) - 1;
        assert!(
            check_common_data_valid(&config, 8, 10, 4, 4, expected, nrw - 1, || false).is_err()
        );
    }

    #[test]
    fn check_gate_shape_rejects_oversized_gates() {
        let config = CircuitConfig::standard_recursion_config();
        let nc = config.num_constants;
        assert!(check_gate_shape(config.num_wires, 0, 1, &config, 1, 0, nc, 4).is_ok());
        assert!(check_gate_shape(config.num_wires + 1, 0, 1, &config, 1, 0, nc, 4).is_err());
        assert!(check_gate_shape(1, nc, 1, &config, 1, 0, nc, 4).is_err());
        assert!(check_gate_shape(1, 0, 5, &config, 1, 0, nc, 4).is_err());
    }

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
}

/// Validate common circuit data fields shared between prover and verifier.
///
/// This is a free function to avoid duplication between `CommonCircuitData::check_valid`
/// (in plonky2) and `CommonVerifierData::check_valid` (in verifier).
///
/// # Arguments
/// * `config` - The circuit configuration
/// * `quotient_degree_factor` - The quotient polynomial degree factor
/// * `rate_bits` - FRI rate bits from config
/// * `public_initial_degree_bits` - Public initial FRI degree bits
/// * `trace_degree_bits` - Trace polynomial degree bits
/// * `num_partial_products` - Declared number of partial products
/// * `k_is_len` - Length of the permutation-argument coset shift vector
/// * `luts` - Lookup tables (as slice of slices for flexibility)
///
/// # Returns
/// `Ok(())` if valid, or an error message describing the validation failure.
pub fn check_common_data_valid(
    config: &CircuitConfig,
    quotient_degree_factor: usize,
    rate_bits: usize,
    public_initial_degree_bits: usize,
    trace_degree_bits: usize,
    num_partial_products: usize,
    k_is_len: usize,
    luts_empty_check: impl Fn() -> bool,
) -> Result<(), &'static str> {
    config.check_valid()?;

    // The permutation argument indexes one coset shift per routed wire.
    if k_is_len != config.num_routed_wires {
        return Err("k_is length must equal num_routed_wires");
    }

    // A zero quotient degree factor breaks partial-product chunking and quotient sizing.
    if quotient_degree_factor == 0 {
        return Err("quotient_degree_factor must not be 0");
    }

    // Quotient degree must fit within FRI rate.
    let quotient_degree_bits = crate::util::log2_ceil(quotient_degree_factor);
    if quotient_degree_bits > rate_bits {
        return Err("quotient_degree_factor exceeds FRI rate_bits");
    }

    // num_partial_products is fully determined by the routed-wire count and quotient degree;
    // an inconsistent value mis-sizes the partial-product portion of every proof.
    let expected_partial_products = config
        .num_routed_wires
        .div_ceil(quotient_degree_factor)
        .checked_sub(1)
        .ok_or("num_partial_products underflows")?;
    if num_partial_products != expected_partial_products {
        return Err("num_partial_products inconsistent with circuit config");
    }

    // Public initial degree must be at least as large as trace degree.
    if public_initial_degree_bits < trace_degree_bits {
        return Err("public_initial_degree_bits must be >= trace_degree_bits");
    }

    // All lookup tables must be non-empty.
    if luts_empty_check() {
        return Err("lookup table is empty");
    }

    Ok(())
}

/// Validate a single gate's declared shape against the surrounding circuit data.
///
/// Intended for deserialization of untrusted data: ensures a gate cannot reference more
/// wires, constants, or constraints than the circuit configuration accounts for, which
/// would otherwise cause out-of-bounds access during constraint evaluation.
///
/// `num_selectors` and `num_lookup_selectors` are the constant slots consumed before a
/// gate's own constants, so a gate's constants must fit in what remains.
pub fn check_gate_shape(
    gate_num_wires: usize,
    gate_num_constants: usize,
    gate_num_constraints: usize,
    config: &CircuitConfig,
    num_selectors: usize,
    num_lookup_selectors: usize,
    num_constants: usize,
    num_gate_constraints: usize,
) -> Result<(), &'static str> {
    if gate_num_wires > config.num_wires {
        return Err("gate uses more wires than the circuit configuration provides");
    }
    let constants_used = num_selectors
        .checked_add(num_lookup_selectors)
        .and_then(|prefix| prefix.checked_add(gate_num_constants))
        .ok_or("gate constant count overflows")?;
    if constants_used > num_constants {
        return Err("gate uses more constants than the circuit configuration provides");
    }
    if gate_num_constraints > num_gate_constraints {
        return Err("gate emits more constraints than num_gate_constraints");
    }
    Ok(())
}
