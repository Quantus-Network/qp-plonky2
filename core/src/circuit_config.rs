//! Circuit configuration types shared between prover and verifier.

use serde::Serialize;

use crate::fri::{FriConfig, FriReductionStrategy};

/// Zero-knowledge settings for a circuit.
///
/// Leaf hiding remains an explicit knob because Merkle salting is orthogonal to whether we also
/// mask PLONK polynomials with the PolyFri logical-mask design.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ZkConfig {
    pub mode: ZkMode,
    pub leaf_hiding: bool,
}

impl ZkConfig {
    pub const fn disabled() -> Self {
        Self {
            mode: ZkMode::Disabled,
            leaf_hiding: false,
        }
    }

    pub const fn row_blinding() -> Self {
        Self {
            mode: ZkMode::RowBlinding,
            leaf_hiding: true,
        }
    }

    pub const fn poly_fri(poly_fri: PolyFriZkConfig) -> Self {
        Self {
            mode: ZkMode::PolyFri(poly_fri),
            leaf_hiding: true,
        }
    }
}

/// Supported zero-knowledge modes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ZkMode {
    Disabled,
    RowBlinding,
    PolyFri(PolyFriZkConfig),
}

/// Poly/FRI logical-mask tuning knobs.
///
/// These degrees control the prover-side masking polynomials. Unlike row blinding, they keep the
/// native trace degree unchanged and move zero-knowledge masking into the oracle commitments via
/// logical masked polynomials `M(X) = f(X) + (X^n - 1)r(X)`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PolyFriZkConfig {
    pub wire_mask_degree: usize,
    pub z_mask_degree: usize,
    pub fri_batch_mask_degree: usize,
}

impl PolyFriZkConfig {
    fn assert_degree_fits(
        knob_name: &str,
        mask_degree: usize,
        split_degree: usize,
        split_name: &str,
    ) {
        assert!(
            mask_degree < split_degree,
            "Invalid PolyFri config: `{knob_name}` must be less than the {split_name} ({split_degree}), got {mask_degree}",
        );
    }

    /// Returns the public initial FRI degree bits needed to hold the logical masked polynomial.
    ///
    /// Phase 1 masks commit to `M(X) = f(X) + (X^n - 1)r(X)`, so the public initial codeword
    /// must be large enough for the masked logical degree rather than the raw trace degree alone.
    pub fn public_initial_degree_bits(trace_degree: usize, max_mask_degree: usize) -> usize {
        let required_degree = trace_degree
            .checked_add(max_mask_degree)
            .and_then(|degree| degree.checked_add(1))
            .expect("PolyFri public initial degree overflowed usize");
        required_degree.next_power_of_two().trailing_zeros() as usize
    }

    pub const fn standard_recursion() -> Self {
        Self {
            // Phase 1 masks only the explicit PLONK opening points. Wires are opened at one point,
            // while the permutation/lookup oracle also opens at `g * zeta`.
            wire_mask_degree: 0,
            z_mask_degree: 1,
            // Phase 2 commits an explicit batch-mask oracle before `fri_alpha`, so native PolyFri
            // keeps the masked FRI reduction transcript-visible and verifier-consistent.
            fri_batch_mask_degree: 1,
        }
    }
}

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
    /// Zero-knowledge controls. `Disabled` reproduces the historical no-zk mode, `RowBlinding`
    /// restores the legacy builder-side blinding strategy, and `PolyFri` enables logical masked
    /// phase-1 commitments together with the explicit batch-mask path for phase 2.
    pub zk_config: ZkConfig,
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
            zk_config: ZkConfig::disabled(),
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

    pub fn standard_recursion_row_blinding_zk_config() -> Self {
        Self {
            zk_config: ZkConfig::row_blinding(),
            ..Self::standard_recursion_config()
        }
    }

    pub fn standard_recursion_zk_config() -> Self {
        Self::standard_recursion_row_blinding_zk_config()
    }

    pub fn standard_recursion_polyfri_zk_config() -> Self {
        Self {
            zk_config: ZkConfig::poly_fri(PolyFriZkConfig::standard_recursion()),
            ..Self::standard_recursion_config()
        }
    }

    pub const fn uses_poly_fri_zk(&self) -> bool {
        matches!(&self.zk_config.mode, ZkMode::PolyFri(_))
    }

    pub const fn uses_row_blinding_zk(&self) -> bool {
        matches!(&self.zk_config.mode, ZkMode::RowBlinding)
    }

    pub const fn uses_leaf_hiding(&self) -> bool {
        self.zk_config.leaf_hiding
    }

    /// Validate PolyFri-specific degree knobs once the builder knows the concrete trace/FRI sizes.
    pub fn validate_poly_fri_params(
        &self,
        trace_degree: usize,
        public_initial_degree_bits: usize,
        batch_mask_chunk_degree: usize,
        has_lookup: bool,
    ) {
        let ZkMode::PolyFri(poly_fri) = &self.zk_config.mode else {
            return;
        };

        PolyFriZkConfig::assert_degree_fits(
            "wire_mask_degree",
            poly_fri.wire_mask_degree,
            trace_degree,
            "trace degree",
        );
        PolyFriZkConfig::assert_degree_fits(
            "z_mask_degree",
            poly_fri.z_mask_degree,
            trace_degree,
            "trace degree",
        );
        let max_phase1_mask_degree = poly_fri.wire_mask_degree.max(poly_fri.z_mask_degree);
        let expected_public_initial_degree_bits =
            PolyFriZkConfig::public_initial_degree_bits(trace_degree, max_phase1_mask_degree);
        assert!(
            public_initial_degree_bits == expected_public_initial_degree_bits,
            "Invalid PolyFri config: public initial degree bits must be {expected_public_initial_degree_bits} for trace degree {trace_degree} and max phase-1 mask degree {max_phase1_mask_degree}, got {public_initial_degree_bits}",
        );
        PolyFriZkConfig::assert_degree_fits(
            "fri_batch_mask_degree",
            poly_fri.fri_batch_mask_degree,
            batch_mask_chunk_degree,
            "FRI batch-mask chunk degree",
        );

        assert!(
            self.max_quotient_degree_factor >= 2,
            "Invalid PolyFri config: `max_quotient_degree_factor` must be at least 2 so masked permutation chunks retain positive degree, got {}",
            self.max_quotient_degree_factor,
        );
        if has_lookup {
            assert!(
                self.max_quotient_degree_factor >= 3,
                "Invalid PolyFri config: `max_quotient_degree_factor` must be at least 3 when lookups are enabled so masked lookup accumulators retain positive degree, got {}",
                self.max_quotient_degree_factor,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CircuitConfig, PolyFriZkConfig, ZkMode};

    #[test]
    fn standard_helpers_select_expected_zk_modes() {
        let disabled = CircuitConfig::standard_recursion_config();
        assert_eq!(disabled.zk_config.mode, ZkMode::Disabled);
        assert!(!disabled.uses_leaf_hiding());
        assert!(!disabled.uses_row_blinding_zk());
        assert!(!disabled.uses_poly_fri_zk());

        let row_blinding = CircuitConfig::standard_recursion_zk_config();
        assert_eq!(row_blinding.zk_config.mode, ZkMode::RowBlinding);
        assert!(row_blinding.uses_leaf_hiding());
        assert!(row_blinding.uses_row_blinding_zk());
        assert!(!row_blinding.uses_poly_fri_zk());

        let polyfri = CircuitConfig::standard_recursion_polyfri_zk_config();
        assert!(matches!(polyfri.zk_config.mode, ZkMode::PolyFri(_)));
        assert!(polyfri.uses_leaf_hiding());
        assert!(!polyfri.uses_row_blinding_zk());
        assert!(polyfri.uses_poly_fri_zk());
    }

    #[test]
    fn polyfri_public_initial_degree_bits_rounds_up_for_masked_polys() {
        assert_eq!(PolyFriZkConfig::public_initial_degree_bits(8, 0), 4);
        assert_eq!(PolyFriZkConfig::public_initial_degree_bits(8, 7), 4);
        assert_eq!(PolyFriZkConfig::public_initial_degree_bits(8, 8), 5);
    }

    #[test]
    fn validate_polyfri_params_accepts_derived_public_degree() {
        let config = CircuitConfig::standard_recursion_polyfri_zk_config();
        let trace_degree = 1 << 3;
        let public_initial_degree_bits =
            PolyFriZkConfig::public_initial_degree_bits(trace_degree, 1);

        config.validate_poly_fri_params(trace_degree, public_initial_degree_bits, 2, false);
    }
}
