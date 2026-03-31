//! Circuit configuration types shared between prover and verifier.

use serde::Serialize;

use crate::fri::FriConfig;
use crate::fri::FriReductionStrategy;

/// Zero-knowledge settings for a circuit.
///
/// Leaf hiding remains an explicit knob because Merkle salting is orthogonal to whether we also
/// mask PLONK polynomials with the split-mask Poly/FRI design.
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
}

/// Supported zero-knowledge modes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ZkMode {
    Disabled,
    PolyFri(PolyFriZkConfig),
}

/// Poly/FRI split-mask tuning knobs.
///
/// These degrees control the prover-side masking polynomials. They replace builder-side row
/// blinding so proof generation keeps the native trace degree unchanged.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PolyFriZkConfig {
    pub wire_mask_degree: usize,
    pub z_mask_degree: usize,
    pub quotient_mask_degree: usize,
    pub fri_batch_mask_degree: usize,
}

impl PolyFriZkConfig {
    pub const fn standard_recursion() -> Self {
        Self {
            // Phase 1 masks only the explicit PLONK opening points. Wires and quotient chunks are
            // opened at one point, while the permutation/lookup oracle also opens at `g * zeta`.
            wire_mask_degree: 0,
            z_mask_degree: 1,
            quotient_mask_degree: 0,
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
    /// Zero-knowledge controls. `Disabled` reproduces the historical no-zk mode, while `PolyFri`
    /// enables the split-mask prover design without reintroducing builder-side blinding rows.
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

    pub fn standard_recursion_zk_config() -> Self {
        CircuitConfig {
            zk_config: ZkConfig {
                mode: ZkMode::PolyFri(PolyFriZkConfig::standard_recursion()),
                leaf_hiding: true,
            },
            ..Self::standard_recursion_config()
        }
    }

    pub fn uses_poly_fri_zk(&self) -> bool {
        matches!(&self.zk_config.mode, ZkMode::PolyFri(_))
    }

    pub const fn uses_leaf_hiding(&self) -> bool {
        self.zk_config.leaf_hiding
    }
}
