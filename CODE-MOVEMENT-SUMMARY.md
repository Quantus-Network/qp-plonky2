# Code Movement Summary

This document summarizes where code moved from `plonky2/src/` to the new `core/` and `verifier/` crates.

For the full detailed mapping (8600 lines), see `FUNCTION-MAPPING.md`.

## Overview

The refactoring created two new crates:
- **`core/`** - Shared types, FRI primitives, and utilities used by both prover and verifier
- **`verifier/`** - Standalone verification code (no prover dependencies)

## Key Structures Moved

### To `core/`

| Original Location | Structure | New Location |
|-------------------|-----------|--------------|
| `fri/mod.rs` | `FriConfig` | `core/src/fri.rs:193` |
| `fri/mod.rs` | `FriParams` | `core/src/fri.rs:238` |
| `fri/proof.rs` | `FriProof` | `core/src/fri_proof.rs:71` |
| `fri/proof.rs` | `CompressedFriProof` | `core/src/fri_proof.rs:85` |
| `fri/proof.rs` | `FriQueryStep` | `core/src/fri_proof.rs:24` |
| `fri/proof.rs` | `FriQueryRound` | `core/src/fri_proof.rs:51` |
| `fri/proof.rs` | `FriInitialTreeProof` | `core/src/fri_proof.rs:33` |
| `fri/proof.rs` | `FriChallenges` | `core/src/fri_structure.rs:72` |
| `fri/structure.rs` | `FriInstanceInfo` | `core/src/fri_structure.rs:14` |
| `fri/structure.rs` | `FriOracleInfo` | `core/src/fri_structure.rs:23` |
| `fri/structure.rs` | `FriBatchInfo` | `core/src/fri_structure.rs:30` |
| `fri/structure.rs` | `FriPolynomialInfo` | `core/src/fri_structure.rs:37` |
| `fri/structure.rs` | `FriOpenings` | `core/src/fri_structure.rs:60` |
| `fri/verifier.rs` | `PrecomputedReducedOpenings` | `core/src/fri_verifier.rs:248` |
| `fri/reduction_strategies.rs` | `FriReductionStrategy` | `core/src/fri.rs:22` |
| `gates/selectors.rs` | `SelectorsInfo` | `core/src/selectors.rs:13` |
| `iop/challenger.rs` | `Challenger` | `core/src/challenger.rs` |
| `hash/merkle_tree.rs` | `MerkleTree` | `core/src/merkle_tree.rs` |
| `hash/hash_types.rs` | `HashOut`, `MerkleCap` | `core/src/hash_types.rs` |
| `plonk/config.rs` | `GenericConfig`, `PoseidonGoldilocksConfig` | `core/src/config.rs` |

### To `verifier/`

| Original Location | Structure | New Location |
|-------------------|-----------|--------------|
| `gates/arithmetic_base.rs` | `ArithmeticGate` | `verifier/src/gates/arithmetic_base.rs:19` |
| `gates/arithmetic_extension.rs` | `ArithmeticExtensionGate` | `verifier/src/gates/arithmetic_extension.rs:16` |
| `gates/base_sum.rs` | `BaseSumGate` | `verifier/src/gates/base_sum.rs:22` |
| `gates/constant.rs` | `ConstantGate` | `verifier/src/gates/constant.rs:20` |
| `gates/coset_interpolation.rs` | `CosetInterpolationGate` | `verifier/src/gates/coset_interpolation.rs:46` |
| `gates/exponentiation.rs` | `ExponentiationGate` | `verifier/src/gates/exponentiation.rs:21` |
| `gates/gate.rs` | `Gate` trait, `CurrentSlot` | `verifier/src/gates/gate.rs` |
| `gates/lookup.rs` | `LookupGate` | `verifier/src/gates/lookup.rs:22` |
| `gates/lookup_table.rs` | `LookupTableGate` | `verifier/src/gates/lookup_table.rs:25` |
| `gates/multiplication_extension.rs` | `MulExtensionGate` | `verifier/src/gates/multiplication_extension.rs:16` |
| `gates/noop.rs` | `NoopGate` | `verifier/src/gates/noop.rs:13` |
| `gates/poseidon.rs` | `PoseidonGate` | `verifier/src/gates/poseidon.rs:22` |
| `gates/poseidon2.rs` | `Poseidon2Gate`, `Poseidon2Params` | `verifier/src/gates/poseidon2.rs` |
| `gates/poseidon2_int_mix.rs` | `Poseidon2IntMixGate` | `verifier/src/gates/poseidon2_int_mix.rs:38` |
| `gates/poseidon2_mds.rs` | `Poseidon2MdsGate` | `verifier/src/gates/poseidon2_mds.rs:22` |
| `gates/poseidon_mds.rs` | `PoseidonMdsGate` | `verifier/src/gates/poseidon_mds.rs:19` |
| `gates/public_input.rs` | `PublicInputGate` | `verifier/src/gates/public_input.rs:19` |
| `gates/random_access.rs` | `RandomAccessGate` | `verifier/src/gates/random_access.rs:22` |
| `gates/reducing.rs` | `ReducingGate` | `verifier/src/gates/reducing.rs:15` |
| `gates/reducing_extension.rs` | `ReducingExtensionGate` | `verifier/src/gates/reducing_extension.rs:15` |
| `hash/batch_merkle_tree.rs` | `BatchMerkleTree` | `verifier/src/hash/batch_merkle_tree.rs:17` |
| `plonk/circuit_data.rs` | `VerifierCircuitData`, `CommonCircuitData` | `verifier/src/plonk/circuit_data.rs` |
| `plonk/verifier.rs` | `verify` function | `verifier/src/plonk/verifier.rs` |
| `plonk/proof.rs` | `Proof`, `ProofWithPublicInputs` | `verifier/src/plonk/proof.rs` |

## Key Functions Moved

### FRI Verification (to `core/`)

| Function | From | To |
|----------|------|-----|
| `verify_fri_proof` | `fri/verifier.rs:62` | `core/src/fri_verifier.rs:64` |
| `fri_verify_initial_proof` | `fri/verifier.rs:111` | `core/src/fri_verifier.rs:113` |
| `fri_verifier_query_round` | `fri/verifier.rs:164` | `core/src/fri_verifier.rs:166` |
| `fri_combine_initial` | `fri/recursive_verifier.rs:334` | `core/src/fri_verifier.rs:125` |

### FRI Configuration (to `core/`)

| Function | From | To |
|----------|------|-----|
| `FriConfig::rate` | `fri/mod.rs:49` | `core/src/fri.rs:211` |
| `FriConfig::fri_params` | `fri/mod.rs:53` | `core/src/fri.rs:215` |
| `FriConfig::num_cap_elements` | `fri/mod.rs:68` | `core/src/fri.rs:230` |
| `FriParams::total_arities` | `fri/mod.rs:121` | `core/src/fri.rs:256` |
| `FriParams::lde_bits` | `fri/mod.rs:129` | `core/src/fri.rs:264` |
| `FriParams::lde_size` | `fri/mod.rs:133` | `core/src/fri.rs:268` |
| `reduction_arity_bits` | `fri/reduction_strategies.rs:32` | `core/src/fri.rs:41` |

### Merkle Tree (to `core/`)

| Function | From | To |
|----------|------|-----|
| `verify_merkle_proof` | `hash/merkle_proofs.rs:43` | `core/src/merkle_proofs.rs:30` |

## New Symbols (not in main)

These are genuinely new functions, traits, and types added in this PR - not moved from existing code.

> **Note:** Many functions that appear "new" are actually moved from different files in main (e.g., `eval_zero_poly` moved from `plonky2/src/plonk/plonk_common.rs` to `core/src/plonk_common.rs`). The list below contains only symbols that don't exist anywhere in main.

### New Traits (Verification Abstractions)

These new traits provide abstractions to separate verifier from prover:

| Location | Symbol | Purpose |
|----------|--------|---------|
| `core/src/fri.rs` | `trait FriConfigObserve` | Abstraction for observing FRI config in challenger |
| `core/src/fri.rs` | `trait FriParamsObserve` | Abstraction for observing FRI params in challenger |
| `core/src/fri.rs` | `trait FriChallenger` | Abstraction for FRI challenge generation |
| `verifier/src/gates/gate.rs` | `trait VerificationGate` | Core trait for gate verification (subset of full Gate trait) |
| `verifier/src/gates/gate.rs` | `trait AnyVerificationGate` | Type-erased verification gate trait |

### New Structs (Verification Infrastructure)

| Location | Symbol | Purpose |
|----------|--------|---------|
| `verifier/src/gates/gate.rs` | `VerificationGateRef` | Reference wrapper for verification gates |
| `verifier/src/gates/gate.rs` | `VerificationGateInstance` | Gate instance for verification |
| `verifier/src/gates/gate.rs` | `PrefixedVerificationGate` | Prefixed gate wrapper for verification |
| `verifier/src/plonk/circuit_data.rs` | `CommonVerifierData` | Verifier-specific subset of circuit data |

### New Accessors

These expose internal state that was previously private:

| Location | Symbol | Purpose |
|----------|--------|---------|
| `core/src/challenger.rs` | `Challenger::sponge_state()` | Access internal sponge state |
| `core/src/challenger.rs` | `Challenger::input_buffer()` | Access input buffer |

### New Type Aliases

| Location | Symbol | Purpose |
|----------|--------|---------|
| `verifier/src/gates/gate.rs` | `type GateRef` | Alias for verification gate reference |
| `verifier/src/gates/gate.rs` | `type GateInstance` | Alias for verification gate instance |
| `verifier/src/gates/gate.rs` | `type PrefixedGate` | Alias for prefixed verification gate |
| `verifier/src/plonk/circuit_data.rs` | `type CommonCircuitData` | Alias for common circuit data |

---

## What Stayed in `plonky2/`

The following remain in `plonky2/src/` (prover-only code):

- `plonk/prover.rs` - Proof generation
- `plonk/circuit_builder.rs` - Circuit construction
- `fri/prover.rs` - FRI proof generation
- `fri/oracle.rs` - Polynomial commitment
- `gadgets/*` - Circuit gadgets
- `recursion/*` - Recursive proof handling
- `iop/generator.rs` - Witness generation

---

## Re-exports

`plonky2/src/lib.rs` now re-exports key types from `core/` and `verifier/`:

```rust
pub use plonky2_verifier::verify;
pub use plonky2_verifier::{
    CommonCircuitData, CompressedProofWithPublicInputs, GenericConfig, 
    GenericHashOut, Hasher, PoseidonGoldilocksConfig, Proof, 
    ProofWithPublicInputs, VerifierCircuitData, VerifierOnlyCircuitData,
};
pub use plonky2_verifier::{C, D, F};
```

## Notes for Reviewers

1. **Gate implementations**: Each gate in `verifier/src/gates/` contains only the verification logic (`eval_unfiltered`, etc.). The prover-specific code (witness generation) stays in `plonky2/src/gates/`.

2. **Common trait methods**: `serialize`, `deserialize`, `new`, `default` appear in many files - these are standard trait implementations and don't need detailed review.

3. **Files with >80% similarity**: These are essentially moves with only import changes:
   - `core/src/strided_view.rs` (100%)
   - `core/src/arch/*` (100%)
   - `verifier/src/plonk/verifier.rs` (97%)
   - `verifier/src/gates/mod.rs` (91%)

4. **Files needing careful review**: These have significant changes due to code extraction:
   - `core/src/challenger.rs` (40% similar to original)
   - `core/src/fri.rs` (consolidates multiple FRI files)
   - `verifier/src/plonk/circuit_data.rs` (42% - prover code removed)

---

*Generated from map-functions.sh analysis*
