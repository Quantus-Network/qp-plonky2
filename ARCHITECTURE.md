# QP-Plonky2 Architecture

## Overview

This repository contains two main crates:
- **`qp-plonky2`** - Full plonky2 implementation with prover and verifier
- **`qp-plonky2-verifier`** - Lightweight verification-only crate

## Canonical Verification Logic

**The verification logic in `qp-plonky2-verifier` is the canonical source of truth for all verification algorithms.**

This means:
- Any bug fixes to verification code should be made in `qp-plonky2-verifier` first
- The verification logic is byte-for-byte identical between both crates
- Code changes should maintain this property

## Architecture Decision

`qp-plonky2` depends on `qp-plonky2-verifier` and re-exports its verification types:

```
qp-plonky2-verifier (canonical verification, no-std, no-rand)
         ↑
         ├─ Re-exported by qp-plonky2
         └─ Used by applications for verification
         
qp-plonky2 (full implementation with prover)
         ├─ Re-exports verification types from verifier
         ├─ Adds prover modules: batch_fri/, gadgets/, recursion/
         └─ Adds circuit building: plonk/circuit_builder.rs
```

## Module Overlap

Some modules exist in both crates:

### Identical Files
The following files are byte-for-byte identical:
- `fri/verifier.rs`, `fri/validate_shape.rs`
- `gates/util.rs`, `gates/packed_util.rs`
- `hash/poseidon*.rs`, `hash/keccak.rs`, `hash/arch/**/*.rs`
- `iop/target.rs`, `iop/wire.rs`
- `plonk/vars.rs`, `plonk/validate_shape.rs`
- `util/strided_view.rs`

### Verification Logic (Similar but with Prover Code Added)
These files in `qp-plonky2` contain the same verification logic as `qp-plonky2-verifier` plus additional prover code:
- `fri/challenges.rs`, `fri/proof.rs`, `fri/mod.rs` - Added prover: `oracle.rs`, `prover.rs`
- `gates/**` - Added: generator implementations
- `hash/**` - Added: merkle tree proving
- `iop/**` - Added: `generator.rs`, `witness.rs`
- `plonk/**` - Added: `circuit_builder.rs`, `prover.rs`, `permutation_argument.rs`
- `util/**` - Added: prover-specific serialization

## Guidelines for Changes

1. **Verification Bug Fixes**: Fix in `qp-plonky2-verifier` first, then apply the same fix to `qp-plonky2`
2. **New Prover Features**: Add directly to `qp-plonky2` only (e.g., new gates, proving optimization)
3. **Type Definitions**: Keep identical between both crates
4. **Serialization**: Must be compatible between verifier and prover

## Future Refactoring

Once Rust becomes better at code sharing (with more advanced macros or monomorphization control), these codebases could be fully merged. For now, the slight duplication ensures:
- `qp-plonky2-verifier` stays lightweight for WASM and no_std environments
- `qp-plonky2` can have prover-specific optimizations
- Clear separation of concerns
