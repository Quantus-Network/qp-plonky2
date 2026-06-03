# V12 audit (4/10): FRI artifact validation, Merkle binding & strided-view memory safety

Evaluation of six high-severity V12 findings against current `main` (post `Remove PolyFRI (#50)`),
and the fixes applied to those still valid.

## Summary

| Finding | Title | Status | Fix |
|---|---|---|---|
| #64696 | Malformed oracle metadata panics verifier | Valid (reshaped) | Batch-aware reference check in FRI shape validation |
| #64699 | Malformed split layout panics verifier | Not valid | Target code removed with PolyFRI |
| #64700 | Mismatched degree bits weaken FRI sampling | Valid | Bind `public_initial_degree_bits == fri_params.degree_bits` |
| #64703 | Merkle leaf binding bypass | Valid | Length-bind `hash_leaf` in the sponge capacity |
| #64704 | Reversed ranges forge out-of-bounds views | Valid | Validate `start <= end`, checked arithmetic |
| #64705 | Offset overflow bypasses bounds validation | Valid | `checked_add` for `offset + P::WIDTH` |

## Findings

### #64696 — Malformed oracle metadata panics verifier (valid, reshaped)

The original vector (`FriOracleLayout.logical_polys` flowing through deserialization) was removed with
PolyFRI. The remaining live surface is `FriInstanceInfo`: `eval_opening_expression` / `unsalted_eval`
index oracle leaves by `polynomial_index` taken from deserialized metadata, with no referential check.

**Fix** (`core/src/fri_validate_shape.rs`): `validate_batch_fri_proof_shape` now verifies every opening
term references an in-range `oracle_index` and a `polynomial_index` within that oracle's leaf. In batch
FRI a single oracle's leaf concatenates polynomials from every instance, so the bound is the **sum** of
`num_polys` across instances (which reduces to `num_polys` for a single-instance plain-FRI call). This
runs on the live verifier path (`verify_fri_proof` → `validate_fri_proof_shape`).

### #64699 — Malformed split layout panics verifier (not valid)

`FriParams::batch_mask_layout`, `FriFinalPolyLayout::Split`, and the batch-masking path it guarded no
longer exist (removed with PolyFRI in `#50`). A repo-wide search for `FriFinalPolyLayout`,
`batch_mask_layout`, and related symbols returns no matches. No fix required.

### #64700 — Mismatched degree bits weaken FRI sampling (valid)

`get_challenges` samples FRI query indices from `common_data.public_initial_degree_bits()`
(`verifier/src/plonk/get_challenges.rs`), which is deserialized independently from the
transcript-bound `fri_params.degree_bits`. A malicious blob can shrink the sampled domain.

**Fix** (`core/src/circuit_config.rs`): `check_common_data_valid` now requires
`public_initial_degree_bits == fri_params.degree_bits`. Honest circuits satisfy this by construction
(`circuit_builder.rs` sets `public_initial_degree_bits = fri_params.degree_bits`). The check is wired
to the live deserialization path in both `verifier` and `plonky2` serialization; `CommonCircuitData`
and `CommonVerifierData::check_valid` were refactored to delegate to the shared function (DRY).

### #64703 — Merkle leaf binding bypass (valid, PoseidonHash)

`hash_leaf` absorbed inputs in overwrite mode with a constant capacity domain separator and no length
binding, so `[a,b,c,d,e]` and `[a,b,c,d,e,0]` produced the same digest.

**Fix** (`core/src/hashing.rs` native, `plonky2/src/hash/hashing.rs` in-circuit): place `len + 1` in
the capacity region (index `RATE`) instead of a constant `1`. The non-zero value still domain-separates
leaves from internal `two_to_one` nodes (zero capacity), and encoding the length makes the digest
injective in length, so zero-suffixed leaves cannot collide. Length lives in the capacity (not extra
absorbed blocks), so the recursive circuit size is unchanged (no `2^12 → 2^13` regression).

### #64704 — Reversed ranges forge out-of-bounds views (valid)

`PackedStridedView::view` computed `length` as `end - start` without checking `start <= end`; underflow
in release builds forged a huge length over a raw-pointer-backed view.

**Fix** (`core/src/strided_view.rs`): `Range`/`RangeInclusive` views assert `start <= end`, compute
`length` with `checked_sub`/`checked_add`, and derive the pointer offset via a `checked_mul` helper
(`checked_stride_offset`).

### #64705 — Offset overflow bypasses bounds validation (valid)

`PackedStridedView::new` checked `offset + P::WIDTH <= stride` in wrapping `usize` arithmetic, so a large
`offset` could wrap and pass.

**Fix** (`core/src/strided_view.rs`): compute `offset + P::WIDTH` with `checked_add` before the bound
comparison, so overflow panics instead of bypassing the check.

## Tests

Added regression tests:
- `core/src/merkle_tree.rs::test_zero_suffix_leaf_collision_rejected` (#64703)
- `core/src/strided_view.rs` — reversed `Range`/`RangeInclusive` and offset-overflow panics + a positive
  sub-view (#64704, #64705)
- `core/src/circuit_config.rs::check_common_data_rejects_mismatched_degree_bits` (#64700)

Suites run green: `qp-plonky2-core`, `qp-plonky2-verifier`, `starky` (batch-FRI + recursive STARK
end-to-end), and the full `qp-plonky2` lib suite (124 passed, incl. `test_recursive_recursive_verifier`
and `test_cyclic_recursion`).
