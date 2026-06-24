/-
  Step 3c — the Poseidon2 `hash_no_pad` sponge, and the bridge to the spec's
  random-oracle hash `RandomOracle.H`.

  Step 3b gave the *permutation* its meaning (`Plonky2Spec.Poseidon2.permState`,
  via `gate_sound_complete`). The wormhole circuits never call the permutation
  directly — they call the **sponge** `Poseidon2Hash::hash_no_pad`. This module
  lifts the permutation to that sponge, transcribing
  `core/src/hashing.rs::hash_n_to_hash_no_pad_p2` line by line:

      msg   := pad10_to_rate(inputs, RATE)      -- append `1`, zero-fill to a rate multiple
      state := 0¹²                              -- all-zero width-12 state
      for block in msg.chunks(RATE):            -- RATE = 8
          state[i] += block[i]   (i < RATE)     -- additive absorption on the rate lanes
          state    := permute(state)            -- the Step-3b permutation
      digest := state[0..4]                     -- squeeze, NO trailing permute

  WIDTH = 12, RATE = 8, CAPACITY = 4, digest = 4 felts (NUM_HASH_OUT_ELTS).

  THE `RandomOracle` INSTANCE NOW LIVES IN `Plonky2Bridge`.
  --------------------------------------------------------
  This module mirrors `WormholeSpec`'s felt/digest interface locally rather than
  importing it, so the spec package stays a mathlib-free leaf that builds in seconds.
  The *instance* that actually joins the two packages — the literal
  `WormholeSpec.RandomOracle` realized by this sponge — is built in the cross-package
  bridge `Plonky2Bridge` (it imports both, on the mathlib side). It was the spec's
  "option 2" refactor (collision resistance as an explicit hypothesis instead of a
  baked-in `injective` field) that made that instance possible at all: over the finite
  field an injective `H` cannot exist (pigeonhole), so the old `RandomOracle` was
  uninhabited. What this module provides is the honest computational object the abstract
  `ro.H` stands for — the real `hash_no_pad` — and the structural facts the spec relies
  on (`hh`/`dummyNull` are exactly the double-sponge `H (H ·)`). Collision resistance
  remains an explicit assumption; `Plonky2Bridge` pins down exactly what it means against
  the concrete sponge (it is a *canonical-input* property — see the `.val` seam below).

  The carrier here is `ZMod p` (the native field, as in `Poseidon2.lean`); the spec
  uses `Felt = ℕ`. The bridge casts inputs `ℕ → ZMod p` and reads digests back with
  `ZMod.val : ZMod p → ℕ`, mirroring the spec's `Digest` (whose `.val`s are the felt
  outputs). For canonical (`< p`) felts the round-trip is the identity; on non-canonical
  felts the input cast is non-injective — the seam `Plonky2Bridge` makes precise.

  FAITHFULNESS OF THE WRAPPER. The permutation underneath is exporter-backed and
  `ring`-bridged (Step 3b). The sponge *wrapper* (`pad10`/`addBlock`/`absorbMsg`/
  `squeeze4`) is pinned to the live Rust by a differential test,
  `sponge_structure_matches_lean_model` (constraint-exporter/src/lib.rs): a Rust
  transliteration of these definitions over the same permutation is checked equal to
  `Poseidon2Hash::hash_no_pad` at random inputs across the rate boundaries. The Rust
  sponge core (`core/src/hashing.rs::hash_n_to_hash_no_pad_p2`) is itself factored into
  helpers with these exact names (`add_block`/`absorb_msg`/`squeeze4`, `pad10_to_rate`)
  so the correspondence is auditable line by line.
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.List.Basic
import Plonky2Spec.Poseidon2

namespace Plonky2Spec.Sponge

open Plonky2Spec.Poseidon2 (St permState)

variable {p : ℕ}

/-! ### Sponge parameters (mirroring `qp_poseidon_core`) -/

/-- Sponge rate: `SPONGE_RATE = 8`. The first 8 lanes are absorbed into; the
    remaining 4 (`SPONGE_CAPACITY`) are never touched by absorption. An `abbrev` so
    `omega` reduces it to the literal `8`. -/
abbrev rate : ℕ := 8

/-! ### The sponge construction (generic in the permutation) -/

/-- The `10*` padding rule (`pad10_to_rate`): append a single `1`, then zero-fill to
    the next multiple of `rate`. When `inputs.length` is already a multiple of `rate`
    (including the empty input) this appends a whole `[1, 0, …, 0]` block. -/
def pad10 (inputs : List (ZMod p)) : List (ZMod p) :=
  let n := inputs.length
  let padded := ((n + 1 + (rate - 1)) / rate) * rate
  inputs ++ (1 : ZMod p) :: List.replicate (padded - (n + 1)) 0

/-- Additive absorption of one block onto the rate lanes: `state[i] += block[i]` for
    `i < block.length` (`≤ rate`), capacity lanes untouched. -/
def addBlock (s : St p) (block : List (ZMod p)) : St p :=
  fun i => if h : i.val < block.length then s i + block[i.val]'h else s i

/-- Absorb a (padded) message rate-block by rate-block, permuting after each block.
    Faithful to the Rust `for block in msg.chunks(RATE) { absorb; permute }` loop. -/
def absorbMsg (perm : St p → St p) (s : St p) : List (ZMod p) → St p
  | [] => s
  | (hd :: tl) =>
      absorbMsg perm (perm (addBlock s ((hd :: tl).take rate))) ((hd :: tl).drop rate)
  termination_by msg => msg.length
  decreasing_by
    simp only [List.length_drop, List.length_cons, rate]
    omega

/-- Squeeze the 4-felt digest: the first `NUM_HASH_OUT_ELTS = 4` lanes (no trailing
    permute), matching `squeeze()[..NUM_HASH_OUT_ELTS]`. -/
def squeeze4 (s : St p) : Fin 4 → ZMod p :=
  fun i => s ⟨i.val, by have := i.isLt; omega⟩

/-- `hash_n_to_hash_no_pad_p2` over the native field, generic in the permutation:
    pad, absorb from the all-zero state, squeeze 4 lanes. -/
def spongeHash (perm : St p → St p) (inputs : List (ZMod p)) : Fin 4 → ZMod p :=
  squeeze4 (absorbMsg perm (fun _ => 0) (pad10 inputs))

/-- The Poseidon2 sponge: the Step-3b verified `permState` driving the sponge. Generic
    in the round constants — instantiate `eInit/eTerm/iRc/diag` at the real Goldilocks
    constants to obtain the concrete `Poseidon2Hash::hash_no_pad`. -/
def poseidon2Hash (eInit eTerm : Fin 4 → St p) (iRc : Fin 22 → ZMod p) (diag : St p)
    (inputs : List (ZMod p)) : Fin 4 → ZMod p :=
  spongeHash (permState eInit eTerm iRc diag) inputs

/-! ### A single permutation suffices for short inputs

The aggregator's dummy-nullifier preimage is 4 felts, and a digest re-expanded for
the outer hash is 4 felts; both pad to exactly one `rate`-block, so the sponge is a
single permutation call. -/

/-- For a non-empty message of at most `rate` elements, the sponge absorbs exactly one
    block: `absorbMsg` collapses to a single `perm (addBlock s msg)`. -/
theorem absorbMsg_short (perm : St p → St p) (s : St p) (msg : List (ZMod p))
    (h0 : 0 < msg.length) (hle : msg.length ≤ rate) :
    absorbMsg perm s msg = perm (addBlock s msg) := by
  obtain _ | ⟨hd, tl⟩ := msg
  · simp at h0
  · have ht : (hd :: tl).take rate = hd :: tl := List.take_of_length_le hle
    have hdp : (hd :: tl).drop rate = [] := List.drop_eq_nil_of_le hle
    simp only [absorbMsg, ht, hdp]

/-- The padded message length: short inputs (`len + 1 ≤ rate`) pad to exactly one
    `rate`-block. -/
theorem pad10_length_short (inputs : List (ZMod p)) (hle : inputs.length + 1 ≤ rate) :
    (pad10 inputs).length = rate := by
  simp only [rate] at hle ⊢
  simp only [pad10, rate, List.length_append, List.length_cons, List.length_replicate]
  omega

/-- A short input (`len + 1 ≤ rate`, so the `1`-pad still fits one block) hashes with a
    single permutation from the zero state. Covers the 4-felt dummy-nullifier preimage
    and the 4-felt digest re-hash inside `hh`. -/
theorem spongeHash_short (perm : St p → St p) (inputs : List (ZMod p))
    (hle : inputs.length + 1 ≤ rate) :
    spongeHash perm inputs = squeeze4 (perm (addBlock (fun _ => 0) (pad10 inputs))) := by
  have hlen := pad10_length_short inputs hle
  have h0 : 0 < (pad10 inputs).length := by rw [hlen]; decide
  have hle2 : (pad10 inputs).length ≤ rate := le_of_eq hlen
  unfold spongeHash
  rw [absorbMsg_short perm _ _ h0 hle2]

/-! ### Bridge to the spec's `RandomOracle.H : List Felt → Digest`

We mirror `WormholeSpec`'s felt/digest interface here (rather than importing it: that
package is deliberately mathlib-free, and keeping it a leaf preserves its
seconds-long hermetic build). The objects below are the *computational realization*
the abstract `ro.H` stands for: inputs cast `ℕ → ZMod p`, the digest read back through
`ZMod.val`. The point of the section is the structural identity `dummyNull = H (H ·)` —
the exact composition the spec's `hh`/`dummyNull` use — discharged by `rfl`. The literal
`WormholeSpec.RandomOracle` instance over these is built in `Plonky2Bridge`. -/

/-- The spec's felt carrier (`WormholeSpec.Felt`). -/
abbrev Felt : Type := ℕ

/-- The 4-felt digest, mirroring `WormholeSpec.Digest`. -/
structure Digest where
  x0 : Felt
  x1 : Felt
  x2 : Felt
  x3 : Felt
  deriving DecidableEq, Repr

/-- Re-expand a digest to its felt list (matching `Digest.toList` / the Rust
    `inner.elements` flattening used by `hash_no_pad(&inner.elements)`). -/
def Digest.toList (d : Digest) : List Felt := [d.x0, d.x1, d.x2, d.x3]

/-- Read the 4 squeezed lanes back as canonical felts via `ZMod.val`. -/
def toDigest (out : Fin 4 → ZMod p) : Digest :=
  ⟨(out 0).val, (out 1).val, (out 2).val, (out 3).val⟩

/-- The computational random-oracle hash `H : List Felt → Digest`: cast the felt
    preimage into the native field, run the Poseidon2 sponge, read the digest back.
    Generic in the permutation; instantiate `perm := permState …` at the Goldilocks
    constants for the concrete `Poseidon2Hash::hash_no_pad`. -/
def H (perm : St p → St p) (input : List Felt) : Digest :=
  toDigest (spongeHash perm (input.map (fun a => (a : ZMod p))))

/-- The double hash `H(H(·))` (spec `RandomOracle.hh`): the inner digest is re-expanded
    to its 4 felts before the outer call, exactly `hash_no_pad(&inner.elements)`. -/
def hh (perm : St p → St p) (preimage : List Felt) : Digest :=
  H perm ((H perm preimage).toList)

/-- Dummy-nullifier replacement `DNull(u) = H(H(u))` (spec `RandomOracle.dummyNull`,
    Rust `hash_dummy_nullifier_pre_image`). -/
def dummyNull (perm : St p → St p) (u : List Felt) : Digest :=
  hh perm u

/-- `dummyNull` is *definitionally* the double sponge `H (H u)` — the precise shape the
    spec's `dummyNull`/`hh` are written in. This is the Step-3c structural bridge. -/
theorem dummyNull_eq (perm : St p → St p) (u : List Felt) :
    dummyNull perm u = H perm ((H perm u).toList) := rfl

/-- The double hash unfolds to two sponge passes (each over a `≤ rate` preimage is a
    single permutation, by `spongeHash_short`): the inner over the 4-felt `u`, the outer
    over the inner digest's 4 felts. Makes the `DNull(u) = H(H(u))` evaluation explicit
    for the aggregator's 4-felt preimages. -/
theorem dummyNull_unfold (perm : St p → St p) (u : List Felt) :
    dummyNull perm u
      = toDigest (spongeHash perm (((H perm u).toList).map (fun a => (a : ZMod p)))) := rfl

end Plonky2Spec.Sponge
