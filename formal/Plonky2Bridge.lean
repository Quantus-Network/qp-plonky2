/-
  The composition bridge (PLAN.md §6): instantiate the spec's random oracle with the
  concrete finite-field Poseidon2 sponge, and pin down the `.val` carrier seam.

  THE PAYOFF OF THE COLLISION-RESISTANCE REFACTOR.
  ------------------------------------------------
  Until `qp-zk-circuits/formal` dropped its baked-in `injective` field (the "option 2"
  refactor), `WormholeSpec.RandomOracle` was *uninhabited* over a finite field — an
  injective `H : List Felt → Digest` cannot exist when `Digest` is finite (pigeonhole),
  so this instance could not be written and every RO theorem was vacuous. Now the oracle
  is just `H`, and `Plonky2Spec.Sponge`'s real `hash_no_pad` is a perfectly good `H`.
  `spongeRO` below is that instance: the *literal* `WormholeSpec.RandomOracle`, realized
  by the Step-3b/3c verified sponge. This is the first object to actually join the two
  packages.

  THE `.val` SEAM (the interface between `Felt = ℕ` and `ZMod p`).
  ---------------------------------------------------------------
  The spec hashes over `Felt = ℕ`; the sponge computes over `ZMod p`. `spongeH` bridges:

      inputs:   `List ℕ`  ──(↑· : ℕ → ZMod p)──▶  `List (ZMod p)`   (NON-injective)
      outputs:  squeezed `ZMod p` lanes  ──ZMod.val──▶  `ℕ`          (injective)

  The output side is clean (`ZMod.val` is injective). The input side is *not*: the cast
  `ℕ → ZMod p` identifies `n` with `n + p`. The honest consequence — made explicit here —
  is that the realized oracle is **not** globally collision-resistant
  (`spongeRO_not_collisionResistant`: `[0]` and `[p]` collide). Collision resistance is
  meaningful only on **canonical** (`< p`) preimages, where the cast is injective
  (`castList_inj_on_canonical`) and a realized-oracle collision is exactly a field-level
  Poseidon2 collision (`spongeH_canonical_collision_is_field_collision`, and the converse
  `field_collision_gives_spongeH_collision`). That is the precise content the spec's
  abstract `CollisionResistant` hypothesis carries once discharged against the real hash,
  and it matches `WormholeSpec.Encoding`'s `{0, p}` non-canonical collision finding.

  This module imports BOTH packages and is deliberately kept out of `defaultTargets`, so
  the hermetic `Plonky2Spec` build (and its sibling-free CI job) is unaffected.
-/
import Mathlib.Data.ZMod.Basic
import WormholeSpec
import Plonky2Spec.Sponge

namespace Plonky2Bridge

open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash)
open WormholeSpec (Digest RandomOracle)

variable {p : ℕ}

/-! ### The realized oracle -/

/-- Embed a felt preimage into the native field (the input side of the seam). -/
def emb (l : List ℕ) : List (ZMod p) := l.map (fun a => (a : ZMod p))

@[simp] theorem emb_nil : emb (p := p) [] = [] := rfl

@[simp] theorem emb_cons (a : ℕ) (l : List ℕ) :
    emb (p := p) (a :: l) = (a : ZMod p) :: emb l := rfl

/-- Realize the spec's `H : List Felt → Digest` with the field sponge: cast the felt
    preimage into `ZMod p`, run `spongeHash`, read the 4 lanes back via `ZMod.val`. -/
def spongeH (perm : St p → St p) (l : List ℕ) : Digest :=
  let out := spongeHash perm (emb l)
  ⟨(out 0).val, (out 1).val, (out 2).val, (out 3).val⟩

/-- **The payoff.** The spec's random oracle, instantiated by the concrete finite-field
    Poseidon2 sponge — impossible under the old injective-`RandomOracle` (uninhabited over
    a finite field). Generic in the permutation; instantiate `perm := permState …` at the
    Goldilocks round constants for the real `Poseidon2Hash::hash_no_pad`. -/
def spongeRO (perm : St p → St p) : RandomOracle := { H := spongeH perm }

@[simp] theorem spongeRO_H (perm : St p → St p) (l : List ℕ) :
    (spongeRO perm).H l = spongeH perm l := rfl

/-- The realized oracle's dummy nullifier is *definitionally* the double sponge `H(H u)`,
    matching `Plonky2Spec.Sponge.dummyNull` and the Rust `hash_dummy_nullifier_pre_image`. -/
theorem spongeRO_dummyNull (perm : St p → St p) (u : List ℕ) :
    (spongeRO perm).dummyNull u = spongeH perm ((spongeH perm u).toList) := rfl

/-! ### The seam: the input cast is non-injective, so global CR fails -/

/-- The realized oracle is **not** globally collision-resistant: `[0]` and `[p]` are a
    collision, because `(↑p : ZMod p) = 0`. So `CollisionResistant` can only be asserted
    on canonical preimages (next section). This is the spec-side `{0, p}` collision made
    concrete on the real hash. -/
theorem spongeRO_not_collisionResistant [NeZero p] (perm : St p → St p) :
    ¬ (spongeRO perm).CollisionResistant := by
  intro cr
  have hcast : emb (p := p) [0] = emb (p := p) [p] := by
    simp [emb_cons, emb_nil]
  have hH : (spongeRO perm).H [0] = (spongeRO perm).H [p] := by
    show spongeH perm [0] = spongeH perm [p]
    unfold spongeH
    rw [hcast]
  have heq : ([0] : List ℕ) = [p] := cr [0] [p] hH
  rw [List.cons.injEq] at heq
  exact NeZero.ne p heq.1.symm

/-! ### The seam: on canonical inputs the cast is injective ⇒ field collisions -/

/-- Canonical felts (`< p`): exactly where the `ℕ → ZMod p` cast is injective. -/
def Canonical (v : ℕ) : Prop := v < p

/-- The cast is injective on canonical felts (`< p`). -/
theorem cast_inj_on_canonical {a b : ℕ} (ha : a < p) (hb : b < p)
    (h : (a : ZMod p) = (b : ZMod p)) : a = b := by
  have hv := congrArg ZMod.val h
  rwa [ZMod.val_natCast, ZMod.val_natCast, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at hv

/-- Lifting `cast_inj_on_canonical` to lists: canonical preimages with equal field
    embeddings are equal. -/
theorem castList_inj_on_canonical :
    ∀ {x y : List ℕ}, (∀ v ∈ x, v < p) → (∀ v ∈ y, v < p) →
      emb (p := p) x = emb (p := p) y → x = y
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, _, h => by simp [emb_nil, emb_cons] at h
  | _ :: _, [], _, _, h => by simp [emb_nil, emb_cons] at h
  | a :: x, b :: y, hx, hy, h => by
      rw [emb_cons, emb_cons, List.cons.injEq] at h
      have ha := hx a (by simp)
      have hb := hy b (by simp)
      have hrec := castList_inj_on_canonical (x := x) (y := y)
        (fun v hv => hx v (List.mem_cons_of_mem a hv))
        (fun v hv => hy v (List.mem_cons_of_mem b hv)) h.2
      rw [cast_inj_on_canonical ha hb h.1, hrec]

/-- **The seam bridge.** Restricted to canonical preimages, a collision of the realized
    oracle is exactly a Poseidon2 collision over the field: the field embeddings are
    distinct (the cast is injective on canonical inputs) yet the sponge maps them to the
    same state. So `spongeRO`-collision-resistance *on canonical felts* is precisely
    field-level Poseidon2 collision resistance. -/
theorem spongeH_canonical_collision_is_field_collision [NeZero p] (perm : St p → St p)
    {x y : List ℕ} (hx : ∀ v ∈ x, v < p) (hy : ∀ v ∈ y, v < p)
    (hne : x ≠ y) (hcol : spongeH perm x = spongeH perm y) :
    emb (p := p) x ≠ emb (p := p) y
      ∧ spongeHash perm (emb x) = spongeHash perm (emb y) := by
  refine ⟨fun hembeq => hne (castList_inj_on_canonical hx hy hembeq), ?_⟩
  have hd := hcol
  simp only [spongeH, Digest.mk.injEq] at hd
  funext i
  fin_cases i
  · exact ZMod.val_injective p hd.1
  · exact ZMod.val_injective p hd.2.1
  · exact ZMod.val_injective p hd.2.2.1
  · exact ZMod.val_injective p hd.2.2.2

/-- The converse direction of the seam: a field-level sponge collision yields a
    realized-oracle collision (`ZMod.val` carries it back). With the forward bridge above,
    `spongeH` collisions on canonical inputs and field sponge collisions are the same
    thing. -/
theorem field_collision_gives_spongeH_collision (perm : St p → St p) {x y : List ℕ}
    (h : spongeHash perm (emb x) = spongeHash perm (emb y)) :
    spongeH perm x = spongeH perm y := by
  simp only [spongeH, h]

end Plonky2Bridge
