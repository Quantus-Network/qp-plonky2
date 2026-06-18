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
import Plonky2Spec.Wrapper

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

/-! ## The wrapper-logic `.val` seam (`Plonky2Spec.Wrapper` ⟶ `WormholeSpec.RL0` over `.val`)

  Step 2c left the cross-package composition open: `Plonky2Spec.Wrapper` proves each layer-0
  wrapper *gadget* (`select` / `and` / `or` / the first-real prefix scan) computes the right
  conditional over `ZMod p`, while `WormholeSpec.AggregationBridge` proves
  `Layer0Circuit ⟹ RL0` over `Felt = ℕ` — but nothing tied the two across the
  `ZMod.val ↔ Felt` boundary. This section is that bridge: a field gadget output, read back
  through `valDigest` / `ZMod.val`, equals the spec building block (`buildNullifiers`,
  `matchSum`, `referenceFromFirstReal`), with the dummy-nullifier digest realized by the
  verified sponge `spongeRO`. Composing the nullifier path end to end yields a genuine
  `RL0 (spongeRO perm) …`.

  What stays abstract is only the public-input *decode* — which field wire carries which
  spec value (`hd`/`hdnull`/`hreal`/`hnull`/`hexits`/… below). That is the wiring/copy-
  constraint model (PLAN §9 gap (a)), the separate next step; here it is the explicit
  hypothesis boundary. The *gadget logic* behind each such hypothesis is the cited
  `Plonky2Spec.Wrapper` lemma, lifted across `.val` by `valDigest_bselect` /
  `match_contribution_val` / `scan_val`. -/

open Plonky2Spec (IsBool bselect nullifier_replacement match_contribution scanStep
  firstRealVal scanFirst_correct)
open WormholeSpec (LeafPublic Layer0Output isDummyL0 buildNullifiers RL0 Layer0Circuit
  layer0_bridge groupExits childPairs metadataConsistent referenceFromFirstReal)
open WormholeSpec (outputExitTotal rawOutputTotal RL0_value_conservation LeafWitness Rleaf
  LeafProofAccepted leaf_proof_sound)

variable [Fact p.Prime]

/-- Read four field lanes back as a spec `Digest` via `ZMod.val` (the output side of the
    seam — `ZMod.val` is injective, so this direction loses nothing). -/
def valDigest (f : Fin 4 → ZMod p) : Digest :=
  ⟨(f 0).val, (f 1).val, (f 2).val, (f 3).val⟩

/-- A boolean digest-select read through `.val`: the lanewise lift of
    `Wrapper.nullifier_replacement`. Picks the `dnull` digest when the flag is set, else
    `real`. -/
theorem valDigest_bselect {is_dummy : ZMod p} {dnull real : Fin 4 → ZMod p}
    (hb : IsBool is_dummy) :
    valDigest (fun i => bselect is_dummy (dnull i) (real i))
      = if is_dummy = 1 then valDigest dnull else valDigest real := by
  by_cases h1 : is_dummy = 1
  · have hfun : (fun i => bselect is_dummy (dnull i) (real i)) = dnull := by
      funext i; rw [nullifier_replacement hb]; simp [h1]
    rw [hfun, if_pos h1]
  · have hfun : (fun i => bselect is_dummy (dnull i) (real i)) = real := by
      funext i; rw [nullifier_replacement hb]; simp [h1]
    rw [hfun, if_neg h1]

/-- **The per-slot nullifier bridge.** A field nullifier slot — `select(is_dummy, dnull, real)`
    lanewise — read through `.val` equals the spec's per-slot body of `buildNullifiers`, with
    the dummy digest realized by the verified sponge oracle `spongeRO`. The hypotheses are the
    decode/gadget facts: `is_dummy` is boolean and decodes the L0 dummy flag (`hd`), the
    `dnull` lanes are the sponge double-hash of the witnessed preimage (`hdnull`, the T3
    "gadget computes `H(H u)`" result — discharged by `spongeRO_dummyNull`), and the `real`
    lanes are the child's nullifier. -/
theorem nullifier_slot_bridge (perm : St p → St p) {is_dummy : ZMod p}
    {dnull real : Fin 4 → ZMod p} {pub : LeafPublic} {u : List ℕ}
    (hb : IsBool is_dummy)
    (hd : is_dummy = 1 ↔ isDummyL0 pub)
    (hdnull : valDigest dnull = (spongeRO perm).dummyNull u)
    (hreal : valDigest real = pub.nullifier) :
    valDigest (fun i => bselect is_dummy (dnull i) (real i))
      = if isDummyL0 pub then (spongeRO perm).dummyNull u else pub.nullifier := by
  rw [valDigest_bselect hb]
  by_cases h1 : is_dummy = 1
  · rw [if_pos h1, if_pos (hd.mp h1), hdnull]
  · rw [if_neg h1, if_neg (fun hh => h1 (hd.mpr hh)), hreal]

/-- Field witness for one nullifier slot, as the circuit lays it out: the boolean dummy
    flag, the dummy-nullifier lanes, and the real-nullifier lanes. -/
structure NullSlot (p : ℕ) where
  isDummy : ZMod p
  dnull : Fin 4 → ZMod p
  real : Fin 4 → ZMod p

/-- The field nullifier output of one slot, read back as a spec `Digest`. -/
def NullSlot.out (s : NullSlot p) : Digest :=
  valDigest (fun i => bselect s.isDummy (s.dnull i) (s.real i))

omit [Fact p.Prime] in
/-- `buildNullifiers` over leaves/preimages drawn from the same aligned witness list reduces
    to a pointwise `map` of its per-slot body. -/
theorem buildNullifiers_map (ro : RandomOracle)
    (triples : List (NullSlot p × LeafPublic × List WormholeSpec.Felt)) :
    buildNullifiers ro (triples.map (fun t => t.2.1)) (triples.map (fun t => t.2.2))
      = triples.map (fun t => if isDummyL0 t.2.1 then ro.dummyNull t.2.2 else t.2.1.nullifier) := by
  induction triples with
  | nil => rfl
  | cons t ts ih =>
      obtain ⟨s, pub, u⟩ := t
      simp only [List.map_cons, buildNullifiers]
      rw [ih]

/-- **List-level nullifier bridge.** Read through `.val`, the field nullifier outputs of an
    aligned witness list are exactly `buildNullifiers (spongeRO perm) leaves us` — i.e. the
    `Layer0Circuit.nulls` content, justified slotwise by `nullifier_slot_bridge`. -/
theorem nullifiers_val_bridge (perm : St p → St p)
    (triples : List (NullSlot p × LeafPublic × List WormholeSpec.Felt))
    (hb : ∀ t ∈ triples, IsBool t.1.isDummy)
    (hd : ∀ t ∈ triples, t.1.isDummy = 1 ↔ isDummyL0 t.2.1)
    (hdnull : ∀ t ∈ triples, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ triples, valDigest t.1.real = t.2.1.nullifier) :
    triples.map (fun t => t.1.out)
      = buildNullifiers (spongeRO perm) (triples.map (fun t => t.2.1))
          (triples.map (fun t => t.2.2)) := by
  rw [buildNullifiers_map]
  apply List.map_congr_left
  intro t ht
  simp only [NullSlot.out]
  exact nullifier_slot_bridge perm (hb t ht) (hd t ht) (hdnull t ht) (hreal t ht)

/-- **Wrapper-logic `.val` composition (layer 0).** Given the field witnesses for the
    nullifier slots (bridged slotwise above, the nullifier + Poseidon path closed end to end)
    plus the remaining wrapper outputs as their `.val`-decoded spec values, the layer-0
    circuit satisfies `RL0` for the verified-sponge oracle. The exit-grouping / metadata /
    first-real-reference equalities are the explicit decode hypotheses; their *gadget logic*
    is `Wrapper.{match_contribution, dedup_select, scanFirst_correct, real_block_matches}`
    (lifted by `match_contribution_val` / `scan_val`), and wiring them out of the field
    witness is the decode model (PLAN §9 gap (a)). -/
theorem layer0_val_RL0 (perm : St p → St p)
    (triples : List (NullSlot p × LeafPublic × List WormholeSpec.Felt)) {out : Layer0Output}
    (hb : ∀ t ∈ triples, IsBool t.1.isDummy)
    (hd : ∀ t ∈ triples, t.1.isDummy = 1 ↔ isDummyL0 t.2.1)
    (hdnull : ∀ t ∈ triples, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ triples, valDigest t.1.real = t.2.1.nullifier)
    (hnull : out.nullifiers = triples.map (fun t => t.1.out))
    (hexits : out.exitSlots = groupExits (childPairs (triples.map (fun t => t.2.1))))
    (hmeta : metadataConsistent (triples.map (fun t => t.2.1)) out)
    (href : referenceFromFirstReal (triples.map (fun t => t.2.1)) out) :
    RL0 (spongeRO perm) (triples.map (fun t => t.2.1)) (triples.map (fun t => t.2.2)) out :=
  layer0_bridge
    { uslen := by simp [List.length_map]
      nulls := by rw [hnull]; exact nullifiers_val_bridge perm triples hb hd hdnull hreal
      exits := hexits
      metaOk := hmeta
      ref := href }

/-! ### Supporting `.val` lifts for the exit-grouping and reference primitives -/

/-- The exit-match contribution read through `.val`: `Wrapper.match_contribution` over the
    seam. With the equality flag correct, the field `select(eq, amount, 0)` decodes to the
    per-element body of `matchSum`, `if k = key then amount.val else 0`. -/
theorem match_contribution_val {eq amount : ZMod p} {P : Prop} [Decidable P]
    (hb : IsBool eq) (hP : eq = 1 ↔ P) :
    (bselect eq amount 0).val = if P then amount.val else 0 := by
  rw [match_contribution hb hP]
  by_cases h : P <;> simp [h]

/-- The first-real prefix scan read through `.val`: `Wrapper.scanFirst_correct` over the seam.
    The scanned reference value equals the first real slot's value (the basis of
    `referenceFromFirstReal`, applied per block-hash limb and the block number). -/
theorem scan_val (xs : List (ZMod p × ZMod p)) (init : ZMod p)
    (hb : ∀ rv ∈ xs, IsBool rv.1) :
    ((List.foldl scanStep (0, init) xs).2).val = (firstRealVal init xs).val :=
  congrArg ZMod.val (scanFirst_correct xs init hb)

/-! ## End-to-end layer-0 soundness over the verified sponge (the trust stack, assembled)

  The capstone: for the *concrete* random oracle `spongeRO perm` (the Step-3b/3c verified
  Poseidon2 sponge), a satisfied layer-0 aggregation circuit whose recursion gadget accepted
  every child leaf proof

    (i)   satisfies the layer-0 relation `RL0`                  — rung (2)→(3), via the wrapper
                                                                  `.val` bridge `layer0_val_RL0`,
                                                                  built on the exporter-verified
                                                                  gadget semantics;
    (ii)  conserves value (`outputExitTotal = rawOutputTotal`)  — rung (4), `RL0_value_conservation`;
    (iii) attests every child's leaf relation `Rleaf`           — the trusted recursion seam (1),
                                                                  `leaf_proof_sound`.

  This is the single statement the PLAN's trust stack (§1) builds toward, instantiated at the
  real hash. Two things sit *outside* the Lean hypotheses, by design:
  * **Fidelity to the Rust** — that `spongeRO`/the wrapper gadgets *are* the deployed circuit —
    is carried by the constraint exporter + differential tests (gate constraints ≡ real
    `eval_unfiltered`, sponge ≡ `hash_no_pad`), not re-proved here.
  * The public-input **decode** (`hd`/`hdnull`/`hreal`/`hnull`/`hexits`/`hmeta`/`href`) is the
    wiring/copy-constraint model (PLAN §9 gap (a)).
  The only trusted *axioms* this theorem depends on are the recursion/proof-system ones in
  `WormholeSpec.Trusted` (pulled in solely by clause (iii)); clauses (i)–(ii) are
  standard-axioms-only. -/
theorem layer0_end_to_end (perm : St p → St p)
    (triples : List (NullSlot p × LeafPublic × List WormholeSpec.Felt)) {out : Layer0Output}
    (hb : ∀ t ∈ triples, IsBool t.1.isDummy)
    (hd : ∀ t ∈ triples, t.1.isDummy = 1 ↔ isDummyL0 t.2.1)
    (hdnull : ∀ t ∈ triples, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ triples, valDigest t.1.real = t.2.1.nullifier)
    (hnull : out.nullifiers = triples.map (fun t => t.1.out))
    (hexits : out.exitSlots = groupExits (childPairs (triples.map (fun t => t.2.1))))
    (hmeta : metadataConsistent (triples.map (fun t => t.2.1)) out)
    (href : referenceFromFirstReal (triples.map (fun t => t.2.1)) out)
    (hacc : ∀ pub ∈ triples.map (fun t => t.2.1), LeafProofAccepted (spongeRO perm) pub) :
    RL0 (spongeRO perm) (triples.map (fun t => t.2.1)) (triples.map (fun t => t.2.2)) out
      ∧ outputExitTotal out = rawOutputTotal (triples.map (fun t => t.2.1))
      ∧ ∀ pub ∈ triples.map (fun t => t.2.1), ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w := by
  have hRL0 := layer0_val_RL0 perm triples hb hd hdnull hreal hnull hexits hmeta href
  exact ⟨hRL0, RL0_value_conservation hRL0,
    fun pub hp => leaf_proof_sound (spongeRO perm) pub (hacc pub hp)⟩

end Plonky2Bridge
