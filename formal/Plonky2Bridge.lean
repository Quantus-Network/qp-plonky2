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
  the hermetic `Plonky2Spec` build (and its `wormholeSpec`-free CI job) is unaffected.

  TRUST BOUNDARY WITH `qp-zk-circuits/formal`.
  ------------------------------------------------
  The capstone (`private_batch_end_to_end`) composes *this* package's sponge/wrapper `.val`
  bridge with definitions and lemmas from the pinned `wormholeSpec` dependency
  (`RPrivateBatch`, `private_batch_bridge`, `RPrivateBatch_value_conservation`,
  `RPrivateBatch_settles_distinct_spends`, `leaf_proof_sound`, …). Those objects are stated
  and proved (or axiomatized, for `leaf_proof_sound`) in qp-zk-circuits/formal at the commit
  pinned in `formal/lakefile.toml` — this package does not restate or re-verify them. What
  *is* here: `spongeRO`, the `.val` seam, and the gadget-level bridges (nullifier selection
  + permutation network, uniqueness loop, fee comparator, public-batch forwarding masks)
  into `RPrivateBatch (spongeRO perm) …`.
-/
import Mathlib.Data.ZMod.Basic
import WormholeSpec
import Plonky2Spec.Sponge
import Plonky2Spec.Wrapper
import Plonky2Spec.Permutation
import Plonky2Spec.FeeCheck

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

/-! ## The wrapper-logic `.val` seam (`Plonky2Spec` gadgets ⟶ `WormholeSpec.RPrivateBatch`)

  `Plonky2Spec.Wrapper` / `Permutation` / `FeeCheck` prove each private-batch wrapper
  *gadget* (`select` / `and` / `or` / `is_equal` / the first-real prefix scan / the
  odd-even switch network / the two fee range checks) computes the right thing over
  `ZMod p`, while `WormholeSpec.AggregationBridge` proves `PrivateBatchCircuit ⟹
  RPrivateBatch` over `Felt = ℕ`. This section ties the two across the `ZMod.val ↔ Felt`
  boundary: a field gadget output, read back through `valDigest` / `ZMod.val`, equals the
  spec building block, with the dummy-nullifier digest realized by the verified sponge
  `spongeRO`. Composed, the nullifier path (selection + Poseidon + permutation), the
  uniqueness loop, and the fee comparator are discharged end to end, yielding a genuine
  `RPrivateBatch (spongeRO perm) …`.

  What stays abstract is only the public-input *decode* — which field wire carries which
  spec value (`hd`/`hdnull`/`hreal`/`hnull`/`hexits`/… below). That is the wiring/copy-
  constraint model (PLAN §9 gap (a)); here it is the explicit hypothesis boundary. -/

open Plonky2Spec (IsBool bselect band bnot nullifier_replacement match_contribution scanStep
  firstRealVal scanFirst_correct Digest4 network network_perm FeeCheck feeCheck_sound
  uniqueness_pair_dummy)
open WormholeSpec (LeafPublic PrivateBatchOutput isDummyPrivateBatch buildNullifiers
  RPrivateBatch PrivateBatchCircuit private_batch_bridge groupExits maskedChildPairs
  metadataConsistent referenceFromFirstReal realNullifiersDistinct privateBatchFeeOk
  maskedInputTotal maskedOutputTotal realLeaves realNullifiers rawOutputTotal
  outputExitTotal RPrivateBatch_value_conservation RPrivateBatch_settles_distinct_spends
  LeafWitness Rleaf LeafProofAccepted leaf_proof_sound ExitSlot isDummyInner
  forwardedSlots forwardedNullifiers)

variable [Fact p.Prime]

/-- Read four field lanes back as a spec `Digest` via `ZMod.val` (the output side of the
    seam — `ZMod.val` is injective, so this direction loses nothing). -/
def valDigest (f : Digest4 p) : Digest :=
  ⟨(f 0).val, (f 1).val, (f 2).val, (f 3).val⟩

theorem valDigest_injective : Function.Injective (valDigest (p := p)) := by
  intro f g h
  simp only [valDigest, Digest.mk.injEq] at h
  funext i
  fin_cases i
  · exact ZMod.val_injective p h.1
  · exact ZMod.val_injective p h.2.1
  · exact ZMod.val_injective p h.2.2.1
  · exact ZMod.val_injective p h.2.2.2

/-- A boolean digest-select read through `.val`: the lanewise lift of
    `Wrapper.nullifier_replacement`. Picks the `dnull` digest when the flag is set, else
    `real`. -/
theorem valDigest_bselect {is_dummy : ZMod p} {dnull real : Digest4 p}
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

/-- The zeroing mask `select(is_dummy, 0, x)` on a digest, read through `.val`: the
    ingress mask (`maskedChildPairs`) and the public-batch forwarding zero
    (`forwardedSlots` / `forwardedNullifiers`). -/
theorem valDigest_mask {is_dummy : ZMod p} {x : Digest4 p} (hb : IsBool is_dummy) :
    valDigest (fun i => bselect is_dummy 0 (x i))
      = if is_dummy = 1 then Digest.zero else valDigest x := by
  rw [valDigest_bselect (dnull := fun _ => 0) hb]
  by_cases h1 : is_dummy = 1
  · rw [if_pos h1, if_pos h1]; simp [valDigest, Digest.zero]
  · rw [if_neg h1, if_neg h1]

/-- The zeroing mask on a scalar, read through `.val`. -/
theorem val_mask {is_dummy x : ZMod p} (hb : IsBool is_dummy) :
    (bselect is_dummy 0 x).val = if is_dummy = 1 then 0 else x.val := by
  rw [Plonky2Spec.ingress_mask hb]
  by_cases h1 : is_dummy = 1 <;> simp [h1]

/-! ### Nullifier slots: selection, Poseidon, and the private permutation -/

/-- **The per-slot nullifier bridge.** A field nullifier slot — `select(is_dummy, dnull, real)`
    lanewise — read through `.val` equals the spec's per-slot body of `buildNullifiers`, with
    the dummy digest realized by the verified sponge oracle `spongeRO`. The hypotheses are the
    decode/gadget facts: `is_dummy` is boolean and decodes the private-batch dummy flag (`hd`),
    the `dnull` lanes are the sponge double-hash of the witnessed preimage (`hdnull`, the T3
    "gadget computes `H(H u)`" result — discharged by `spongeRO_dummyNull`), and the `real`
    lanes are the child's nullifier. -/
theorem nullifier_slot_bridge (perm : St p → St p) {is_dummy : ZMod p}
    {dnull real : Digest4 p} {pub : LeafPublic} {u : List ℕ}
    (hb : IsBool is_dummy)
    (hd : is_dummy = 1 ↔ isDummyPrivateBatch pub)
    (hdnull : valDigest dnull = (spongeRO perm).dummyNull u)
    (hreal : valDigest real = pub.nullifier) :
    valDigest (fun i => bselect is_dummy (dnull i) (real i))
      = if isDummyPrivateBatch pub then (spongeRO perm).dummyNull u else pub.nullifier := by
  rw [valDigest_bselect hb]
  by_cases h1 : is_dummy = 1
  · rw [if_pos h1, if_pos (hd.mp h1), hdnull]
  · rw [if_neg h1, if_neg (fun hh => h1 (hd.mpr hh)), hreal]

/-- Field witness for one nullifier slot, as the circuit lays it out: the boolean dummy
    flag, the dummy-nullifier lanes, and the real-nullifier lanes. -/
structure NullSlot (p : ℕ) where
  isDummy : ZMod p
  dnull : Digest4 p
  real : Digest4 p

/-- The selected nullifier of one slot (`select(is_dummy, dnull, real)` lanewise) — the
    field digest fed into the permutation network. -/
def NullSlot.sel (s : NullSlot p) : Digest4 p :=
  fun i => bselect s.isDummy (s.dnull i) (s.real i)

/-- The selected nullifier read back as a spec `Digest`. -/
def NullSlot.out (s : NullSlot p) : Digest := valDigest s.sel

/-- One aligned witness row: the field slot, the decoded child public inputs, and the
    dummy-nullifier preimage. -/
abbrev SlotRow (p : ℕ) := NullSlot p × LeafPublic × List WormholeSpec.Felt

omit [Fact p.Prime] in
/-- `buildNullifiers` over leaves/preimages drawn from the same aligned witness list reduces
    to a pointwise `map` of its per-slot body. -/
theorem buildNullifiers_map (ro : RandomOracle) (rows : List (SlotRow p)) :
    buildNullifiers ro (rows.map (fun t => t.2.1)) (rows.map (fun t => t.2.2))
      = rows.map (fun t =>
          if isDummyPrivateBatch t.2.1 then ro.dummyNull t.2.2 else t.2.1.nullifier) := by
  induction rows with
  | nil => rfl
  | cons t ts ih =>
      obtain ⟨s, pub, u⟩ := t
      simp only [List.map_cons, buildNullifiers]
      rw [ih]

/-- **List-level nullifier bridge.** Read through `.val`, the field selections of an aligned
    witness list are exactly `buildNullifiers (spongeRO perm) leaves us` (the
    pre-permutation list `raw` of `RPrivateBatch`), justified slotwise by
    `nullifier_slot_bridge`. -/
theorem nullifiers_val_bridge (perm : St p → St p) (rows : List (SlotRow p))
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1)
    (hdnull : ∀ t ∈ rows, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier) :
    (rows.map (fun t => t.1.sel)).map valDigest
      = buildNullifiers (spongeRO perm) (rows.map (fun t => t.2.1))
          (rows.map (fun t => t.2.2)) := by
  rw [buildNullifiers_map, List.map_map]
  apply List.map_congr_left
  intro t ht
  exact nullifier_slot_bridge perm (hb t ht) (hd t ht) (hdnull t ht) (hreal t ht)

/-- **Permutation bridge.** The public nullifier region — the switch network's output read
    through `.val` — is a permutation of `buildNullifiers`, for every boolean switch witness.
    This discharges `PrivateBatchCircuit.nullsPerm` from the gadget (`network_perm`) rather
    than assuming it. -/
theorem nullifiers_perm_bridge (perm : St p → St p) (rounds : List (List (ZMod p)))
    (rows : List (SlotRow p))
    (hsw : ∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s)
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1)
    (hdnull : ∀ t ∈ rows, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier) :
    ((network rounds (rows.map (fun t => t.1.sel))).map valDigest).Perm
      (buildNullifiers (spongeRO perm) (rows.map (fun t => t.2.1)) (rows.map (fun t => t.2.2))) := by
  rw [← nullifiers_val_bridge perm rows hb hd hdnull hreal]
  exact (network_perm rounds _ hsw).map valDigest

/-! ### Real-nullifier uniqueness across `.val` -/

/-- **Uniqueness bridge.** The pairwise `and(and(is_real_i, is_real_j), digest_eq) = 0`
    constraints over all `i < j` (with each `digest_eq` flag correct, `bytesDigestEq_spec`)
    give the spec's `realNullifiersDistinct` on the decoded children: two real slots cannot
    decode to the same nullifier, since `valDigest` is injective. -/
theorem uniqueness_val_bridge (rows : List (SlotRow p))
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1)
    (hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier)
    (hcol : ∀ (i j : ℕ) (hi : i < rows.length) (hj : j < rows.length), i < j →
      ∃ eq : ZMod p, (eq = 1 ↔ rows[i].1.real = rows[j].1.real) ∧
        band (band (bnot rows[i].1.isDummy) (bnot rows[j].1.isDummy)) eq = 0) :
    realNullifiersDistinct (rows.map (fun t => t.2.1)) := by
  unfold realNullifiersDistinct
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  have hi' : i < rows.length := by simpa using hi
  have hj' : j < rows.length := by simpa using hj
  simp only [List.getElem_map]
  intro hri hrj heqnull
  obtain ⟨eq, heq, hcon⟩ := hcol i j hi' hj' hij
  have hmi := List.getElem_mem hi'
  have hmj := List.getElem_mem hj'
  have hdi : rows[i].1.isDummy ≠ 1 := fun h1 => hri ((hd _ hmi).mp h1)
  have hdj : rows[j].1.isDummy ≠ 1 := fun h1 => hrj ((hd _ hmj).mp h1)
  apply uniqueness_pair_dummy (hb _ hmi) (hb _ hmj) heq hcon hdi hdj
  apply valDigest_injective
  rw [hreal _ hmi, hreal _ hmj]
  exact heqnull

/-! ### The fee comparator across `.val` -/

/-- **Fee bridge.** The two fee range checks (`FeeCheck`), with the accumulator wires decoding
    to the dummy-masked totals and the reference fee, enforce `privateBatchFeeOk`. The
    `FeeCheck.Assumptions` are discharged from the spec's own bounds: the leaf 32-bit range
    checks (through the first-real scan for the fee), the `≤ 64`-leaf cap, and the two-sided
    `2^52` bounds — which is where the Goldilocks modulus (`goldilocks ≤ p`) enters. -/
theorem fee_val_bridge (hpg : WormholeSpec.goldilocks ≤ p)
    {leaves : List LeafPublic} {out : PrivateBatchOutput} {fee totalIn totalOut : ZMod p}
    (hlen : leaves.length ≤ 64)
    (h32 : ∀ q ∈ leaves, WormholeSpec.inRange 32 q.inputAmount ∧
      WormholeSpec.inRange 32 q.outputAmount1 ∧ WormholeSpec.inRange 32 q.outputAmount2 ∧
      WormholeSpec.inRange 32 q.volumeFeeBps)
    (href : referenceFromFirstReal leaves out)
    (hfee : fee.val = out.volumeFeeBps)
    (hin : totalIn.val = maskedInputTotal leaves)
    (hout : totalOut.val = maskedOutputTotal leaves)
    (h : FeeCheck fee totalIn totalOut) : privateBatchFeeOk leaves out := by
  have hp : 2 ^ 53 ≤ p := le_trans (by decide) hpg
  have ha : FeeCheck.Assumptions fee totalIn totalOut :=
    { fee32 := by
        rw [hfee]
        exact WormholeSpec.referenceFromFirstReal_volumeFeeBps_lt (by decide) href
          (fun q hq => (h32 q hq).2.2.2)
      rhsBound := by
        rw [hin, hfee]
        have := WormholeSpec.privateBatchFeeRhs_lt_two_pow_52 (out := out) hlen
          (fun q hq => (h32 q hq).1)
        unfold WormholeSpec.feeDenominator at this
        exact this
      lhsBound := by
        rw [hout]
        have := WormholeSpec.privateBatchFeeLhs_add_two_pow_52_lt_modulus hlen
          (fun q hq => ⟨(h32 q hq).2.1, (h32 q hq).2.2.1⟩)
        unfold WormholeSpec.feeDenominator at this
        exact Nat.lt_of_lt_of_le this hpg }
  obtain ⟨h1, h2⟩ := feeCheck_sound hp ha h
  unfold privateBatchFeeOk WormholeSpec.feeDenominator
  rw [← hfee, ← hin, ← hout]
  exact ⟨h1, h2⟩

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
    `referenceFromFirstReal`, applied per block-hash limb, the block number and the fee). -/
theorem scan_val (xs : List (ZMod p × ZMod p)) (init : ZMod p)
    (hb : ∀ rv ∈ xs, IsBool rv.1) :
    ((List.foldl scanStep (0, init) xs).2).val = (firstRealVal init xs).val :=
  congrArg ZMod.val (scanFirst_correct xs init hb)

/-! ## Wrapper-logic `.val` composition and end-to-end soundness (private batch) -/

/-- **Wrapper-logic `.val` composition (private batch).** Given the field witnesses for the
    nullifier slots and the switch network (the nullifier + Poseidon + permutation path closed
    end to end), the pairwise uniqueness constraints, the fee comparator with its accumulator
    decodes, plus the remaining wrapper outputs as their `.val`-decoded spec values, the
    private-batch circuit satisfies `RPrivateBatch` for the verified-sponge oracle. The
    exit-grouping / metadata / first-real-reference / slot-count equalities are the explicit
    decode hypotheses; their *gadget logic* is `Wrapper.{match_contribution, dedup_select,
    scanFirst_correct, real_block_matches, ingress_mask}` (lifted by `match_contribution_val`
    / `scan_val` / `valDigest_mask`), and wiring them out of the field witness is the decode
    model (PLAN §9 gap (a)). -/
theorem private_batch_val (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (rounds : List (List (ZMod p))) (rows : List (SlotRow p)) {out : PrivateBatchOutput}
    {fee totalIn totalOut : ZMod p}
    -- nullifier slots + switches
    (hsw : ∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s)
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1)
    (hdnull : ∀ t ∈ rows, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier)
    (hnull : out.nullifiers = (network rounds (rows.map (fun t => t.1.sel))).map valDigest)
    -- uniqueness loop
    (hcol : ∀ (i j : ℕ) (hi : i < rows.length) (hj : j < rows.length), i < j →
      ∃ eq : ZMod p, (eq = 1 ↔ rows[i].1.real = rows[j].1.real) ∧
        band (band (bnot rows[i].1.isDummy) (bnot rows[j].1.isDummy)) eq = 0)
    -- fee comparator
    (hlen : rows.length ≤ 64)
    (h32 : ∀ q ∈ rows.map (fun t => t.2.1), WormholeSpec.inRange 32 q.inputAmount ∧
      WormholeSpec.inRange 32 q.outputAmount1 ∧ WormholeSpec.inRange 32 q.outputAmount2 ∧
      WormholeSpec.inRange 32 q.volumeFeeBps)
    (hfee : fee.val = out.volumeFeeBps)
    (hin : totalIn.val = maskedInputTotal (rows.map (fun t => t.2.1)))
    (hout : totalOut.val = maskedOutputTotal (rows.map (fun t => t.2.1)))
    (hfc : FeeCheck fee totalIn totalOut)
    -- remaining decoded wrapper outputs
    (hexits : out.exitSlots = groupExits (maskedChildPairs (rows.map (fun t => t.2.1))))
    (hmeta : metadataConsistent (rows.map (fun t => t.2.1)) out)
    (href : referenceFromFirstReal (rows.map (fun t => t.2.1)) out)
    (hnum : out.numExitSlots = 2 * rows.length) :
    RPrivateBatch (spongeRO perm) (rows.map (fun t => t.2.1)) (rows.map (fun t => t.2.2)) out :=
  private_batch_bridge
    { uslen := by simp [List.length_map]
      nullsPerm := by rw [hnull]; exact nullifiers_perm_bridge perm rounds rows hsw hb hd hdnull hreal
      feeOk := fee_val_bridge hpg (by simpa using hlen) h32 href hfee hin hout hfc
      exits := hexits
      metaOk := hmeta
      ref := href
      nullsDistinct := uniqueness_val_bridge rows hb hd hreal hcol
      numSlots := by rw [hnum, List.length_map] }

/-! ## End-to-end private-batch soundness over the verified sponge (the trust stack, assembled)

  The capstone: for the *concrete* random oracle `spongeRO perm` (the Step-3b/3c verified
  Poseidon2 sponge), a satisfied private-batch aggregation circuit whose recursion gadget
  accepted every child leaf proof

    (i)   satisfies the private-batch relation `RPrivateBatch` — rung (2)→(3), via
          `private_batch_val`, built on the exporter-verified gadget semantics;
    (ii)  conserves value (`outputExitTotal = maskedOutputTotal`) — rung (4);
    (iii) settles only pairwise-distinct spends (`RPrivateBatch_settles_distinct_spends`) —
          the anti-replay property the uniqueness loop exists for;
    (iv)  attests every child's leaf relation `Rleaf` — the trusted recursion seam (1),
          `leaf_proof_sound`.

  The children's 32-bit amount/fee ranges that the fee comparator's no-wrap bounds rest on
  are not assumed: they are read off each child's `Rleaf` (its `collect_32_bit_targets`
  range checks), obtained from the accepted proof through `leaf_proof_sound`.

  Two things sit *outside* the Lean hypotheses, by design:
  * **Fidelity to the Rust** — that `spongeRO`/the wrapper gadgets *are* the deployed circuit —
    is carried by the constraint exporter + differential tests, not re-proved here.
  * The public-input **decode** (`hd`/`hdnull`/`hreal`/`hnull`/`hexits`/`hmeta`/`href`/the
    fee accumulator decodes) is the wiring/copy-constraint model (PLAN §9 gap (a)).
  The only trusted *axiom* this theorem depends on is `leaf_proof_sound` (used for clause (iv)
  and for the children's range facts feeding clause (i)); `private_batch_val`, which takes
  those ranges as an explicit premise, is standard-axioms-only. -/
theorem Rleaf_ranges {ro : RandomOracle} {q : LeafPublic} {w : LeafWitness}
    (h : Rleaf ro q w) :
    WormholeSpec.inRange 32 q.inputAmount ∧ WormholeSpec.inRange 32 q.outputAmount1 ∧
      WormholeSpec.inRange 32 q.outputAmount2 ∧ WormholeSpec.inRange 32 q.volumeFeeBps :=
  ⟨h.2.2.1, h.1, h.2.1, h.2.2.2.2.1⟩

theorem private_batch_end_to_end (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (rounds : List (List (ZMod p))) (rows : List (SlotRow p)) {out : PrivateBatchOutput}
    {fee totalIn totalOut : ZMod p}
    (hsw : ∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s)
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1)
    (hdnull : ∀ t ∈ rows, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2)
    (hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier)
    (hnull : out.nullifiers = (network rounds (rows.map (fun t => t.1.sel))).map valDigest)
    (hcol : ∀ (i j : ℕ) (hi : i < rows.length) (hj : j < rows.length), i < j →
      ∃ eq : ZMod p, (eq = 1 ↔ rows[i].1.real = rows[j].1.real) ∧
        band (band (bnot rows[i].1.isDummy) (bnot rows[j].1.isDummy)) eq = 0)
    (hlen : rows.length ≤ 64)
    (hfee : fee.val = out.volumeFeeBps)
    (hin : totalIn.val = maskedInputTotal (rows.map (fun t => t.2.1)))
    (hout : totalOut.val = maskedOutputTotal (rows.map (fun t => t.2.1)))
    (hfc : FeeCheck fee totalIn totalOut)
    (hexits : out.exitSlots = groupExits (maskedChildPairs (rows.map (fun t => t.2.1))))
    (hmeta : metadataConsistent (rows.map (fun t => t.2.1)) out)
    (href : referenceFromFirstReal (rows.map (fun t => t.2.1)) out)
    (hnum : out.numExitSlots = 2 * rows.length)
    (hacc : ∀ pub ∈ rows.map (fun t => t.2.1), LeafProofAccepted (spongeRO perm) pub) :
    RPrivateBatch (spongeRO perm) (rows.map (fun t => t.2.1)) (rows.map (fun t => t.2.2)) out
      ∧ outputExitTotal out = maskedOutputTotal (rows.map (fun t => t.2.1))
      ∧ (outputExitTotal out = rawOutputTotal (realLeaves (rows.map (fun t => t.2.1)))
          ∧ (realNullifiers (rows.map (fun t => t.2.1))).Nodup)
      ∧ ∀ pub ∈ rows.map (fun t => t.2.1), ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w := by
  have hleaf : ∀ pub ∈ rows.map (fun t => t.2.1), ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w :=
    fun pub hp => leaf_proof_sound (spongeRO perm) pub (hacc pub hp)
  have h32 : ∀ q ∈ rows.map (fun t => t.2.1), WormholeSpec.inRange 32 q.inputAmount ∧
      WormholeSpec.inRange 32 q.outputAmount1 ∧ WormholeSpec.inRange 32 q.outputAmount2 ∧
      WormholeSpec.inRange 32 q.volumeFeeBps := by
    intro q hq
    obtain ⟨w, hw⟩ := hleaf q hq
    exact Rleaf_ranges hw
  have hR := private_batch_val perm hpg rounds rows hsw hb hd hdnull hreal hnull hcol hlen h32
    hfee hin hout hfc hexits hmeta href hnum
  exact ⟨hR, RPrivateBatch_value_conservation hR, RPrivateBatch_settles_distinct_spends hR, hleaf⟩

/-! ## Public-batch forwarding across `.val`

  The public-batch wrapper forwards each inner's exit slots and nullifiers through
  `select(is_dummy_i, 0, ·)`. Read through `.val`, that is exactly `forwardedSlots` /
  `forwardedNullifiers` — a `map` of the per-element mask, with the flag decoding
  `isDummyInner`. -/

/-- A field exit slot `[sum, exit(4)]`. -/
abbrev SlotF (p : ℕ) := ZMod p × Digest4 p

/-- Read a field slot back as a spec `ExitSlot`. -/
def valSlot (s : SlotF p) : ExitSlot := ⟨s.1.val, valDigest s.2⟩

/-- The public-batch forwarding mask on one slot: `select(d, 0, ·)` on every limb. -/
def maskSlot (d : ZMod p) (s : SlotF p) : SlotF p :=
  (bselect d 0 s.1, fun j => bselect d 0 (s.2 j))

theorem valSlot_maskSlot {d : ZMod p} (hb : IsBool d) (s : SlotF p) :
    valSlot (maskSlot d s) = if d = 1 then ⟨0, Digest.zero⟩ else valSlot s := by
  unfold valSlot maskSlot
  rw [val_mask hb, valDigest_mask hb]
  by_cases h1 : d = 1 <;> simp [h1]

/-- **Forwarded exit slots.** An inner's slot region, masked by its (correct, boolean) dummy
    flag and read through `.val`, is `forwardedSlots`. -/
theorem forwardedSlots_val_bridge {d : ZMod p} {o : PrivateBatchOutput} (slots : List (SlotF p))
    (hb : IsBool d) (hd : d = 1 ↔ isDummyInner o)
    (hdec : slots.map valSlot = o.exitSlots) :
    (slots.map (maskSlot d)).map valSlot = forwardedSlots o := by
  unfold forwardedSlots
  rw [← hdec, List.map_map, List.map_map]
  by_cases h1 : d = 1
  · rw [if_pos (hd.mp h1)]
    apply List.map_congr_left
    intro s _
    simp only [Function.comp, valSlot_maskSlot hb, if_pos h1]
  · rw [if_neg (fun hh => h1 (hd.mpr hh))]
    apply List.map_congr_left
    intro s _
    simp only [Function.comp, valSlot_maskSlot hb, if_neg h1]

/-- **Forwarded nullifiers.** Same for the nullifier region. -/
theorem forwardedNullifiers_val_bridge {d : ZMod p} {o : PrivateBatchOutput}
    (nulls : List (Digest4 p)) (hb : IsBool d) (hd : d = 1 ↔ isDummyInner o)
    (hdec : nulls.map valDigest = o.nullifiers) :
    (nulls.map (fun x => fun j => bselect d 0 (x j))).map valDigest = forwardedNullifiers o := by
  unfold forwardedNullifiers
  rw [← hdec, List.map_map, List.map_map]
  by_cases h1 : d = 1
  · rw [if_pos (hd.mp h1)]
    apply List.map_congr_left
    intro x _
    simp only [Function.comp, valDigest_mask hb, if_pos h1]
  · rw [if_neg (fun hh => h1 (hd.mpr hh))]
    apply List.map_congr_left
    intro x _
    simp only [Function.comp, valDigest_mask hb, if_neg h1]

end Plonky2Bridge
