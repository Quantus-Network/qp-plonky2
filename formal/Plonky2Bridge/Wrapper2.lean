/-
  Step 8d — the recorded `n = 2` private-batch wrapper, landed on `PrivateBatchConstraints`.

  `Generated/PrivateBatchWrapper2.lean` proves, from `Satisfies` on the wiring the real
  `CircuitBuilder` emitted plus `Poseidon2Rows perm`, the meaning of each of the 344 gadget
  calls `build_private_batch_constraints` made. This module reads the spec objects off the
  named targets and public inputs (`leaf`, `us`, `out`) and derives the clauses of
  `PrivateBatchConstraints` that `Plonky2Bridge` used to take as decode hypotheses.
  `private_batch_end_to_end_wired` at the end restates the capstone
  `private_batch_end_to_end` on that wiring, with `leaf_proof_sound` as its only axiom.
-/
import Plonky2Bridge.Complete
import Plonky2Spec.Generated.PrivateBatchWrapper2

namespace Plonky2Bridge.Wrapper2

open Plonky2Spec (IsBool bselect band bnot bor Digest4 network digestEq digestSelect
  bytesDigestEq_spec IsEqual match_contribution dedup_select bselect_false bselect_true
  isEqual_iff isEqual_isBool real_block_matches bor_eq_one FeeCheck rangeCheck feeDen)
open Plonky2Spec.Wiring
open Plonky2Spec.Generated (privateBatchWrapper2 privateBatchWrapper2_decode)
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash)
open WormholeSpec (Digest LeafPublic PrivateBatchOutput ExitSlot isDummyPrivateBatch Felt matchSum
  maskedChildPairs groupExits groupAux maskedInputTotal maskedOutputTotal metadataConsistent
  referenceFromFirstReal isRealB inRange RPrivateBatch)

variable {p : ℕ} [Fact p.Prime]

/-! ### Reading the spec objects off the wiring -/

/-- The 22 leaf public inputs of slot `s` (`leaf_pis_s`). -/
def leafPis : Fin 2 → Fin 22 → Target
  | 0 => privateBatchWrapper2.leaf_pis_0
  | 1 => privateBatchWrapper2.leaf_pis_1

/-- The 4-felt dummy-nullifier preimage of slot `s` (`dummy_pre_image_s`). -/
def preimage : Fin 2 → Fin 4 → Target
  | 0 => privateBatchWrapper2.dummy_pre_image_0
  | 1 => privateBatchWrapper2.dummy_pre_image_1

/-- Slot `s`'s child, decoded through `.val` at the leaf PI layout
    (`private_batch/circuit/constants.rs`). -/
def leaf (a : Assignment p) (s : Fin 2) : LeafPublic :=
  let v (k : Fin 22) : Felt := (a (leafPis s k)).val
  { assetId := v 0, outputAmount1 := v 1, outputAmount2 := v 2, volumeFeeBps := v 3,
    nullifier := ⟨v 4, v 5, v 6, v 7⟩, exitAccount1 := ⟨v 8, v 9, v 10, v 11⟩,
    exitAccount2 := ⟨v 12, v 13, v 14, v 15⟩, blockHash := ⟨v 16, v 17, v 18, v 19⟩,
    blockNumber := v 20, inputAmount := v 21 }

/-- Slot `s`'s dummy-nullifier preimage, decoded through `.val`. -/
def u (a : Assignment p) (s : Fin 2) : List Felt :=
  [(a (preimage s 0)).val, (a (preimage s 1)).val, (a (preimage s 2)).val, (a (preimage s 3)).val]

def leaves (a : Assignment p) : List LeafPublic := [leaf a 0, leaf a 1]
def us (a : Assignment p) : List (List Felt) := [u a 0, u a 1]

/-- Public input `k` of the wrapper. -/
def pi (k : ℕ) : Target := (privateBatchWrapper2 p).publicInputs.getD k (.virt 0)

/-- Public input `k`, decoded through `.val`. -/
def pv (a : Assignment p) (k : ℕ) : Felt := (a (pi (p := p) k)).val

/-- The aggregated output at the `aggregated_output` layout: header `[num_exit_slots,
    asset_id, volume_fee_bps, block_hash(4), block_number]`, `2N = 4` exit slots
    `[sum, account(4)]`, `N = 2` nullifiers. -/
def out (a : Assignment p) : PrivateBatchOutput :=
  { numExitSlots := pv a 0, assetId := pv a 1, volumeFeeBps := pv a 2,
    blockHash := ⟨pv a 3, pv a 4, pv a 5, pv a 6⟩, blockNumber := pv a 7,
    exitSlots := [⟨pv a 8, ⟨pv a 9, pv a 10, pv a 11, pv a 12⟩⟩,
      ⟨pv a 13, ⟨pv a 14, pv a 15, pv a 16, pv a 17⟩⟩,
      ⟨pv a 18, ⟨pv a 19, pv a 20, pv a 21, pv a 22⟩⟩,
      ⟨pv a 23, ⟨pv a 24, pv a 25, pv a 26, pv a 27⟩⟩],
    nullifiers := [⟨pv a 28, pv a 29, pv a 30, pv a 31⟩, ⟨pv a 32, pv a 33, pv a 34, pv a 35⟩] }

/-- Slot `s`'s field nullifier slot: the dummy flag is the `and`-fold of
    `bytes_digest_eq(block_hash_s, 0)` (calls 6 / 13), the dummy digest is the outer
    `poseidon2_hash` output (calls 324 / 330), the real digest is the leaf's nullifier limbs. -/
def slot (a : Assignment p) : Fin 2 → NullSlot p
  | 0 => { isDummy := a (.wire 1 43), dnull := fun j => a (.wire 60 (12 + j)),
           real := fun j => a (leafPis 0 ⟨4 + j, by omega⟩) }
  | 1 => { isDummy := a (.wire 2 27), dnull := fun j => a (.wire 62 (12 + j)),
           real := fun j => a (leafPis 1 ⟨4 + j, by omega⟩) }

def rows (a : Assignment p) : List (SlotRow p) :=
  [(slot a 0, leaf a 0, u a 0), (slot a 1, leaf a 1, u a 1)]

omit [Fact p.Prime] in
theorem rows_leaves (a : Assignment p) : (rows a).map (fun t => t.2.1) = leaves a := rfl
omit [Fact p.Prime] in
theorem rows_us (a : Assignment p) : (rows a).map (fun t => t.2.2) = us a := rfl

/-- The single switch of the `n = 2` odd-even network. -/
def rounds (a : Assignment p) : List (List (ZMod p)) := [[a privateBatchWrapper2.switches]]

/-! ### Gadget shapes as the trace records them -/

theorem valDigest_zero : valDigest (fun _ : Fin 4 => (0 : ZMod p)) = Digest.zero := by
  simp [valDigest, Digest.zero]

/-- `bytes_digest_eq(x, y)` as recorded: four `is_equal` flags and their `and`-fold
    `and(and(e0, e1), and(e2, e3))`. -/
theorem digestEq_fold {x y e inv : Fin 4 → ZMod p} {ab cd d : ZMod p}
    (h : ∀ j, IsEqual (x j) (y j) (e j) (inv j))
    (hab : ab = band (e 0) (e 1)) (hcd : cd = band (e 2) (e 3)) (hd : d = band ab cd) :
    IsBool d ∧ (d = 1 ↔ x = y) := by
  have hd' : d = digestEq e := by rw [hd, hab, hcd]; rfl
  rw [hd']
  exact bytesDigestEq_spec h

/-- The dummy flag `bytes_digest_eq(block_hash, 0)`: boolean, and set exactly on the
    private-batch sentinel of the decoded child. -/
theorem dummyFlag_spec {x e inv : Fin 4 → ZMod p} {z ab cd d : ZMod p} (hz : z = 0)
    (h : ∀ j, IsEqual (x j) z (e j) (inv j))
    (hab : ab = band (e 0) (e 1)) (hcd : cd = band (e 2) (e 3)) (hd : d = band ab cd) :
    IsBool d ∧ (d = 1 ↔ valDigest x = Digest.zero) := by
  subst hz
  obtain ⟨hb, hiff⟩ := digestEq_fold (y := fun _ => 0) h hab hcd hd
  refine ⟨hb, hiff.trans ⟨fun hx => by rw [hx]; exact valDigest_zero, fun hv => ?_⟩⟩
  exact valDigest_injective (hv.trans valDigest_zero.symm)

/-- `hash_dummy_nullifier_pre_image` as two recorded `poseidon2_hash` calls: the outer
    output, read through `.val`, is the realized oracle's `dummyNull` of the decoded
    preimage. -/
theorem dummyNull_spec (perm : St p → St p) {u : Fin 4 → ZMod p} {i o : Fin 4 → ZMod p}
    (hi : ∀ j, i j = spongeHash perm [u 0, u 1, u 2, u 3] j)
    (ho : ∀ j, o j = spongeHash perm [i 0, i 1, i 2, i 3] j) :
    valDigest o = (spongeRO perm).dummyNull [(u 0).val, (u 1).val, (u 2).val, (u 3).val] := by
  rw [spongeRO_dummyNull]
  have hin : spongeH perm [(u 0).val, (u 1).val, (u 2).val, (u 3).val]
      = ⟨(i 0).val, (i 1).val, (i 2).val, (i 3).val⟩ := by
    simp only [spongeH, emb_cons, emb_nil, ZMod.natCast_zmod_val, hi]
  rw [hin]
  simp only [Digest.toList, spongeH, emb_cons, emb_nil, ZMod.natCast_zmod_val, valDigest, ho]

/-! ### The exit-grouping loop -/

/-- A field `(exit, amount, flag)` triple: one masked exit slot plus its `bytes_digest_eq`
    flag against the slot being grouped. -/
abbrev Cand (p : ℕ) := Digest4 p × ZMod p × ZMod p

theorem matchSum_le (k : Digest) :
    ∀ l : List (Digest × Felt), matchSum k l ≤ (l.map Prod.snd).sum
  | [] => le_refl _
  | (k', a') :: rest => by
      simp only [matchSum, List.map_cons, List.sum_cons]
      have := matchSum_le k rest
      split
      · exact Nat.add_le_add_left this _
      · rw [zero_add]; exact le_trans this (Nat.le_add_left _ _)

/-- The field match-sum, read through `.val`, is the spec's `matchSum` on the decoded pairs
    (no wraparound when the amounts' values sum below `p`). -/
theorem matchSum_val (Ek : Digest4 p) :
    ∀ l : List (Cand p), (l.map (fun q : Cand p => q.2.1.val)).sum < p →
      ((l.map (fun q : Cand p => if q.1 = Ek then q.2.1 else 0)).sum).val
        = matchSum (valDigest Ek) (l.map (fun q : Cand p => (valDigest q.1, q.2.1.val)))
  | [], _ => by simp [matchSum]
  | q :: rest, hb => by
      simp only [List.map_cons, List.sum_cons] at hb ⊢
      have ih := matchSum_val Ek rest (by omega)
      have hle := matchSum_le (valDigest Ek) (rest.map
          (fun q : Cand p => (valDigest q.1, q.2.1.val)))
      simp only [List.map_map, Function.comp_def] at hle
      have hrest : ((rest.map (fun q : Cand p => if q.1 = Ek then q.2.1 else 0)).sum).val
          ≤ (rest.map (fun q : Cand p => q.2.1.val)).sum := by
        rw [ih]; exact hle
      have ht : (if q.1 = Ek then q.2.1 else 0).val
          = if valDigest q.1 = valDigest Ek then q.2.1.val else 0 := by
        by_cases h : q.1 = Ek
        · rw [if_pos h, if_pos (by rw [h])]
        · rw [if_neg h, if_neg (fun hv => h (valDigest_injective hv)), ZMod.val_zero]
      have hlt : (if q.1 = Ek then q.2.1 else 0).val
          + ((rest.map (fun q : Cand p => if q.1 = Ek then q.2.1 else 0)).sum).val < p := by
        rw [ht]
        split <;> omega
      rw [ZMod.val_add_of_lt hlt, ht, ih, matchSum]

/-- The decoded match-sum is bounded by the amounts. -/
theorem matchSum_val_le (Ek : Digest4 p) (l : List (Cand p))
    (hb : (l.map (fun q : Cand p => q.2.1.val)).sum < p) :
    ((l.map (fun q : Cand p => if q.1 = Ek then q.2.1 else 0)).sum).val
      ≤ (l.map (fun q : Cand p => q.2.1.val)).sum := by
  rw [matchSum_val Ek l hb]
  have hle := matchSum_le (valDigest Ek) (l.map (fun q : Cand p => (valDigest q.1, q.2.1.val)))
  simp only [List.map_map, Function.comp_def] at hle
  exact hle

/-- Every earlier candidate's flag is off when none of them matches. -/
theorem earlier_sum_zero (Ek : Digest4 p) :
    ∀ earlier : List (Cand p), (∀ q ∈ earlier, IsBool q.2.2 ∧ (q.2.2 = 1 ↔ q.1 = Ek)) →
      (¬ ∃ q ∈ earlier, q.1 = Ek) →
      (earlier.map (fun q => bselect q.2.2 q.2.1 0)).sum = 0
  | [], _, _ => rfl
  | q :: rest, hq, hn => by
      simp only [List.map_cons, List.sum_cons]
      have ⟨hb, hiff⟩ := hq q List.mem_cons_self
      have hne : q.1 ≠ Ek := fun h => hn ⟨q, List.mem_cons_self, h⟩
      have h0 : q.2.2 = 0 := hb.resolve_right (fun h1 => hne (hiff.mp h1))
      rw [bselect_false h0, zero_add]
      exact earlier_sum_zero Ek rest (fun q hq' => hq q (List.mem_cons_of_mem _ hq'))
        (fun ⟨q', hq', h⟩ => hn ⟨q', List.mem_cons_of_mem _ hq', h⟩)

theorem later_sum_eq (Ek : Digest4 p) :
    ∀ later : List (Cand p), (∀ q ∈ later, IsBool q.2.2 ∧ (q.2.2 = 1 ↔ q.1 = Ek)) →
      (later.map (fun q => bselect q.2.2 q.2.1 0)).sum
        = (later.map (fun q : Cand p => if q.1 = Ek then q.2.1 else 0)).sum
  | [], _ => rfl
  | q :: rest, hq => by
      simp only [List.map_cons, List.sum_cons]
      have ⟨hb, hiff⟩ := hq q List.mem_cons_self
      rw [match_contribution hb hiff, later_sum_eq Ek rest
        (fun q hq' => hq q (List.mem_cons_of_mem _ hq'))]

theorem mem_seen_iff (Ek : Digest4 p) (earlier : List (Cand p)) :
    valDigest Ek ∈ earlier.map (fun q => valDigest q.1) ↔ ∃ q ∈ earlier, q.1 = Ek := by
  rw [List.mem_map]
  constructor
  · rintro ⟨q, hq, h⟩; exact ⟨q, hq, valDigest_injective h⟩
  · rintro ⟨q, hq, h⟩; exact ⟨q, hq, by rw [h]⟩

/-- **One exit slot of the grouping loop, read through `.val`.** With the duplicate flag
    `dup` correct for the earlier candidates, the accumulator the sum of every candidate's
    `select(eq, amount, 0)` (self included), and the amounts not wrapping, the emitted
    `(select(dup, 0, acc), select(dup, 0, exit))` is `groupAux`'s slot for `seen` the earlier
    keys and `rest` the later pairs. -/
theorem exit_slot_val {dup acc : ZMod p} (Ek : Digest4 p) (Ak eqk : ZMod p)
    (earlier later : List (Cand p))
    (hdup : IsBool dup) (hdupP : dup = 1 ↔ ∃ q ∈ earlier, q.1 = Ek)
    (heq : ∀ q ∈ earlier ++ later, IsBool q.2.2 ∧ (q.2.2 = 1 ↔ q.1 = Ek))
    (heqk : IsBool eqk ∧ (eqk = 1 ↔ Ek = Ek))
    (hacc : acc = (earlier.map (fun q => bselect q.2.2 q.2.1 0)).sum + bselect eqk Ak 0
      + (later.map (fun q => bselect q.2.2 q.2.1 0)).sum)
    (hbound : Ak.val + (later.map (fun q => q.2.1.val)).sum < p)
    {seen : List Digest} (hseen : ∀ k, k ∈ seen ↔ k ∈ earlier.map (fun q => valDigest q.1)) :
    (⟨(bselect dup 0 acc).val, valDigest (fun j => bselect dup 0 (Ek j))⟩ : ExitSlot)
      = if valDigest Ek ∈ seen then ⟨0, Digest.zero⟩
        else ⟨Ak.val + matchSum (valDigest Ek) (later.map (fun q => (valDigest q.1, q.2.1.val))),
          valDigest Ek⟩ := by
  have hP : dup = 1 ↔ valDigest Ek ∈ seen := by
    rw [hdupP, hseen, mem_seen_iff]
  rw [dedup_select hdup hP, valDigest_mask hdup]
  by_cases hin : valDigest Ek ∈ seen
  · simp only [if_pos hin, if_pos (hP.mpr hin), ZMod.val_zero]
  · simp only [if_neg hin, if_neg (fun h1 => hin (hP.mp h1))]
    have hnone : ¬ ∃ q ∈ earlier, q.1 = Ek := fun h => hin ((hseen _).mpr ((mem_seen_iff _ _).mpr h))
    have hk : bselect eqk Ak 0 = Ak := by rw [match_contribution heqk.1 heqk.2, if_pos rfl]
    rw [hacc, earlier_sum_zero Ek earlier (fun q hq => heq q (List.mem_append_left _ hq)) hnone,
      zero_add, hk, later_sum_eq Ek later (fun q hq => heq q (List.mem_append_right _ hq))]
    have hb' : (later.map (fun q : Cand p => q.2.1.val)).sum < p :=
      lt_of_le_of_lt (Nat.le_add_left _ _) hbound
    rw [ZMod.val_add_of_lt (lt_of_le_of_lt (Nat.add_le_add_left (matchSum_val_le Ek later hb') _)
      hbound), matchSum_val Ek later hb']

/-! ### The constant targets -/

theorem consts (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a) :
    a (.virt 18978) = 1 ∧ a (.virt 18979) = 0 ∧ a (.virt 18980) = 4 ∧
      a (.virt 19017) = 10000 := by
  have hc := h.2.2
  simp only [privateBatchWrapper2, List.forall_mem_cons, List.not_mem_nil, false_implies,
    implies_true, and_true] at hc
  exact ⟨hc.1, hc.2.1, hc.2.2.1, hc.2.2.2.1⟩

/-! ### The nullifier path -/

set_option maxHeartbeats 1000000 in
/-- **The nullifier path, decoded from the wiring.** For every satisfying assignment (with
    the four `Poseidon2Gate` rows computing `perm`): the switch and both dummy flags are
    boolean, each flag is set exactly on the decoded child's sentinel, each dummy digest is
    `dummyNull` of the decoded preimage, each real digest is the decoded child's nullifier,
    the public nullifier region is the switch network applied to the slot selections, the
    pairwise uniqueness constraint holds, and the slot count is within the cap — the
    `hsw`/`hb`/`hd`/`hdnull`/`hreal`/`hnull`/`hcol`/`hlen` clauses of
    `PrivateBatchConstraints`, plus `hnum` for the constant header. -/
theorem nullifier_path (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a) :
    (∀ ss ∈ rounds a, ∀ s ∈ ss, IsBool s) ∧
    (∀ t ∈ rows a, IsBool t.1.isDummy) ∧
    (∀ t ∈ rows a, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1) ∧
    (∀ t ∈ rows a, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2) ∧
    (∀ t ∈ rows a, valDigest t.1.real = t.2.1.nullifier) ∧
    (out a).nullifiers = (network (rounds a) ((rows a).map (fun t => t.1.sel))).map valDigest ∧
    (∀ (i j : ℕ) (hi : i < (rows a).length) (hj : j < (rows a).length), i < j →
      ∃ eq : ZMod p, (eq = 1 ↔ (rows a)[i].1.real = (rows a)[j].1.real) ∧
        band (band (bnot (rows a)[i].1.isDummy) (bnot (rows a)[j].1.isDummy)) eq = 0) ∧
    (rows a).length ≤ 64 ∧
    (out a).numExitSlots = 2 * (rows a).length := by
  obtain ⟨-, kzero, kfour, -⟩ := consts a h
  have hf := privateBatchWrapper2_decode perm a h hp
  have e00 := hf.1.1
  have e01 := hf.1.2.1
  have e02 := hf.1.2.2.1
  have e03 := hf.1.2.2.2.1
  have a0a := hf.1.2.2.2.2.1
  have a0b := hf.1.2.2.2.2.2.1
  have d0 := hf.1.2.2.2.2.2.2.1
  have e10 := hf.1.2.2.2.2.2.2.2.1
  have e11 := hf.1.2.2.2.2.2.2.2.2.1
  have e12 := hf.1.2.2.2.2.2.2.2.2.2.1
  have e13 := hf.1.2.2.2.2.2.2.2.2.2.2.1
  have a1a := hf.1.2.2.2.2.2.2.2.2.2.2.2.1
  have a1b := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have d1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have n0 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have n1 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have br := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q0 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q1 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q2 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q3 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have qa := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have qb := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have qc := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have col := hf.2.2.2.2.2.2.2.2.2.2.1
  have cz := hf.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨hi00, hi01, hi02, hi03⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨ho00, ho01, ho02, ho03⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s00 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s01 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s02 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s03 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨hi10, hi11, hi12, hi13⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨ho10, ho11, ho12, ho13⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s10 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s11 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s12 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s13 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sw := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o00 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o10 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o01 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o11 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o02 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o12 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o03 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o13 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  -- dummy flags
  obtain ⟨hb0, hd0⟩ := dummyFlag_spec (x := fun j => a (leafPis 0 ⟨16 + j, by omega⟩))
    (e := ![a (.virt 18981), a (.virt 18983), a (.virt 18985), a (.virt 18987)])
    (inv := ![a (.virt 18982), a (.virt 18984), a (.virt 18986), a (.virt 18988)]) kzero
    (fun j => by fin_cases j <;> assumption) a0a a0b d0
  obtain ⟨hb1, hd1⟩ := dummyFlag_spec (x := fun j => a (leafPis 1 ⟨16 + j, by omega⟩))
    (e := ![a (.virt 18989), a (.virt 18991), a (.virt 18993), a (.virt 18995)])
    (inv := ![a (.virt 18990), a (.virt 18992), a (.virt 18994), a (.virt 18996)]) kzero
    (fun j => by fin_cases j <;> assumption) a1a a1b d1
  -- dummy digests
  have hn0 := dummyNull_spec perm (u := fun j => a (preimage 0 j))
    (i := fun j => a (.wire 59 (12 + j))) (o := fun j => a (.wire 60 (12 + j)))
    (fun j => by fin_cases j <;> assumption) (fun j => by fin_cases j <;> assumption)
  have hn1 := dummyNull_spec perm (u := fun j => a (preimage 1 j))
    (i := fun j => a (.wire 61 (12 + j))) (o := fun j => a (.wire 62 (12 + j)))
    (fun j => by fin_cases j <;> assumption) (fun j => by fin_cases j <;> assumption)
  -- uniqueness flag
  obtain ⟨-, hq⟩ := digestEq_fold (x := fun j => a (leafPis 0 ⟨4 + j, by omega⟩))
    (y := fun j => a (leafPis 1 ⟨4 + j, by omega⟩))
    (e := ![a (.virt 19195), a (.virt 19197), a (.virt 19199), a (.virt 19201)])
    (inv := ![a (.virt 19196), a (.virt 19198), a (.virt 19200), a (.virt 19202)])
    (fun j => by fin_cases j <;> assumption) qa qb qc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [rounds, List.mem_singleton, forall_eq]
    exact sw
  · simp only [rows, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hb0, hb1⟩
  · simp only [rows, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hd0, hd1⟩
  · simp only [rows, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hn0, hn1⟩
  · simp only [rows, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨rfl, rfl⟩
  · show [(⟨(a (.wire 63 19)).val, (a (.wire 63 35)).val, (a (.wire 63 51)).val,
        (a (.wire 64 7)).val⟩ : Digest),
      ⟨(a (.wire 63 27)).val, (a (.wire 63 43)).val, (a (.wire 63 59)).val, (a (.wire 64 15)).val⟩]
      = [valDigest (digestSelect (a privateBatchWrapper2.switches) (slot a 1).sel (slot a 0).sel),
         valDigest (digestSelect (a privateBatchWrapper2.switches) (slot a 0).sel (slot a 1).sel)]
    rw [o00, o01, o02, o03, o10, o11, o12, o13, s00, s01, s02, s03, s10, s11, s12, s13]
    rfl
  · intro i j hi hj hij
    have hlen : (rows a).length = 2 := rfl
    rw [hlen] at hi hj
    obtain rfl : i = 0 := by omega
    obtain rfl : j = 1 := by omega
    refine ⟨a (.wire 57 19), hq, ?_⟩
    show band (band (bnot (a (.wire 1 43))) (bnot (a (.wire 2 27)))) (a (.wire 57 19)) = 0
    rw [← n0, ← n1, ← br, ← col, cz, kzero]
  · show 2 ≤ 64
    decide
  · show (a (.virt 18980)).val = 2 * 2
    rw [kfour]
    have h4 : (4 : ℕ) < p := lt_of_lt_of_le (by decide) hpg
    rw [← Nat.cast_ofNat, ZMod.val_natCast_of_lt h4]


/-! ### The full constraint predicate -/

/-- `bytes_digest_eq` on two four-limb digests, with its four `is_equal` flags spelled out. -/
theorem digestEq_fold4 {x y : Fin 4 → ZMod p} {e0 e1 e2 e3 i0 i1 i2 i3 ab cd d : ZMod p}
    (h0 : IsEqual (x 0) (y 0) e0 i0) (h1 : IsEqual (x 1) (y 1) e1 i1)
    (h2 : IsEqual (x 2) (y 2) e2 i2) (h3 : IsEqual (x 3) (y 3) e3 i3)
    (hab : ab = band e0 e1) (hcd : cd = band e2 e3) (hd : d = band ab cd) :
    IsBool d ∧ (d = 1 ↔ x = y) :=
  digestEq_fold (e := ![e0, e1, e2, e3]) (inv := ![i0, i1, i2, i3])
    (fun j => by fin_cases j <;> assumption) hab hcd hd

/-- Two masked values of one slot, read through `.val`. -/
theorem val_mask2 {d x y : ZMod p} (hb : IsBool d) :
    (bselect d 0 x).val + (bselect d 0 y).val = if d = 1 then 0 else x.val + y.val := by
  rw [val_mask hb, val_mask hb]
  split <;> simp

/-- The masked exit accounts of the four `(child, output)` slots, in slot order
    (calls 60–63, 65–68, 70–73, 75–78). -/
def E (a : Assignment p) : Fin 4 → Digest4 p :=
  ![fun j => a (![Target.wire 9 35, .wire 9 43, .wire 9 51, .wire 9 59] j),
    fun j => a (![Target.wire 11 15, .wire 11 23, .wire 11 31, .wire 11 39] j),
    fun j => a (![Target.wire 11 55, .wire 12 3, .wire 12 11, .wire 12 19] j),
    fun j => a (![Target.wire 12 35, .wire 12 43, .wire 12 51, .wire 12 59] j)]

set_option maxHeartbeats 2000000 in
/-- **The wrapper's constraints, decoded from the wiring.** Every satisfying assignment (with
    the `Poseidon2Gate` rows computing `perm`) satisfies `PrivateBatchConstraints` on the
    decoded children, preimages and output, with the reference fee and the two dummy-masked
    totals read off their accumulator wires. One hypothesis comes from outside the wrapper:
    `h32`, the children's 32-bit amount/fee ranges, which the *leaf* circuit range-checks
    (`Rleaf_ranges` obtains them from accepted leaf proofs). -/
theorem constraints (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a)
    (h32 : ∀ q ∈ leaves a, inRange 32 q.inputAmount ∧ inRange 32 q.outputAmount1 ∧
      inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps) :
    PrivateBatchConstraints perm (rounds a) (rows a) (a (.wire 4 27)) (a (.wire 6 23))
      (a (.wire 6 35)) (out a) := by
  obtain ⟨hsw, hb, hd, hdnull, hreal, hnull, hcol, hlen, hnum⟩ := nullifier_path perm hpg a h hp
  obtain ⟨kone, kzero, -, kten⟩ := consts a h
  have hb0 : IsBool (a (.wire 1 43)) := hb _ List.mem_cons_self
  have hb1 : IsBool (a (.wire 2 27)) := hb _ (List.mem_cons_of_mem _ List.mem_cons_self)
  have hd0 : a (.wire 1 43) = 1 ↔ isDummyPrivateBatch (leaf a 0) := hd _ List.mem_cons_self
  have hd1 : a (.wire 2 27) = 1 ↔ isDummyPrivateBatch (leaf a 1) :=
    hd _ (List.mem_cons_of_mem _ List.mem_cons_self)
  obtain ⟨hin0, ho10, ho20, hf0⟩ := h32 (leaf a 0) List.mem_cons_self
  obtain ⟨hin1, ho11, ho21, hf1⟩ := h32 (leaf a 1) (List.mem_cons_of_mem _ List.mem_cons_self)
  simp only [leaf, inRange] at hin0 ho10 ho20 hf0 hin1 ho11 ho21 hf1
  have hin0 : (a (.virt 9484)).val < 2 ^ 32 := hin0
  have ho10 : (a (.virt 9464)).val < 2 ^ 32 := ho10
  have ho20 : (a (.virt 9465)).val < 2 ^ 32 := ho20
  have hf0 : (a (.virt 9466)).val < 2 ^ 32 := hf0
  have hin1 : (a (.virt 18969)).val < 2 ^ 32 := hin1
  have ho11 : (a (.virt 18949)).val < 2 ^ 32 := ho11
  have ho21 : (a (.virt 18950)).val < 2 ^ 32 := ho21
  have hf1 : (a (.virt 18951)).val < 2 ^ 32 := hf1
  have hp34 : 2 ^ 34 ≤ p := le_trans (by decide) hpg
  have hf := privateBatchWrapper2_decode perm a h hp
  have e00 := hf.1.1
  have e01 := hf.1.2.1
  have e02 := hf.1.2.2.1
  have e03 := hf.1.2.2.2.1
  have a0a := hf.1.2.2.2.2.1
  have a0b := hf.1.2.2.2.2.2.1
  have d0 := hf.1.2.2.2.2.2.2.1
  have e10 := hf.1.2.2.2.2.2.2.2.1
  have e11 := hf.1.2.2.2.2.2.2.2.2.1
  have e12 := hf.1.2.2.2.2.2.2.2.2.2.1
  have e13 := hf.1.2.2.2.2.2.2.2.2.2.2.1
  have a1a := hf.1.2.2.2.2.2.2.2.2.2.2.2.1
  have a1b := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have d1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have r0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc0_0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc0_1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc0_2 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc0_3 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc0_4 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc0_5 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have r1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have nf := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have tk1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc1_0 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc1_1 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc1_2 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc1_3 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sc1_4 := hf.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have sc1_5 := hf.2.1.1
  have cb0_0 := hf.2.1.2.2.1
  have cb0_1 := hf.2.1.2.2.2.1
  have cb0_2 := hf.2.1.2.2.2.2.1
  have cb0_3 := hf.2.1.2.2.2.2.2.1
  have cba0 := hf.2.1.2.2.2.2.2.2.1
  have cbb0 := hf.2.1.2.2.2.2.2.2.2.1
  have cbc0 := hf.2.1.2.2.2.2.2.2.2.2.1
  have cbo0 := hf.2.1.2.2.2.2.2.2.2.2.2.1
  have cbk0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.1
  have cf0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cfo0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cfk0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_2 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cb1_3 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cba1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbb1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbc1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbo1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cbk1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ca1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cf1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cfo1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have cfk1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mE0_0 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mE0_1 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mE0_2 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mE0_3 := hf.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have mA0 := hf.2.2.1.1
  have mE1_0 := hf.2.2.1.2.1
  have mE1_1 := hf.2.2.1.2.2.1
  have mE1_2 := hf.2.2.1.2.2.2.1
  have mE1_3 := hf.2.2.1.2.2.2.2.1
  have mA1 := hf.2.2.1.2.2.2.2.2.1
  have mE2_0 := hf.2.2.1.2.2.2.2.2.2.1
  have mE2_1 := hf.2.2.1.2.2.2.2.2.2.2.1
  have mE2_2 := hf.2.2.1.2.2.2.2.2.2.2.2.1
  have mE2_3 := hf.2.2.1.2.2.2.2.2.2.2.2.2.1
  have mA2 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have mE3_0 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have mE3_1 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mE3_2 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mE3_3 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mA3 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mi0 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mi1 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ti := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have to1 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have to2 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have to3 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fc := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have fcr := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have lhs := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have rhs := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dif := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dr := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq00_0 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq00_1 := hf.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have mq00_2 := hf.2.2.2.1.1
  have mq00_3 := hf.2.2.2.1.2.1
  have mqa00 := hf.2.2.2.1.2.2.1
  have mqb00 := hf.2.2.2.1.2.2.2.1
  have mqc00 := hf.2.2.2.1.2.2.2.2.1
  have ms00 := hf.2.2.2.1.2.2.2.2.2.1
  have mq01_0 := hf.2.2.2.1.2.2.2.2.2.2.2.1
  have mq01_1 := hf.2.2.2.1.2.2.2.2.2.2.2.2.1
  have mq01_2 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have mq01_3 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have mqa01 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb01 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc01 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms01 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma01 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq02_0 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq02_1 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq02_2 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq02_3 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa02 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb02 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc02 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms02 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma02 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq03_0 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq03_1 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq03_2 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq03_3 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa03 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb03 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc03 := hf.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have ms03 := hf.2.2.2.2.1.1
  have ma03 := hf.2.2.2.2.1.2.1
  have os0 := hf.2.2.2.2.1.2.2.1
  have oe0_0 := hf.2.2.2.2.1.2.2.2.1
  have oe0_1 := hf.2.2.2.2.1.2.2.2.2.1
  have oe0_2 := hf.2.2.2.2.1.2.2.2.2.2.1
  have oe0_3 := hf.2.2.2.2.1.2.2.2.2.2.2.1
  have rc0 := hf.2.2.2.2.1.2.2.2.2.2.2.2.1
  have dq10_0 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.1
  have dq10_1 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have dq10_2 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have dq10_3 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have dqa10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqb10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqc10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq10_0 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq10_1 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq10_2 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq10_3 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms10 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq11_0 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq11_1 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq11_2 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq11_3 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa11 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb11 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc11 := hf.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have ms11 := hf.2.2.2.2.2.1.1
  have ma11 := hf.2.2.2.2.2.1.2.1
  have mq12_0 := hf.2.2.2.2.2.1.2.2.1
  have mq12_1 := hf.2.2.2.2.2.1.2.2.2.1
  have mq12_2 := hf.2.2.2.2.2.1.2.2.2.2.1
  have mq12_3 := hf.2.2.2.2.2.1.2.2.2.2.2.1
  have mqa12 := hf.2.2.2.2.2.1.2.2.2.2.2.2.1
  have mqb12 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.1
  have mqc12 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.1
  have ms12 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have ma12 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have mq13_0 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have mq13_1 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq13_2 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq13_3 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa13 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb13 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc13 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms13 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma13 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have os1 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe1_0 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe1_1 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe1_2 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe1_3 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have rc1 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq20_0 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq20_1 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq20_2 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq20_3 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqa20 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqb20 := hf.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have dqc20 := hf.2.2.2.2.2.2.1.1
  have dq21_0 := hf.2.2.2.2.2.2.1.2.2.1
  have dq21_1 := hf.2.2.2.2.2.2.1.2.2.2.1
  have dq21_2 := hf.2.2.2.2.2.2.1.2.2.2.2.1
  have dq21_3 := hf.2.2.2.2.2.2.1.2.2.2.2.2.1
  have dqa21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.1
  have dqb21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.1
  have dqc21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.1
  have do21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have mq20_0 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have mq20_1 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have mq20_2 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq20_3 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa20 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb20 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc20 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms20 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq21_0 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq21_1 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq21_2 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq21_3 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma21 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq22_0 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq22_1 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq22_2 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq22_3 := hf.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have mqa22 := hf.2.2.2.2.2.2.2.1.1
  have mqb22 := hf.2.2.2.2.2.2.2.1.2.1
  have mqc22 := hf.2.2.2.2.2.2.2.1.2.2.1
  have ms22 := hf.2.2.2.2.2.2.2.1.2.2.2.1
  have ma22 := hf.2.2.2.2.2.2.2.1.2.2.2.2.1
  have mq23_0 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.1
  have mq23_1 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.1
  have mq23_2 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.1
  have mq23_3 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.1
  have mqa23 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have mqb23 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have mqc23 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have ms23 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma23 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have os2 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe2_0 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe2_1 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe2_2 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe2_3 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have rc2 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq30_0 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq30_1 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq30_2 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq30_3 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqa30 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqb30 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dqc30 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq31_0 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq31_1 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq31_2 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have dq31_3 := hf.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have dqa31 := hf.2.2.2.2.2.2.2.2.1.1
  have dqb31 := hf.2.2.2.2.2.2.2.2.1.2.1
  have dqc31 := hf.2.2.2.2.2.2.2.2.1.2.2.1
  have do31 := hf.2.2.2.2.2.2.2.2.1.2.2.2.1
  have dq32_0 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.1
  have dq32_1 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.1
  have dq32_2 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.1
  have dq32_3 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.1
  have dqa32 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.1
  have dqb32 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have dqc32 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have do32 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have mq30_0 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq30_1 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq30_2 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq30_3 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa30 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb30 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc30 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms30 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq31_0 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq31_1 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq31_2 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq31_3 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqa31 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb31 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc31 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms31 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma31 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq32_0 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mq32_1 := hf.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have mq32_2 := hf.2.2.2.2.2.2.2.2.2.1.1
  have mq32_3 := hf.2.2.2.2.2.2.2.2.2.1.2.1
  have mqa32 := hf.2.2.2.2.2.2.2.2.2.1.2.2.1
  have mqb32 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.1
  have mqc32 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.1
  have ms32 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.1
  have ma32 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.1
  have mq33_0 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.1
  have mq33_1 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.1
  have mq33_2 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.1
  have mq33_3 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.1
  have mqa33 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.1
  have mqb33 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1
  have mqc33 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ms33 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have ma33 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have os3 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe3_0 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe3_1 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe3_2 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have oe3_3 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have rc3 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have n0 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have n1 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have br := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q0 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q1 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q2 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have q3 := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have qa := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have qb := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have qc := hf.2.2.2.2.2.2.2.2.2.1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have col := hf.2.2.2.2.2.2.2.2.2.2.1
  have cz := hf.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨hi00, hi01, hi02, hi03⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨ho00, ho01, ho02, ho03⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s00 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s01 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s02 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s03 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨hi10, hi11, hi12, hi13⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨ho10, ho11, ho12, ho13⟩ := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s10 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s11 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s12 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have s13 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have sw := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o00 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o10 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o01 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o11 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o02 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o12 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o03 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have o13 := hf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  -- the masked amounts are 32-bit
  have hA0 : (a (.wire 11 7)).val < 2 ^ 32 := by
    rw [mA0, kzero, val_mask hb0]; split <;> omega
  have hA1 : (a (.wire 11 47)).val < 2 ^ 32 := by
    rw [mA1, kzero, val_mask hb0]; split <;> omega
  have hA2 : (a (.wire 12 27)).val < 2 ^ 32 := by
    rw [mA2, kzero, val_mask hb1]; split <;> omega
  have hA3 : (a (.wire 13 7)).val < 2 ^ 32 := by
    rw [mA3, kzero, val_mask hb1]; split <;> omega
  -- the masked pairs are `maskedChildPairs`
  have hE0 : valDigest (E a 0) = if isDummyPrivateBatch (leaf a 0) then Digest.zero
      else (leaf a 0).exitAccount1 := by
    have : E a 0 = fun j : Fin 4 => bselect (a (.wire 1 43)) 0 (a (leafPis 0 ⟨8 +
        j, by omega⟩)) := by
      rw [kzero] at mE0_0 mE0_1 mE0_2 mE0_3
      funext j; fin_cases j <;> assumption
    rw [this, valDigest_mask hb0]
    simp only [hd0]
    rfl
  have hE1 : valDigest (E a 1) = if isDummyPrivateBatch (leaf a 0) then Digest.zero
      else (leaf a 0).exitAccount2 := by
    have : E a 1 = fun j : Fin 4 => bselect (a (.wire 1 43)) 0 (a (leafPis 0 ⟨12 +
        j, by omega⟩)) := by
      rw [kzero] at mE1_0 mE1_1 mE1_2 mE1_3
      funext j; fin_cases j <;> assumption
    rw [this, valDigest_mask hb0]
    simp only [hd0]
    rfl
  have hE2 : valDigest (E a 2) = if isDummyPrivateBatch (leaf a 1) then Digest.zero
      else (leaf a 1).exitAccount1 := by
    have : E a 2 = fun j : Fin 4 => bselect (a (.wire 2 27)) 0 (a (leafPis 1 ⟨8 +
        j, by omega⟩)) := by
      rw [kzero] at mE2_0 mE2_1 mE2_2 mE2_3
      funext j; fin_cases j <;> assumption
    rw [this, valDigest_mask hb1]
    simp only [hd1]
    rfl
  have hE3 : valDigest (E a 3) = if isDummyPrivateBatch (leaf a 1) then Digest.zero
      else (leaf a 1).exitAccount2 := by
    have : E a 3 = fun j : Fin 4 => bselect (a (.wire 2 27)) 0 (a (leafPis 1 ⟨12 +
        j, by omega⟩)) := by
      rw [kzero] at mE3_0 mE3_1 mE3_2 mE3_3
      funext j; fin_cases j <;> assumption
    rw [this, valDigest_mask hb1]
    simp only [hd1]
    rfl
  have hV0 : (a (.wire 11 7)).val = if isDummyPrivateBatch (leaf a 0) then 0
      else (leaf a 0).outputAmount1 := by
    rw [mA0, kzero, val_mask hb0]; simp only [hd0]; rfl
  have hV1 : (a (.wire 11 47)).val = if isDummyPrivateBatch (leaf a 0) then 0
      else (leaf a 0).outputAmount2 := by
    rw [mA1, kzero, val_mask hb0]; simp only [hd0]; rfl
  have hV2 : (a (.wire 12 27)).val = if isDummyPrivateBatch (leaf a 1) then 0
      else (leaf a 1).outputAmount1 := by
    rw [mA2, kzero, val_mask hb1]; simp only [hd1]; rfl
  have hV3 : (a (.wire 13 7)).val = if isDummyPrivateBatch (leaf a 1) then 0
      else (leaf a 1).outputAmount2 := by
    rw [mA3, kzero, val_mask hb1]; simp only [hd1]; rfl
  have hpairs : maskedChildPairs (leaves a) =
      [(valDigest (E a 0), (a (.wire 11 7)).val), (valDigest (E a 1), (a (.wire 11 47)).val),
       (valDigest (E a 2), (a (.wire 12 27)).val), (valDigest (E a 3), (a (.wire 13 7)).val)] := by
    rw [hE0, hE1, hE2, hE3, hV0, hV1, hV2, hV3]
    simp only [leaves, maskedChildPairs]
    split_ifs <;> rfl
  -- exit slot 0
  have hc00 := digestEq_fold4 (x := E a 0) (y := E a 0)
      mq00_0 mq00_1 mq00_2 mq00_3 mqa00 mqb00 mqc00
  have hc01 := digestEq_fold4 (x := E a 1) (y := E a 0)
      mq01_0 mq01_1 mq01_2 mq01_3 mqa01 mqb01 mqc01
  have hc02 := digestEq_fold4 (x := E a 2) (y := E a 0)
      mq02_0 mq02_1 mq02_2 mq02_3 mqa02 mqb02 mqc02
  have hc03 := digestEq_fold4 (x := E a 3) (y := E a 0)
      mq03_0 mq03_1 mq03_2 mq03_3 mqa03 mqb03 mqc03
  have hdup0 : IsBool (0 : ZMod p) := Or.inl rfl
  have hdupP0 : (0 : ZMod p) = 1 ↔ ∃ q ∈ ([] : List (Cand p)), q.1 = E a 0 := by
    exact ⟨fun h => absurd h zero_ne_one, fun ⟨_, hq, _⟩ => absurd hq List.not_mem_nil⟩
  have heq0 : ∀ q ∈ ([] : List (Cand p)) ++ [(E a 1, (a (.wire 11 47)), a (.wire 17 51)),
      (E a 2, (a (.wire 12 27)), a (.wire 19 35)), (E a 3,
      (a (.wire 13 7)), a (.wire 21 19))], IsBool q.2.2 ∧ (q.2.2 = 1 ↔ q.1 = E a 0) := by
    simp only [List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hc01, hc02, hc03⟩
  have hacc0 : a (.wire 6 47) = (([] : List (Cand p)).map
      (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum + bselect (a (.wire 17 7))
      (a (.wire 11 7)) 0 + (([(E a 1, (a (.wire 11 47)), a (.wire 17 51)), (E a 2,
      (a (.wire 12 27)), a (.wire 19 35)), (E a 3,
      (a (.wire 13 7)), a (.wire 21 19))] : List (Cand p)).map
      (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum := by
    rw [ma03, ma02, ma01, ms00, ms01, ms02, ms03, kzero]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    ring
  have hbound0 : (a (.wire 11 7)).val + (([(E a 1, (a (.wire 11 47)), a (.wire 17 51)),
      (E a 2, (a (.wire 12 27)), a (.wire 19 35)), (E a 3,
      (a (.wire 13 7)), a (.wire 21 19))] : List (Cand p)).map
      (fun q : Cand p => q.2.1.val)).sum < p := by
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    omega
  have hseen0 : ∀ x, x ∈ ([] : List Digest) ↔ x ∈ ([] : List (Cand p)).map
      (fun q : Cand p => valDigest q.1) := by
    intro x; simp only [List.map_nil]
  have hx0 : (⟨(a (.wire 22 7)).val, ⟨(a (.wire 22 15)).val, (a (.wire 22 23)).val,
      (a (.wire 22 31)).val, (a (.wire 22 39)).val⟩⟩ : ExitSlot)
      = if valDigest (E a 0) ∈ ([] : List Digest) then ⟨0, Digest.zero⟩
        else ⟨(a (.wire 11 7)).val + matchSum (valDigest (E a 0)) [(valDigest (E a 1),
            (a (.wire 11 47)).val), (valDigest (E a 2),
            (a (.wire 12 27)).val), (valDigest (E a 3), (a (.wire 13 7)).val)],
            valDigest (E a 0)⟩ := by
    have := exit_slot_val (dup := 0) (acc := a (.wire 6 47)) (E a 0) (a (.wire 11 7))
        (a (.wire 17 7)) [] [(E a 1, (a (.wire 11 47)), a (.wire 17 51)), (E a 2,
        (a (.wire 12 27)), a (.wire 19 35)), (E a 3, (a (.wire 13 7)), a (.wire 21 19))
        ] hdup0 hdupP0 heq0 hc00 hacc0 hbound0 hseen0
    simp only [List.map_cons, List.map_nil] at this
    rw [os0, oe0_0, oe0_1, oe0_2, oe0_3, kzero]
    exact this
  -- exit slot 1
  have hc10 := digestEq_fold4 (x := E a 0) (y := E a 1)
      mq10_0 mq10_1 mq10_2 mq10_3 mqa10 mqb10 mqc10
  have hc11 := digestEq_fold4 (x := E a 1) (y := E a 1)
      mq11_0 mq11_1 mq11_2 mq11_3 mqa11 mqb11 mqc11
  have hc12 := digestEq_fold4 (x := E a 2) (y := E a 1)
      mq12_0 mq12_1 mq12_2 mq12_3 mqa12 mqb12 mqc12
  have hc13 := digestEq_fold4 (x := E a 3) (y := E a 1)
      mq13_0 mq13_1 mq13_2 mq13_3 mqa13 mqb13 mqc13
  have hdq10 := digestEq_fold4 (x := E a 0) (y := E a 1)
      dq10_0 dq10_1 dq10_2 dq10_3 dqa10 dqb10 dqc10
  have hdup1 : IsBool (a (.wire 25 3)) := hdq10.1
  have hdupP1 : a (.wire 25 3) = 1 ↔ ∃ q ∈ ([(E a 0,
      (a (.wire 11 7)), a (.wire 25 47))] : List (Cand p)), q.1 = E a 1 := by
    rw [hdq10.2]; simp
  have heq1 : ∀ q ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 25 47))] : List (Cand p)) ++
      [(E a 2, (a (.wire 12 27)), a (.wire 29 15)), (E a 3,
      (a (.wire 13 7)), a (.wire 29 59))], IsBool q.2.2 ∧ (q.2.2 = 1 ↔ q.1 = E a 1) := by
    simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hc10, hc12, hc13⟩
  have hacc1 : a (.wire 6 59) = (([(E a 0, (a (.wire 11 7)), a (.wire 25 47))]
      : List (Cand p)).map (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum + bselect
      (a (.wire 27 31)) (a (.wire 11 47)) 0 + (([(E a 2, (a (.wire 12 27)), a (.wire 29 15)),
      (E a 3, (a (.wire 13 7)), a (.wire 29 59))] : List (Cand p)).map
      (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum := by
    rw [ma13, ma12, ma11, ms10, ms11, ms12, ms13, kzero]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    ring
  have hbound1 : (a (.wire 11 47)).val + (([(E a 2, (a (.wire 12 27)), a (.wire 29 15)),
      (E a 3, (a (.wire 13 7)), a (.wire 29 59))] : List (Cand p)).map
      (fun q : Cand p => q.2.1.val)).sum < p := by
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    omega
  have hseen1 : ∀ x, x ∈ ([valDigest (E a 0)] : List Digest) ↔ x ∈ ([(E a 0,
      (a (.wire 11 7)), a (.wire 25 47))] : List (Cand p)).map
      (fun q : Cand p => valDigest q.1) := by
    intro x; simp only [List.map_cons, List.map_nil]
  have hx1 : (⟨(a (.wire 30 47)).val, ⟨(a (.wire 30 55)).val, (a (.wire 31 3)).val,
      (a (.wire 31 11)).val, (a (.wire 31 19)).val⟩⟩ : ExitSlot)
      = if valDigest (E a 1) ∈ ([valDigest (E a 0)] : List Digest) then ⟨0, Digest.zero⟩
        else ⟨(a (.wire 11 47)).val + matchSum (valDigest (E a 1)) [(valDigest (E a 2),
            (a (.wire 12 27)).val), (valDigest (E a 3), (a (.wire 13 7)).val)],
            valDigest (E a 1)⟩ := by
    have := exit_slot_val (dup := a (.wire 25 3)) (acc := a (.wire 6 59)) (E a 1)
        (a (.wire 11 47)) (a (.wire 27 31)) [(E a 0, (a (.wire 11 7)), a (.wire 25 47))]
        [(E a 2, (a (.wire 12 27)), a (.wire 29 15)), (E a 3,
        (a (.wire 13 7)), a (.wire 29 59))] hdup1 hdupP1 heq1 hc11 hacc1 hbound1 hseen1
    simp only [List.map_cons, List.map_nil] at this
    rw [os1, oe1_0, oe1_1, oe1_2, oe1_3, kzero]
    exact this
  -- exit slot 2
  have hc20 := digestEq_fold4 (x := E a 0) (y := E a 2)
      mq20_0 mq20_1 mq20_2 mq20_3 mqa20 mqb20 mqc20
  have hc21 := digestEq_fold4 (x := E a 1) (y := E a 2)
      mq21_0 mq21_1 mq21_2 mq21_3 mqa21 mqb21 mqc21
  have hc22 := digestEq_fold4 (x := E a 2) (y := E a 2)
      mq22_0 mq22_1 mq22_2 mq22_3 mqa22 mqb22 mqc22
  have hc23 := digestEq_fold4 (x := E a 3) (y := E a 2)
      mq23_0 mq23_1 mq23_2 mq23_3 mqa23 mqb23 mqc23
  have hdq20 := digestEq_fold4 (x := E a 0) (y := E a 2)
      dq20_0 dq20_1 dq20_2 dq20_3 dqa20 dqb20 dqc20
  have hdq21 := digestEq_fold4 (x := E a 1) (y := E a 2)
      dq21_0 dq21_1 dq21_2 dq21_3 dqa21 dqb21 dqc21
  have hdup2 : IsBool (a (.wire 36 3)) := by rw [do21]; exact Plonky2Spec.bor_isBool hdq20.1 hdq21.1
  have hdupP2 : a (.wire 36 3) = 1 ↔ ∃ q ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 38 11)),
      (E a 1, (a (.wire 11 47)), a (.wire 38 55))] : List (Cand p)), q.1 = E a 2 := by
    rw [do21, bor_eq_one hdq20.1 hdq21.1, hdq20.2, hdq21.2]; simp
  have heq2 : ∀ q ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 38 11)), (E a 1,
      (a (.wire 11 47)), a (.wire 38 55))] : List (Cand p)) ++ [(E a 3,
      (a (.wire 13 7)), a (.wire 42 23))], IsBool q.2.2 ∧ (q.2.2 = 1 ↔ q.1 = E a 2) := by
    simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hc20, hc21, hc23⟩
  have hacc2 : a (.wire 36 15) = (([(E a 0, (a (.wire 11 7)), a (.wire 38 11)), (E a 1,
      (a (.wire 11 47)), a (.wire 38 55))] : List (Cand p)).map
      (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum + bselect (a (.wire 40 39))
      (a (.wire 12 27)) 0 + (([(E a 3, (a (.wire 13 7)), a (.wire 42 23))]
      : List (Cand p)).map (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum := by
    rw [ma23, ma22, ma21, ms20, ms21, ms22, ms23, kzero]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    ring
  have hbound2 : (a (.wire 12 27)).val + (([(E a 3,
      (a (.wire 13 7)), a (.wire 42 23))] : List (Cand p)).map
      (fun q : Cand p => q.2.1.val)).sum < p := by
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    omega
  have hseen2 : ∀ x, x ∈ ([valDigest (E a 1), valDigest (E a 0)] : List Digest) ↔ x
      ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 38 11)), (E a 1,
      (a (.wire 11 47)), a (.wire 38 55))] : List (Cand p)).map
      (fun q : Cand p => valDigest q.1) := by
    intro x; simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false]
    exact or_comm
  have hx2 : (⟨(a (.wire 41 59)).val, ⟨(a (.wire 43 7)).val, (a (.wire 43 15)).val,
      (a (.wire 43 23)).val, (a (.wire 43 31)).val⟩⟩ : ExitSlot)
      = if valDigest (E a 2) ∈ ([valDigest (E a 1), valDigest (E a 0)] : List Digest)
          then ⟨0, Digest.zero⟩
        else ⟨(a (.wire 12 27)).val + matchSum (valDigest (E a 2)) [(valDigest (E a 3),
            (a (.wire 13 7)).val)], valDigest (E a 2)⟩ := by
    have := exit_slot_val (dup := a (.wire 36 3)) (acc := a (.wire 36 15)) (E a 2)
        (a (.wire 12 27)) (a (.wire 40 39)) [(E a 0, (a (.wire 11 7)), a (.wire 38 11)),
        (E a 1, (a (.wire 11 47)), a (.wire 38 55))] [(E a 3,
        (a (.wire 13 7)), a (.wire 42 23))] hdup2 hdupP2 heq2 hc22 hacc2 hbound2 hseen2
    simp only [List.map_cons, List.map_nil] at this
    rw [os2, oe2_0, oe2_1, oe2_2, oe2_3, kzero]
    exact this
  -- exit slot 3
  have hc30 := digestEq_fold4 (x := E a 0) (y := E a 3)
      mq30_0 mq30_1 mq30_2 mq30_3 mqa30 mqb30 mqc30
  have hc31 := digestEq_fold4 (x := E a 1) (y := E a 3)
      mq31_0 mq31_1 mq31_2 mq31_3 mqa31 mqb31 mqc31
  have hc32 := digestEq_fold4 (x := E a 2) (y := E a 3)
      mq32_0 mq32_1 mq32_2 mq32_3 mqa32 mqb32 mqc32
  have hc33 := digestEq_fold4 (x := E a 3) (y := E a 3)
      mq33_0 mq33_1 mq33_2 mq33_3 mqa33 mqb33 mqc33
  have hdq30 := digestEq_fold4 (x := E a 0) (y := E a 3)
      dq30_0 dq30_1 dq30_2 dq30_3 dqa30 dqb30 dqc30
  have hdq31 := digestEq_fold4 (x := E a 1) (y := E a 3)
      dq31_0 dq31_1 dq31_2 dq31_3 dqa31 dqb31 dqc31
  have hdq32 := digestEq_fold4 (x := E a 2) (y := E a 3)
      dq32_0 dq32_1 dq32_2 dq32_3 dqa32 dqb32 dqc32
  have hb31 : IsBool (a (.wire 36 19)) := by rw [do31]; exact Plonky2Spec.bor_isBool hdq30.1 hdq31.1
  have hdup3 : IsBool (a (.wire 36 23)) := by rw [do32]; exact Plonky2Spec.bor_isBool hb31 hdq32.1
  have hdupP3 : a (.wire 36 23) = 1 ↔ ∃ q ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 50 19)),
      (E a 1, (a (.wire 11 47)), a (.wire 52 3)), (E a 2,
      (a (.wire 12 27)), a (.wire 52 47))] : List (Cand p)), q.1 = E a 3 := by
    rw [do32, bor_eq_one hb31 hdq32.1, do31, bor_eq_one hdq30.1 hdq31.1, hdq30.2, hdq31.2, hdq32.2]
    simp [or_assoc]
  have heq3 : ∀ q ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 50 19)), (E a 1,
      (a (.wire 11 47)), a (.wire 52 3)), (E a 2,
      (a (.wire 12 27)), a (.wire 52 47))] : List (Cand p)) ++ [], IsBool q.2.2 ∧ (q.2.2 = 1 ↔
      q.1 = E a 3) := by
    simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq_or_imp, forall_eq]
    exact ⟨hc30, hc31, hc32⟩
  have hacc3 : a (.wire 36 35) = (([(E a 0, (a (.wire 11 7)), a (.wire 50 19)), (E a 1,
      (a (.wire 11 47)), a (.wire 52 3)), (E a 2,
      (a (.wire 12 27)), a (.wire 52 47))] : List (Cand p)).map
      (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum + bselect (a (.wire 54 31))
      (a (.wire 13 7)) 0 + (([] : List (Cand p)).map
      (fun q : Cand p => bselect q.2.2 q.2.1 0)).sum := by
    rw [ma33, ma32, ma31, ms30, ms31, ms32, ms33, kzero]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    ring
  have hbound3 : (a (.wire 13 7)).val + (([] : List (Cand p)).map
      (fun q : Cand p => q.2.1.val)).sum < p := by
    simp only [List.map_nil, List.sum_nil]
    omega
  have hseen3 : ∀ x, x ∈ ([valDigest (E a 2), valDigest (E a 1),
      valDigest (E a 0)] : List Digest) ↔ x ∈ ([(E a 0, (a (.wire 11 7)), a (.wire 50 19)),
      (E a 1, (a (.wire 11 47)), a (.wire 52 3)), (E a 2,
      (a (.wire 12 27)), a (.wire 52 47))] : List (Cand p)).map
      (fun q : Cand p => valDigest q.1) := by
    intro x; simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false]
    constructor <;> rintro (h | h | h) <;> simp only [h, true_or, or_true]
  have hx3 : (⟨(a (.wire 53 43)).val, ⟨(a (.wire 53 51)).val, (a (.wire 53 59)).val,
      (a (.wire 55 7)).val, (a (.wire 55 15)).val⟩⟩ : ExitSlot)
      = if valDigest (E a 3) ∈ ([valDigest (E a 2), valDigest (E a 1),
          valDigest (E a 0)] : List Digest) then ⟨0, Digest.zero⟩
        else ⟨(a (.wire 13 7)).val + matchSum (valDigest (E a 3)) [], valDigest (E a 3)⟩ := by
    have := exit_slot_val (dup := a (.wire 36 23)) (acc := a (.wire 36 35)) (E a 3)
        (a (.wire 13 7)) (a (.wire 54 31)) [(E a 0, (a (.wire 11 7)), a (.wire 50 19)),
        (E a 1, (a (.wire 11 47)), a (.wire 52 3)), (E a 2, (a (.wire 12 27)), a (.wire 52 47))]
        [] hdup3 hdupP3 heq3 hc33 hacc3 hbound3 hseen3
    simp only [List.map_nil] at this
    rw [os3, oe3_0, oe3_1, oe3_2, oe3_3, kzero]
    exact this
  -- the first-real scan
  have hr0 : a (.wire 3 7) = bnot (a (.wire 1 43)) := r0
  have htk : a (.wire 2 31) = band (bnot (a (.wire 2 27))) (bnot (bnot (a (.wire 1 43)))) := by
    rw [tk1, r1, nf, hr0]
  refine
    { hsw := hsw, hb := hb, hd := hd, hdnull := hdnull, hreal := hreal, hnull := hnull
      hcol := hcol, hlen := hlen, h32 := by rw [rows_leaves]; exact h32
      hfee := rfl
      hin := ?_, hout := ?_, hfc := ?_, hexits := ?_, hmeta := ?_, href := ?_, hnum := hnum }
  · -- dummy-masked input total
    show (a (.wire 6 23)).val = maskedInputTotal (leaves a)
    rw [ti, mi0, mi1, kzero]
    have hm0 := val_mask (x := a (.virt 9484)) hb0
    have hm1 := val_mask (x := a (.virt 18969)) hb1
    rw [ZMod.val_add_of_lt (by rw [hm0, hm1]; split <;> split <;> omega), hm0, hm1]
    simp only [hd0, hd1, leaves, maskedInputTotal, 
      add_zero]
    rfl
  · -- dummy-masked output total
    show (a (.wire 6 35)).val = maskedOutputTotal (leaves a)
    rw [to3, to2, to1]
    rw [ZMod.val_add_of_lt (lt_of_le_of_lt (Nat.add_le_add_right (ZMod.val_add_le _ _) _)
        (lt_of_le_of_lt (Nat.add_le_add_right (Nat.add_le_add_right (ZMod.val_add_le _ _) _) _)
          (by omega))),
      ZMod.val_add_of_lt (lt_of_le_of_lt (Nat.add_le_add_right (ZMod.val_add_le _ _) _) (by omega)),
      ZMod.val_add_of_lt (by omega)]
    rw [mA0, mA1, mA2, mA3, kzero, Nat.add_assoc ((bselect _ _ _).val + _), val_mask2 hb0,
      val_mask2 hb1]
    simp only [hd0, hd1, leaves, maskedOutputTotal, 
      add_zero]
    rfl
  · -- the fee comparator
    refine ⟨?_, ?_⟩
    · show rangeCheck (feeDen - a (.wire 4 27)) 14
      unfold feeDen
      rw [Nat.cast_ofNat, ← kten, ← fc]
      exact fcr
    · show rangeCheck (a (.wire 6 23) * (feeDen - a (.wire 4 27)) - a (.wire 6 35) * feeDen) 52
      unfold feeDen
      rw [Nat.cast_ofNat, ← kten, ← fc, ← rhs, ← lhs, ← dif]
      exact dr
  · -- exit grouping
    show [(⟨(a (.wire 22 7)).val, ⟨(a (.wire 22 15)).val, (a (.wire 22 23)).val,
        (a (.wire 22 31)).val,
        (a (.wire 22 39)).val⟩⟩ : ExitSlot),
      ⟨(a (.wire 30 47)).val, ⟨(a (.wire 30 55)).val, (a (.wire 31 3)).val, (a (.wire 31 11)).val,
        (a (.wire 31 19)).val⟩⟩,
      ⟨(a (.wire 41 59)).val, ⟨(a (.wire 43 7)).val, (a (.wire 43 15)).val, (a (.wire 43 23)).val,
        (a (.wire 43 31)).val⟩⟩,
      ⟨(a (.wire 53 43)).val, ⟨(a (.wire 53 51)).val, (a (.wire 53 59)).val, (a (.wire 55 7)).val,
        (a (.wire 55 15)).val⟩⟩] = groupExits (maskedChildPairs (leaves a))
    rw [hpairs, hx0, hx1, hx2, hx3]
    rfl
  · -- metadata consistency
    intro q hq hreal_q
    rw [rows_leaves] at hq
    simp only [leaves, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · have hnd : a (.wire 1 43) ≠ 1 := fun h1 => hreal_q (hd0.mp h1)
      refine ⟨rfl, ?_, ?_⟩
      · have := real_block_matches hb0 (isEqual_isBool cf0) (cfo0.symm.trans (cfk0.trans kone))
          (isEqual_iff cf0) hnd
        show (a (.virt 9466)).val = (a (.wire 4 27)).val
        rw [this]
      · obtain ⟨hmb, hmeq⟩ := digestEq_fold4
          (x := ![a (.virt 9479), a (.virt 9480), a (.virt 9481), a (.virt 9482)])
          (y := ![a (.wire 3 47), a (.wire 3 55), a (.wire 4 3), a (.wire 4 11)])
          cb0_0 cb0_1 cb0_2 cb0_3 cba0 cbb0 cbc0
        have := real_block_matches hb0 hmb (cbo0.symm.trans (cbk0.trans kone)) hmeq hnd
        show valDigest ![a (.virt 9479), a (.virt 9480), a (.virt 9481), a (.virt 9482)]
          = valDigest ![a (.wire 3 47), a (.wire 3 55), a (.wire 4 3), a (.wire 4 11)]
        rw [this]
    · have hnd : a (.wire 2 27) ≠ 1 := fun h1 => hreal_q (hd1.mp h1)
      refine ⟨?_, ?_, ?_⟩
      · show (a (leafPis 1 0)).val = (a (leafPis 0 0)).val
        rw [show a (leafPis 1 0) = a (leafPis 0 0) from ca1]
      · have := real_block_matches hb1 (isEqual_isBool cf1) (cfo1.symm.trans (cfk1.trans kone))
          (isEqual_iff cf1) hnd
        show (a (.virt 18951)).val = (a (.wire 4 27)).val
        rw [this]
      · obtain ⟨hmb, hmeq⟩ := digestEq_fold4
          (x := ![a (.virt 18964), a (.virt 18965), a (.virt 18966), a (.virt 18967)])
          (y := ![a (.wire 3 47), a (.wire 3 55), a (.wire 4 3), a (.wire 4 11)])
          cb1_0 cb1_1 cb1_2 cb1_3 cba1 cbb1 cbc1
        have := real_block_matches hb1 hmb (cbo1.symm.trans (cbk1.trans kone)) hmeq hnd
        show valDigest ![a (.virt 18964), a (.virt 18965), a (.virt 18966), a (.virt 18967)]
          = valDigest ![a (.wire 3 47), a (.wire 3 55), a (.wire 4 3), a (.wire 4 11)]
        rw [this]
  · -- the first-real reference
    rw [rows_leaves]
    unfold referenceFromFirstReal
    rcases hb0 with h0 | h0
    · -- slot 0 is real: the scan takes slot 0
      have hnd0 : ¬ isDummyPrivateBatch (leaf a 0) :=
          fun hh => absurd (hd0.mpr hh) (by rw [h0]; exact zero_ne_one)
      have hfind : (leaves a).find? isRealB = some (leaf a 0) :=
        List.find?_cons_of_pos (WormholeSpec.isRealB_true_iff.mpr hnd0)
      rw [hfind]
      have htk0 : a (.wire 2 31) = 0 := by rw [htk, h0]; simp [bnot, band]
      have hr0' : a (.wire 3 7) = 1 := by rw [hr0, h0]; simp [bnot]
      refine ⟨?_, ?_, rfl, ?_⟩
      · show (⟨(a (.wire 3 47)).val, (a (.wire 3 55)).val, (a (.wire 4 3)).val,
          (a (.wire 4 11)).val⟩ : Digest)
          = ⟨(a (leafPis 0 16)).val, (a (leafPis 0 17)).val, (a (leafPis 0 18)).val, (a (leafPis 0 19)).val⟩
        rw [sc1_0, sc1_1, sc1_2, sc1_3, bselect_false htk0, bselect_false htk0, bselect_false htk0,
          bselect_false htk0, sc0_0, sc0_1, sc0_2, sc0_3, bselect_true hr0', bselect_true hr0',
          bselect_true hr0', bselect_true hr0']
        rfl
      · show (a (.wire 4 19)).val = (a (leafPis 0 20)).val
        rw [sc1_4, bselect_false htk0, sc0_4, bselect_true hr0']
        rfl
      · show (a (.wire 4 27)).val = (a (leafPis 0 3)).val
        rw [sc1_5, bselect_false htk0, sc0_5, bselect_true hr0']
        rfl
    · -- slot 0 is a dummy
      have hdm0 : isDummyPrivateBatch (leaf a 0) := hd0.mp h0
      have hr0' : a (.wire 3 7) = 0 := by rw [hr0, h0]; simp [bnot]
      have hfind0 : (leaves a).find? isRealB = [leaf a 1].find? isRealB :=
        List.find?_cons_of_neg (by rw [WormholeSpec.isRealB_true_iff]; exact not_not.mpr hdm0)
      rw [hfind0]
      rcases hb1 with h1 | h1
      · -- slot 1 is real: the scan takes slot 1
        have hnd1 : ¬ isDummyPrivateBatch (leaf a 1) :=
            fun hh => absurd (hd1.mpr hh) (by rw [h1]; exact zero_ne_one)
        rw [List.find?_cons_of_pos (WormholeSpec.isRealB_true_iff.mpr hnd1)]
        have htk1' : a (.wire 2 31) = 1 := by rw [htk, h0, h1]; simp [bnot, band]
        refine ⟨?_, ?_, ?_, ?_⟩
        · show (⟨(a (.wire 3 47)).val, (a (.wire 3 55)).val, (a (.wire 4 3)).val,
            (a (.wire 4 11)).val⟩ : Digest)
            = ⟨(a (leafPis 1 16)).val, (a (leafPis 1 17)).val, (a (leafPis 1 18)).val, (a (leafPis 1 19)).val⟩
          rw [sc1_0, sc1_1, sc1_2, sc1_3, bselect_true htk1', bselect_true htk1', bselect_true htk1',
            bselect_true htk1']
          rfl
        · show (a (.wire 4 19)).val = (a (leafPis 1 20)).val
          rw [sc1_4, bselect_true htk1']
          rfl
        · show (a (leafPis 0 0)).val = (a (leafPis 1 0)).val
          rw [show a (leafPis 1 0) = a (leafPis 0 0) from ca1]
        · show (a (.wire 4 27)).val = (a (leafPis 1 3)).val
          rw [sc1_5, bselect_true htk1']
          rfl
      · -- both dummies: the scan keeps its zero initial values
        have hdm1 : isDummyPrivateBatch (leaf a 1) := hd1.mp h1
        rw [List.find?_cons_of_neg (by rw [WormholeSpec.isRealB_true_iff]; exact not_not.mpr hdm1),
          List.find?_nil]
        have htk0 : a (.wire 2 31) = 0 := by rw [htk, h0, h1]; simp [bnot, band]
        refine ⟨?_, ?_, ?_⟩
        · show (⟨(a (.wire 3 47)).val, (a (.wire 3 55)).val, (a (.wire 4 3)).val,
            (a (.wire 4 11)).val⟩ : Digest)
            = Digest.zero
          rw [sc1_0, sc1_1, sc1_2, sc1_3, bselect_false htk0, bselect_false htk0, bselect_false htk0,
            bselect_false htk0, sc0_0, sc0_1, sc0_2, sc0_3, bselect_false hr0', bselect_false hr0',
            bselect_false hr0', bselect_false hr0', kzero]
          simp [Digest.zero]
        · show (a (.wire 4 19)).val = 0
          rw [sc1_4, bselect_false htk0, sc0_4, bselect_false hr0', kzero, ZMod.val_zero]
        · show (a (.wire 4 27)).val = 0
          rw [sc1_5, bselect_false htk0, sc0_5, bselect_false hr0', kzero, ZMod.val_zero]

/-- **Soundness of the recorded wrapper.** A satisfying assignment of the `n = 2` private-batch
    wrapper — with the `Poseidon2Gate` rows computing `perm` and the children's leaf-level
    32-bit ranges — decodes to an `RPrivateBatch` instance over the realized sponge oracle. -/
theorem sound (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a)
    (h32 : ∀ q ∈ leaves a, inRange 32 q.inputAmount ∧ inRange 32 q.outputAmount1 ∧
      inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps) :
    RPrivateBatch (spongeRO perm) (leaves a) (us a) (out a) := by
  have := (constraints perm hpg a h hp h32).sound hpg
  rwa [rows_leaves, rows_us] at this

end Plonky2Bridge.Wrapper2

namespace Plonky2Bridge

open Plonky2Spec.Wiring
open Plonky2Spec.Generated (privateBatchWrapper2)
open Plonky2Spec.Poseidon2 (St)
open WormholeSpec (RPrivateBatch maskedOutputTotal realLeaves realNullifiers rawOutputTotal
  outputExitTotal RPrivateBatch_value_conservation RPrivateBatch_settles_distinct_spends
  LeafWitness Rleaf LeafProofAccepted leaf_proof_sound)

variable {p : ℕ} [Fact p.Prime]

/-- **The capstone on the recorded wiring.** `private_batch_end_to_end` with its decode
    hypotheses discharged by `Wrapper2.constraints`: a satisfying assignment of the `n = 2`
    private-batch wrapper the `CircuitBuilder` emitted, whose `Poseidon2Gate` rows compute
    `perm` and whose recursion gadgets accepted the two child leaf proofs, (i) satisfies
    `RPrivateBatch` on the decoded children/preimages/output, (ii) conserves value,
    (iii) settles only pairwise-distinct spends and (iv) attests every child's `Rleaf`. The
    children's 32-bit ranges come from `Rleaf` through `leaf_proof_sound`, the one trusted
    axiom; the wiring model and the exporter carry fidelity to the Rust. -/
theorem private_batch_end_to_end_wired (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (a : Assignment p) (h : Satisfies (privateBatchWrapper2 p) a)
    (hp : Poseidon2Rows perm (privateBatchWrapper2 p) a)
    (hacc : ∀ pub ∈ Wrapper2.leaves a, LeafProofAccepted (spongeRO perm) pub) :
    RPrivateBatch (spongeRO perm) (Wrapper2.leaves a) (Wrapper2.us a) (Wrapper2.out a)
      ∧ outputExitTotal (Wrapper2.out a) = maskedOutputTotal (Wrapper2.leaves a)
      ∧ (outputExitTotal (Wrapper2.out a) = rawOutputTotal (realLeaves (Wrapper2.leaves a))
          ∧ (realNullifiers (Wrapper2.leaves a)).Nodup)
      ∧ ∀ pub ∈ Wrapper2.leaves a, ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w := by
  have hleaf : ∀ pub ∈ Wrapper2.leaves a, ∃ w : LeafWitness, Rleaf (spongeRO perm) pub w :=
    fun pub hq => leaf_proof_sound (spongeRO perm) pub (hacc pub hq)
  have hR := Wrapper2.sound perm hpg a h hp fun q hq =>
    let ⟨_, hw⟩ := hleaf q hq
    Rleaf_ranges hw
  exact ⟨hR, RPrivateBatch_value_conservation hR, RPrivateBatch_settles_distinct_spends hR, hleaf⟩

end Plonky2Bridge
