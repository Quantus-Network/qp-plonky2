/-
  The public-batch wrapper across the `.val` seam (PLAN.md Step 7a).

  `build_public_batch_constraints` (wormhole/aggregator/src/public_batch/circuit/
  circuit_logic.rs) does, per inner private-batch proof:
    * `is_dummy_i = bytes_digest_eq(block_hash_i, 0)`;
    * a first-real prefix scan (`scanStep`) selecting the header (block hash limbs,
      block number, asset id, fee) from the first non-dummy inner, initial value zero;
    * `or(is_dummy_i, is_equal(field_i, ref)) == 1` for asset id, fee and block hash;
    * `select(is_dummy_i, 0, ·)` on every exit-slot limb and nullifier limb, forwarded in
      order;
    * the constant `n_inner · slots_per_inner` as the slot-count header.

  This module lifts each of those through `ZMod.val` into the conjuncts of
  `WormholeSpec.RPublicBatch` (`public_batch_val`) and composes with the trusted
  `private_batch_proof_sound` (`public_batch_end_to_end`). As in the private-batch bridge,
  the *decode* hypotheses (which public-input wire carries which spec value) are the explicit
  boundary; the dummy flag's meaning, the header selection, the consistency checks and the
  forwarding masks are all derived from the gadget lemmas rather than assumed.
-/
import Plonky2Bridge

namespace Plonky2Bridge

open Plonky2Spec (IsBool bselect band bnot bor bnot_isBool bnot_eq_one Digest4 scanStep
  firstRealVal scanFirst_correct real_block_matches)
open WormholeSpec (Digest RandomOracle PrivateBatchOutput PublicBatchOutput RPublicBatch
  isDummyInner isRealInnerB forwardedSlots forwardedNullifiers innerReferenceFromFirstReal
  ExitSlot PrivateBatchProofAccepted private_batch_proof_sound RPrivateBatch
  RPublicBatch_totalExitSlots)

variable {p : ℕ} [Fact p.Prime]

/-! ### Field witness for one inner proof -/

/-- The public-input wires of one inner private-batch proof that the public-batch wrapper
    reads, plus its derived dummy flag. -/
structure InnerRow (p : ℕ) where
  isDummy : ZMod p
  blockHash : Digest4 p
  blockNumber : ZMod p
  assetId : ZMod p
  fee : ZMod p
  slots : List (SlotF p)
  nulls : List (Digest4 p)

/-- One aligned witness row: the field wires and the decoded inner output. -/
abbrev InnerPair (p : ℕ) := InnerRow p × PrivateBatchOutput

/-- The all-zero digest sentinel the dummy check compares against. -/
def zeroDigest4 : Digest4 p := fun _ => 0

theorem valDigest_zeroDigest4 : valDigest (zeroDigest4 (p := p)) = Digest.zero := by
  simp [valDigest, zeroDigest4, Digest.zero]

/-- The dummy flag's field meaning (`bytes_digest_eq(block_hash, 0) = 1 ↔ block_hash = 0`,
    from `bytesDigestEq_spec`) decodes to the spec's `isDummyInner`. -/
theorem innerDummy_val {t : InnerPair p}
    (hflag : t.1.isDummy = 1 ↔ t.1.blockHash = zeroDigest4)
    (hdec : valDigest t.1.blockHash = t.2.blockHash) :
    t.1.isDummy = 1 ↔ isDummyInner t.2 := by
  rw [hflag]
  show t.1.blockHash = zeroDigest4 ↔ t.2.blockHash = Digest.zero
  rw [← hdec, ← valDigest_zeroDigest4 (p := p)]
  exact ⟨fun h => h ▸ rfl, fun h => valDigest_injective h⟩

theorem isRealInnerB_eq_true_iff {o : PrivateBatchOutput} :
    isRealInnerB o = true ↔ ¬ isDummyInner o := by
  simp [isRealInnerB]

/-- `bnot is_dummy = 1` is exactly the spec's real-inner test. -/
theorem real_flag_iff {t : InnerPair p} (hb : IsBool t.1.isDummy)
    (hd : t.1.isDummy = 1 ↔ isDummyInner t.2) :
    bnot t.1.isDummy = 1 ↔ isRealInnerB t.2 = true := by
  rw [bnot_eq_one, isRealInnerB_eq_true_iff, ← hd]
  rcases hb with h0 | h1
  · rw [h0]; simp
  · rw [h1]; simp

/-! ### The header scan -/

/-- The reference the first-real scan produces for a per-inner field `f`, starting from the
    zero initial reference the circuit uses. -/
def scanRef (rows : List (InnerPair p)) (f : InnerPair p → ZMod p) : ZMod p :=
  (List.foldl scanStep (0, 0) (rows.map fun t => (bnot t.1.isDummy, f t))).2

/-- The scanned block-hash reference, limb by limb. -/
def blockRef (rows : List (InnerPair p)) : Digest4 p :=
  fun j => scanRef rows (fun t => t.1.blockHash j)

theorem firstRealVal_find? (f : InnerPair p → ZMod p) (init : ZMod p) :
    ∀ rows : List (InnerPair p), (∀ t ∈ rows, IsBool t.1.isDummy) →
    (∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2) →
    firstRealVal init (rows.map fun t => (bnot t.1.isDummy, f t))
      = match rows.find? (fun t => isRealInnerB t.2) with
        | some t => f t
        | none => init := by
  intro rows
  induction rows with
  | nil => intros; rfl
  | cons t rest ih =>
      intro hb hd
      have ht : bnot t.1.isDummy = 1 ↔ isRealInnerB t.2 = true :=
        real_flag_iff (hb t List.mem_cons_self) (hd t List.mem_cons_self)
      have ih' := ih (fun q hq => hb q (List.mem_cons_of_mem _ hq))
        (fun q hq => hd q (List.mem_cons_of_mem _ hq))
      simp only [List.map_cons, firstRealVal, List.find?_cons]
      by_cases hr : isRealInnerB t.2 = true
      · rw [if_pos (ht.mpr hr), hr]
      · rw [if_neg (fun h => hr (ht.mp h)), ih']
        cases hr' : isRealInnerB t.2
        · rfl
        · exact absurd hr' hr

/-- `scanRef` is the first real inner's field value (`scanFirst_correct` + `firstRealVal_find?`),
    or zero when every inner is dummy. -/
theorem scanRef_eq (rows : List (InnerPair p)) (f : InnerPair p → ZMod p)
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2) :
    scanRef rows f
      = match rows.find? (fun t => isRealInnerB t.2) with
        | some t => f t
        | none => 0 := by
  unfold scanRef
  rw [scanFirst_correct _ _ (by
    intro rv hrv
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hrv
    exact bnot_isBool (hb t ht))]
  exact firstRealVal_find? f 0 rows hb hd

/-- **Header bridge.** The four scanned references, read through `.val`, are the spec's
    `innerReferenceFromFirstReal` (including the all-dummy zero case). -/
theorem innerReference_val_bridge (rows : List (InnerPair p)) {out : PublicBatchOutput}
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2)
    (hdecBlock : ∀ t ∈ rows, valDigest t.1.blockHash = t.2.blockHash)
    (hdecNum : ∀ t ∈ rows, t.1.blockNumber.val = t.2.blockNumber)
    (hdecAsset : ∀ t ∈ rows, t.1.assetId.val = t.2.assetId)
    (hdecFee : ∀ t ∈ rows, t.1.fee.val = t.2.volumeFeeBps)
    (hblock : out.blockHash = valDigest (blockRef rows))
    (hnum : out.blockNumber = (scanRef rows fun t => t.1.blockNumber).val)
    (hasset : out.assetId = (scanRef rows fun t => t.1.assetId).val)
    (hfee : out.volumeFeeBps = (scanRef rows fun t => t.1.fee).val) :
    innerReferenceFromFirstReal (rows.map Prod.snd) out := by
  unfold innerReferenceFromFirstReal
  have hfm : (rows.map Prod.snd).find? isRealInnerB
      = (rows.find? fun t => isRealInnerB t.2).map Prod.snd := List.find?_map
  rw [hfm]
  have hblockRef : blockRef rows
      = match rows.find? (fun t => isRealInnerB t.2) with
        | some t => t.1.blockHash
        | none => zeroDigest4 := by
    funext j
    unfold blockRef
    rw [scanRef_eq rows _ hb hd]
    cases rows.find? (fun t => isRealInnerB t.2) <;> rfl
  rw [hblock, hnum, hasset, hfee, hblockRef, scanRef_eq rows _ hb hd, scanRef_eq rows _ hb hd,
    scanRef_eq rows _ hb hd]
  cases hf : rows.find? (fun t => isRealInnerB t.2) with
  | none =>
      simp only [Option.map_none]
      exact ⟨valDigest_zeroDigest4, ZMod.val_zero, ZMod.val_zero, ZMod.val_zero⟩
  | some t =>
      have ht : t ∈ rows := List.mem_of_find?_eq_some hf
      simp only [Option.map_some]
      exact ⟨hdecBlock t ht, hdecNum t ht, hdecAsset t ht, hdecFee t ht⟩

/-! ### Metadata consistency -/

/-- One satisfied `or(is_dummy, is_equal(x, ref)) == 1` constraint, with its `is_equal`
    flag correct. -/
def ConsistencyCheck (is_dummy x ref : ZMod p) : Prop :=
  ∃ m : ZMod p, IsBool m ∧ bor is_dummy m = 1 ∧ (m = 1 ↔ x = ref)

theorem consistency_val {is_dummy x ref : ZMod p} (hb : IsBool is_dummy)
    (hnd : is_dummy ≠ 1) (h : ConsistencyCheck is_dummy x ref) : x.val = ref.val := by
  obtain ⟨m, hm, hcons, hmEq⟩ := h
  exact congrArg ZMod.val (real_block_matches hb hm hcons hmEq hnd)

/-- The digest-valued consistency check, one `bytes_digest_eq` flag for all four limbs. -/
def DigestConsistencyCheck (is_dummy : ZMod p) (x ref : Digest4 p) : Prop :=
  ∃ m : ZMod p, IsBool m ∧ bor is_dummy m = 1 ∧ (m = 1 ↔ x = ref)

theorem digestConsistency_val {is_dummy : ZMod p} {x ref : Digest4 p} (hb : IsBool is_dummy)
    (hnd : is_dummy ≠ 1) (h : DigestConsistencyCheck is_dummy x ref) :
    valDigest x = valDigest ref := by
  obtain ⟨m, hm, hcons, hmEq⟩ := h
  exact congrArg valDigest (real_block_matches hb hm hcons hmEq hnd)

/-- **Metadata bridge.** Every non-dummy inner agrees with the scanned header. -/
theorem innerMetadata_val_bridge (rows : List (InnerPair p)) {out : PublicBatchOutput}
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2)
    (hdecBlock : ∀ t ∈ rows, valDigest t.1.blockHash = t.2.blockHash)
    (hdecAsset : ∀ t ∈ rows, t.1.assetId.val = t.2.assetId)
    (hdecFee : ∀ t ∈ rows, t.1.fee.val = t.2.volumeFeeBps)
    (hblock : out.blockHash = valDigest (blockRef rows))
    (hasset : out.assetId = (scanRef rows fun t => t.1.assetId).val)
    (hfee : out.volumeFeeBps = (scanRef rows fun t => t.1.fee).val)
    (hcAsset : ∀ t ∈ rows, ConsistencyCheck t.1.isDummy t.1.assetId (scanRef rows fun t => t.1.assetId))
    (hcFee : ∀ t ∈ rows, ConsistencyCheck t.1.isDummy t.1.fee (scanRef rows fun t => t.1.fee))
    (hcBlock : ∀ t ∈ rows, DigestConsistencyCheck t.1.isDummy t.1.blockHash (blockRef rows)) :
    ∀ o ∈ rows.map Prod.snd, ¬ isDummyInner o →
      o.assetId = out.assetId ∧ o.volumeFeeBps = out.volumeFeeBps ∧ o.blockHash = out.blockHash := by
  intro o ho hreal
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 ho
  have hnd : t.1.isDummy ≠ 1 := fun h1 => hreal ((hd t ht).mp h1)
  refine ⟨?_, ?_, ?_⟩
  · rw [hasset, ← hdecAsset t ht]; exact consistency_val (hb t ht) hnd (hcAsset t ht)
  · rw [hfee, ← hdecFee t ht]; exact consistency_val (hb t ht) hnd (hcFee t ht)
  · rw [hblock, ← hdecBlock t ht]; exact digestConsistency_val (hb t ht) hnd (hcBlock t ht)

/-! ### Forwarding -/

/-- The masked slot region the wrapper emits, in order. -/
def forwardedSlotsF (rows : List (InnerPair p)) : List (SlotF p) :=
  (rows.map fun t => t.1.slots.map (maskSlot t.1.isDummy)).flatten

/-- The masked nullifier region the wrapper emits, in order. -/
def forwardedNullsF (rows : List (InnerPair p)) : List (Digest4 p) :=
  (rows.map fun t => t.1.nulls.map fun x => fun j => bselect t.1.isDummy 0 (x j)).flatten

theorem forwardedSlotsF_val (rows : List (InnerPair p))
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2)
    (hdec : ∀ t ∈ rows, t.1.slots.map valSlot = t.2.exitSlots) :
    (forwardedSlotsF rows).map valSlot = ((rows.map Prod.snd).map forwardedSlots).flatten := by
  unfold forwardedSlotsF
  rw [List.map_flatten, List.map_map, List.map_map]
  congr 1
  apply List.map_congr_left
  intro t ht
  exact forwardedSlots_val_bridge _ (hb t ht) (hd t ht) (hdec t ht)

theorem forwardedNullsF_val (rows : List (InnerPair p))
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2)
    (hdec : ∀ t ∈ rows, t.1.nulls.map valDigest = t.2.nullifiers) :
    (forwardedNullsF rows).map valDigest
      = ((rows.map Prod.snd).map forwardedNullifiers).flatten := by
  unfold forwardedNullsF
  rw [List.map_flatten, List.map_map, List.map_map]
  congr 1
  apply List.map_congr_left
  intro t ht
  exact forwardedNullifiers_val_bridge _ (hb t ht) (hd t ht) (hdec t ht)

/-- The slot-count header: `n_inner · slots_per_inner` is the forwarded region's length
    once every inner has the shape-checked `slots_per_inner` slots. -/
theorem forwardedSlotsF_length (rows : List (InnerPair p)) {k : ℕ}
    (hshape : ∀ t ∈ rows, t.1.slots.length = k) :
    (forwardedSlotsF rows).length = rows.length * k := by
  unfold forwardedSlotsF
  rw [List.length_flatten, List.map_map]
  have : (rows.map ((fun l : List (SlotF p) => l.length) ∘ fun t => t.1.slots.map (maskSlot t.1.isDummy)))
      = rows.map fun _ => k := by
    apply List.map_congr_left
    intro t ht
    simp only [Function.comp, List.length_map]
    exact hshape t ht
  rw [this, List.map_const', List.sum_replicate, smul_eq_mul]

/-! ### Composition -/

/-- **Wrapper-logic `.val` composition (public batch).** With the per-inner dummy flags
    derived from `bytes_digest_eq` against zero, the header from the first-real scan, the
    per-inner consistency constraints satisfied, and the forwarded regions decoded from the
    masked field regions, the public-batch wrapper satisfies `RPublicBatch`. -/
theorem public_batch_val (ro : RandomOracle) (rows : List (InnerPair p)) {addr : Digest}
    {out : PublicBatchOutput} {k : ℕ}
    (haddr : out.aggregatorAddress = addr)
    -- dummy flags
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hflag : ∀ t ∈ rows, t.1.isDummy = 1 ↔ t.1.blockHash = zeroDigest4)
    -- inner public-input decodes
    (hdecBlock : ∀ t ∈ rows, valDigest t.1.blockHash = t.2.blockHash)
    (hdecNum : ∀ t ∈ rows, t.1.blockNumber.val = t.2.blockNumber)
    (hdecAsset : ∀ t ∈ rows, t.1.assetId.val = t.2.assetId)
    (hdecFee : ∀ t ∈ rows, t.1.fee.val = t.2.volumeFeeBps)
    (hdecSlots : ∀ t ∈ rows, t.1.slots.map valSlot = t.2.exitSlots)
    (hdecNulls : ∀ t ∈ rows, t.1.nulls.map valDigest = t.2.nullifiers)
    -- header outputs are the scan results
    (hblock : out.blockHash = valDigest (blockRef rows))
    (hnum : out.blockNumber = (scanRef rows fun t => t.1.blockNumber).val)
    (hasset : out.assetId = (scanRef rows fun t => t.1.assetId).val)
    (hfee : out.volumeFeeBps = (scanRef rows fun t => t.1.fee).val)
    -- consistency constraints
    (hcAsset : ∀ t ∈ rows, ConsistencyCheck t.1.isDummy t.1.assetId (scanRef rows fun t => t.1.assetId))
    (hcFee : ∀ t ∈ rows, ConsistencyCheck t.1.isDummy t.1.fee (scanRef rows fun t => t.1.fee))
    (hcBlock : ∀ t ∈ rows, DigestConsistencyCheck t.1.isDummy t.1.blockHash (blockRef rows))
    -- forwarded regions and the slot-count constant
    (hexits : out.exitSlots = (forwardedSlotsF rows).map valSlot)
    (hnulls : out.nullifiers = (forwardedNullsF rows).map valDigest)
    (hshape : ∀ t ∈ rows, t.1.slots.length = k)
    (htot : out.totalExitSlots = rows.length * k) :
    RPublicBatch ro (rows.map Prod.snd) addr out := by
  have hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyInner t.2 :=
    fun t ht => innerDummy_val (hflag t ht) (hdecBlock t ht)
  refine ⟨haddr,
    innerReference_val_bridge rows hb hd hdecBlock hdecNum hdecAsset hdecFee hblock hnum hasset hfee,
    innerMetadata_val_bridge rows hb hd hdecBlock hdecAsset hdecFee hblock hasset hfee
      hcAsset hcFee hcBlock,
    ?_, ?_, ?_⟩
  · rw [hexits]; exact forwardedSlotsF_val rows hb hd hdecSlots
  · rw [hnulls]; exact forwardedNullsF_val rows hb hd hdecNulls
  · rw [htot, hexits, List.length_map]; exact (forwardedSlotsF_length rows hshape).symm

/-- **End-to-end public-batch soundness.** A satisfied public-batch wrapper whose recursion
    gadget accepted every inner private-batch proof (i) satisfies `RPublicBatch`, (ii) has the
    slot-count header equal to the sum of the inners' slot counts, and (iii) attests every
    inner's `RPrivateBatch` for some children — the latter through the trusted
    `private_batch_proof_sound`, the only axiom this theorem adds. -/
theorem public_batch_end_to_end (ro : RandomOracle) (rows : List (InnerPair p)) {addr : Digest}
    {out : PublicBatchOutput} {k : ℕ}
    (haddr : out.aggregatorAddress = addr)
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hflag : ∀ t ∈ rows, t.1.isDummy = 1 ↔ t.1.blockHash = zeroDigest4)
    (hdecBlock : ∀ t ∈ rows, valDigest t.1.blockHash = t.2.blockHash)
    (hdecNum : ∀ t ∈ rows, t.1.blockNumber.val = t.2.blockNumber)
    (hdecAsset : ∀ t ∈ rows, t.1.assetId.val = t.2.assetId)
    (hdecFee : ∀ t ∈ rows, t.1.fee.val = t.2.volumeFeeBps)
    (hdecSlots : ∀ t ∈ rows, t.1.slots.map valSlot = t.2.exitSlots)
    (hdecNulls : ∀ t ∈ rows, t.1.nulls.map valDigest = t.2.nullifiers)
    (hblock : out.blockHash = valDigest (blockRef rows))
    (hnum : out.blockNumber = (scanRef rows fun t => t.1.blockNumber).val)
    (hasset : out.assetId = (scanRef rows fun t => t.1.assetId).val)
    (hfee : out.volumeFeeBps = (scanRef rows fun t => t.1.fee).val)
    (hcAsset : ∀ t ∈ rows, ConsistencyCheck t.1.isDummy t.1.assetId (scanRef rows fun t => t.1.assetId))
    (hcFee : ∀ t ∈ rows, ConsistencyCheck t.1.isDummy t.1.fee (scanRef rows fun t => t.1.fee))
    (hcBlock : ∀ t ∈ rows, DigestConsistencyCheck t.1.isDummy t.1.blockHash (blockRef rows))
    (hexits : out.exitSlots = (forwardedSlotsF rows).map valSlot)
    (hnulls : out.nullifiers = (forwardedNullsF rows).map valDigest)
    (hshape : ∀ t ∈ rows, t.1.slots.length = k)
    (htot : out.totalExitSlots = rows.length * k)
    (hacc : ∀ o ∈ rows.map Prod.snd, PrivateBatchProofAccepted ro o) :
    RPublicBatch ro (rows.map Prod.snd) addr out
      ∧ out.totalExitSlots = ((rows.map Prod.snd).map fun o => o.exitSlots.length).sum
      ∧ ∀ o ∈ rows.map Prod.snd, ∃ leaves us, RPrivateBatch ro leaves us o := by
  have hR := public_batch_val ro rows haddr hb hflag hdecBlock hdecNum hdecAsset hdecFee hdecSlots
    hdecNulls hblock hnum hasset hfee hcAsset hcFee hcBlock hexits hnulls hshape htot
  exact ⟨hR, RPublicBatch_totalExitSlots hR,
    fun o ho => private_batch_proof_sound ro o (hacc o ho)⟩

end Plonky2Bridge
