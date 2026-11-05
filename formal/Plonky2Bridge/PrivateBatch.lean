/-
  The private-batch wrapper across the `.val` seam, for any number of leaves (PLAN.md
  Step 8e).

  `build_private_batch_constraints` (wormhole/aggregator/src/private_batch/circuit/
  circuit_logic.rs) does, per child leaf proof: the dummy check
  `bytes_digest_eq(block_hash_i, 0)`, a first-real prefix scan selecting the block
  reference from the first non-dummy child, the consistency constraints against the
  scanned reference and the slot-0 asset id, the dummy-masked input total; then, over the
  batch-level witnesses, the 32-bit range check of each exit amount and their total, the
  fee comparator on the witnessed `volume_fee_bps`; and finally the pairwise
  nullifier-uniqueness constraints, the dummy-nullifier double hash and the per-slot
  nullifier selects registered in slot order.

  This module states each of those as a list-shaped hypothesis over `LeafRow`s — the field
  wires of one child together with the `is_equal` witnesses of its dummy check — and the
  exit-slot wires, and lifts them into `PrivateBatchConstraints`
  (`private_batch_constraints_rows`), hence into `RPrivateBatch` (`private_batch_val_rows`). The exporter-generated
  `Plonky2Bridge/Generated/Wrapper{N}.lean` discharges every hypothesis for a recorded
  `n = N` trace by rewriting with the facts of `privateBatchWrapper{N}_decode`.
-/
import Plonky2Bridge.Complete
import Plonky2Bridge.PublicBatch

namespace Plonky2Bridge

open Plonky2Spec (IsBool bselect band bnot bor bnot_isBool bor_isBool Digest4 digestEq
  bytesDigestEq_spec IsEqual scanStep firstRealVal scanFirst_correct real_block_matches
  isEqual_iff isEqual_isBool bselect_true bselect_false FeeCheck bnot_eq_one bor_eq_one
  rangeCheck rangeCheck_sound)
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash)
open WormholeSpec (Digest LeafPublic PrivateBatchOutput ExitSlot isDummyPrivateBatch isRealB
  Felt maskedInputTotal outputExitTotal slotsTotal
  metadataConsistent referenceFromFirstReal inRange RPrivateBatch isRealB_true_iff)

variable {p : ℕ} [Fact p.Prime]

/-! ### Field witness for one child -/

/-- The public-input wires of one child leaf proof the wrapper reads, the `is_equal`
    witnesses (`equal` flag and auxiliary inverse per limb) of its
    `bytes_digest_eq(block_hash, 0)` dummy check, its dummy-nullifier preimage and the
    outer hash output. The dummy flag itself is the gadget's output, `LeafRow.isDummy`. -/
structure LeafRow (p : ℕ) where
  assetId : ZMod p
  nullifier : Digest4 p
  blockHash : Digest4 p
  blockNumber : ZMod p
  inputAmount : ZMod p
  dummyEq : Fin 4 → ZMod p
  dummyInv : Fin 4 → ZMod p
  pre : Fin 4 → ZMod p
  dnull : Digest4 p

/-- `is_dummy = bytes_digest_eq(block_hash, 0)`: the `and`-tree of the per-limb flags. -/
def LeafRow.isDummy (r : LeafRow p) : ZMod p := digestEq r.dummyEq

/-- The nullifier slot: dummy flag, dummy nullifier, real nullifier. -/
def LeafRow.slot (r : LeafRow p) : NullSlot p := ⟨r.isDummy, r.dnull, r.nullifier⟩

/-- The child's public inputs, decoded through `.val`. -/
def LeafRow.leaf (r : LeafRow p) : LeafPublic :=
  { assetId := r.assetId.val, nullifier := valDigest r.nullifier,
    blockHash := valDigest r.blockHash, blockNumber := r.blockNumber.val,
    inputAmount := r.inputAmount.val }

/-- The dummy-nullifier preimage, decoded. -/
def LeafRow.u (r : LeafRow p) : List Felt :=
  [(r.pre 0).val, (r.pre 1).val, (r.pre 2).val, (r.pre 3).val]

/-- The aligned witness row `PrivateBatchConstraints` consumes. -/
def LeafRow.toSlotRow (r : LeafRow p) : SlotRow p := (r.slot, r.leaf, r.u)

/-- `bytes_digest_eq(x, y)` as recorded: per-limb `is_equal` witnesses whose `and`-tree is
    the output `m`. -/
def DigestEqW (x y : Digest4 p) (m : ZMod p) : Prop :=
  ∃ e inv : Fin 4 → ZMod p, (∀ j, IsEqual (x j) (y j) (e j) (inv j)) ∧ m = digestEq e

theorem digestEqW_spec {x y : Digest4 p} {m : ZMod p} (h : DigestEqW x y m) :
    IsBool m ∧ (m = 1 ↔ x = y) := by
  obtain ⟨e, inv, he, rfl⟩ := h
  exact bytesDigestEq_spec he

/-- The constraints `bytes_digest_eq(block_hash, 0)` emits: one `is_equal` per limb against
    the zero sentinel. -/
def DummyCheckL (r : LeafRow p) : Prop :=
  ∀ j, IsEqual (r.blockHash j) 0 (r.dummyEq j) (r.dummyInv j)

theorem valDigest_zero : valDigest (fun _ : Fin 4 => (0 : ZMod p)) = Digest.zero := by
  simp [valDigest, Digest.zero]

/-- The dummy flag, derived from its constraints, is boolean and decodes to the spec's
    `isDummyPrivateBatch` of the decoded child. -/
theorem dummyCheckL_spec {r : LeafRow p} (h : DummyCheckL r) :
    IsBool r.isDummy ∧ (r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) := by
  obtain ⟨hb, hiff⟩ := bytesDigestEq_spec (a := r.blockHash) (c := fun _ => 0) h
  refine ⟨hb, hiff.trans ⟨fun hx => ?_, fun hv => ?_⟩⟩
  · show valDigest r.blockHash = Digest.zero
    rw [hx]; exact valDigest_zero
  · exact valDigest_injective (hv.trans valDigest_zero.symm)

/-- `hash_dummy_nullifier_pre_image` as two recorded `poseidon2_hash` calls: an inner
    digest of the preimage, and the outer hash of the inner digest. -/
def DummyNullCheck (perm : St p → St p) (r : LeafRow p) : Prop :=
  ∃ i : Digest4 p, (∀ j, i j = spongeHash perm [r.pre 0, r.pre 1, r.pre 2, r.pre 3] j) ∧
    (∀ j, r.dnull j = spongeHash perm [i 0, i 1, i 2, i 3] j)

/-- The outer output, read through `.val`, is the realized oracle's `dummyNull` of the
    decoded preimage. -/
theorem dummyNullCheck_val {perm : St p → St p} {r : LeafRow p} (h : DummyNullCheck perm r) :
    valDigest r.dnull = (spongeRO perm).dummyNull r.u := by
  obtain ⟨i, hi, ho⟩ := h
  rw [spongeRO_dummyNull]
  have hin : spongeH perm r.u = ⟨(i 0).val, (i 1).val, (i 2).val, (i 3).val⟩ := by
    simp only [LeafRow.u, spongeH, emb_cons, emb_nil, ZMod.natCast_zmod_val, hi]
  rw [hin]
  simp only [Digest.toList, spongeH, emb_cons, emb_nil, ZMod.natCast_zmod_val, valDigest, ho]

/-! ### The header scan -/

/-- The reference the first-real scan produces for a per-child field `f`, from the zero
    initial reference the circuit uses. -/
def scanRefL (rows : List (LeafRow p)) (f : LeafRow p → ZMod p) : ZMod p :=
  (List.foldl scanStep (0, 0) (rows.map fun r => (bnot r.isDummy, f r))).2

/-- The scanned block-hash reference, limb by limb. -/
def blockRefL (rows : List (LeafRow p)) : Digest4 p :=
  fun j => scanRefL rows (fun r => r.blockHash j)

theorem real_flagL_iff {r : LeafRow p} (hb : IsBool r.isDummy)
    (hd : r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) :
    bnot r.isDummy = 1 ↔ isRealB r.leaf = true := by
  rw [bnot_eq_one, isRealB_true_iff, ← hd]
  rcases hb with h0 | h1
  · rw [h0]; simp
  · rw [h1]; simp

theorem firstRealVal_find?L (f : LeafRow p → ZMod p) (init : ZMod p) :
    ∀ rows : List (LeafRow p), (∀ r ∈ rows, IsBool r.isDummy) →
    (∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) →
    firstRealVal init (rows.map fun r => (bnot r.isDummy, f r))
      = match rows.find? (fun r => isRealB r.leaf) with
        | some r => f r
        | none => init := by
  intro rows
  induction rows with
  | nil => intros; rfl
  | cons r rest ih =>
      intro hb hd
      have hr : bnot r.isDummy = 1 ↔ isRealB r.leaf = true :=
        real_flagL_iff (hb r List.mem_cons_self) (hd r List.mem_cons_self)
      have ih' := ih (fun q hq => hb q (List.mem_cons_of_mem _ hq))
        (fun q hq => hd q (List.mem_cons_of_mem _ hq))
      simp only [List.map_cons, firstRealVal, List.find?_cons]
      by_cases hreal : isRealB r.leaf = true
      · rw [if_pos (hr.mpr hreal), hreal]
      · rw [if_neg (fun h => hreal (hr.mp h)), ih']
        cases hr' : isRealB r.leaf
        · rfl
        · exact absurd hr' hreal

/-- `scanRefL` is the first real child's field value, or zero when every child is dummy. -/
theorem scanRefL_eq (rows : List (LeafRow p)) (f : LeafRow p → ZMod p)
    (hb : ∀ r ∈ rows, IsBool r.isDummy)
    (hd : ∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) :
    scanRefL rows f
      = match rows.find? (fun r => isRealB r.leaf) with
        | some r => f r
        | none => 0 := by
  unfold scanRefL
  rw [scanFirst_correct _ _ (by
    intro rv hrv
    obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hrv
    exact bnot_isBool (hb r hr))]
  exact firstRealVal_find?L f 0 rows hb hd

/-- **Header bridge.** The scanned references, read through `.val`, together with the
    asset id every child is connected to, are the spec's `referenceFromFirstReal`. -/
theorem referenceL_val_bridge (rows : List (LeafRow p)) {out : PrivateBatchOutput}
    {assetRef : ZMod p}
    (hb : ∀ r ∈ rows, IsBool r.isDummy)
    (hd : ∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf)
    (hblock : out.blockHash = valDigest (blockRefL rows))
    (hnum : out.blockNumber = (scanRefL rows fun r => r.blockNumber).val)
    (hasset : out.assetId = assetRef.val)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef) :
    referenceFromFirstReal (rows.map LeafRow.leaf) out := by
  unfold referenceFromFirstReal
  have hfm : (rows.map LeafRow.leaf).find? isRealB
      = (rows.find? fun r => isRealB r.leaf).map LeafRow.leaf := List.find?_map
  rw [hfm]
  have hblockRef : blockRefL rows
      = match rows.find? (fun r => isRealB r.leaf) with
        | some r => r.blockHash
        | none => fun _ => 0 := by
    funext j
    unfold blockRefL
    rw [scanRefL_eq rows _ hb hd]
    cases rows.find? (fun r => isRealB r.leaf) <;> rfl
  rw [hblock, hnum, hblockRef, scanRefL_eq rows _ hb hd]
  cases hf : rows.find? (fun r => isRealB r.leaf) with
  | none =>
      simp only [Option.map_none]
      exact ⟨valDigest_zero, ZMod.val_zero⟩
  | some r =>
      have hr : r ∈ rows := List.mem_of_find?_eq_some hf
      simp only [Option.map_some]
      exact ⟨rfl, rfl, by rw [hasset, ← hcAsset r hr]; rfl⟩

/-- **Metadata bridge.** Every non-dummy child agrees with the header. -/
theorem metadataL_val_bridge (rows : List (LeafRow p)) {out : PrivateBatchOutput}
    {assetRef : ZMod p} {blockRef : Digest4 p}
    (hb : ∀ r ∈ rows, IsBool r.isDummy)
    (hd : ∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef)
    (hcBlock : ∀ r ∈ rows, DigestConsistencyCheck r.isDummy r.blockHash blockRef)
    (hoAsset : out.assetId = assetRef.val)
    (hoBlock : out.blockHash = valDigest blockRef) :
    metadataConsistent (rows.map LeafRow.leaf) out := by
  intro q hq hreal
  obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hq
  have hnd : r.isDummy ≠ 1 := fun h1 => hreal ((hd r hr).mp h1)
  refine ⟨?_, ?_⟩
  · rw [hoAsset, ← hcAsset r hr]; rfl
  · rw [hoBlock]; exact digestConsistency_val (hb r hr) hnd (hcBlock r hr)

/-! ### Dummy-masked totals -/

/-- A field sum of bounded values reads through `.val` as the sum of the values when it
    cannot wrap. -/
theorem val_sum_of_le (B : ℕ) :
    ∀ l : List (ZMod p), (∀ x ∈ l, x.val ≤ B) → l.length * B < p →
      l.sum.val = (l.map ZMod.val).sum ∧ (l.map ZMod.val).sum ≤ l.length * B
  | [], _, _ => by simp
  | x :: rest, hB, hp => by
      simp only [List.length_cons, Nat.succ_mul] at hp
      obtain ⟨ih, ihle⟩ := val_sum_of_le B rest
        (fun y hy => hB y (List.mem_cons_of_mem _ hy)) (by omega)
      have hx := hB x List.mem_cons_self
      simp only [List.sum_cons, List.map_cons, List.length_cons, Nat.succ_mul]
      rw [ZMod.val_add_of_lt (by rw [ih]; omega), ih]
      omega

theorem maskedInputTotal_val :
    ∀ rows : List (LeafRow p), (∀ r ∈ rows, IsBool r.isDummy) →
      (∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) →
      ((rows.map fun r => bselect r.isDummy 0 r.inputAmount).map ZMod.val).sum
        = maskedInputTotal (rows.map LeafRow.leaf)
  | [], _, _ => rfl
  | r :: rest, hb, hd => by
      simp only [List.map_cons, List.sum_cons, maskedInputTotal]
      rw [maskedInputTotal_val rest (fun q hq => hb q (List.mem_cons_of_mem _ hq))
        (fun q hq => hd q (List.mem_cons_of_mem _ hq)),
        val_mask (hb r List.mem_cons_self)]
      by_cases h : r.isDummy = 1
      · rw [if_pos h, if_pos ((hd r List.mem_cons_self).mp h)]
      · rw [if_neg h, if_neg (fun h' => h ((hd r List.mem_cons_self).mpr h'))]; rfl

/-! ### The exit slots

  Every slot `[amount, account(4)]` is a prover-chosen witness forwarded verbatim; the
  circuit only range-checks each amount to 32 bits and sums the amounts for the fee
  comparator. -/

omit [Fact p.Prime] in
/-- The spec's slot total over decoded field slots is the sum of the decoded amounts. -/
theorem slotsTotal_map_valSlot :
    ∀ slots : List (SlotF p), slotsTotal (slots.map valSlot) = ((slots.map Prod.fst).map ZMod.val).sum
  | [] => rfl
  | s :: rest => by
      simp only [List.map_cons, List.sum_cons, slotsTotal, valSlot]
      rw [slotsTotal_map_valSlot rest]

/-! ### Nullifier uniqueness -/

/-- One pairwise constraint: `and(and(not d_i, not d_j), bytes_digest_eq(null_i, null_j))`
    connected to zero. -/
def UniqCheck (x y : Digest4 p) (dx dy : ZMod p) : Prop :=
  ∃ m, DigestEqW x y m ∧ band (band (bnot dx) (bnot dy)) m = 0

/-! ### Composition -/

/-- **Wrapper-logic composition (private batch).** From the recorded gadget constraints —
    stated per child, per exit slot and per pair — the wrapper's field witness satisfies
    `PrivateBatchConstraints` on the decoded rows. Booleanity of the dummy flags and every
    equality flag's meaning are derived from the gadget lemmas, not assumed. -/
theorem private_batch_constraints_rows (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (rows : List (LeafRow p)) {out : PrivateBatchOutput}
    {fee totalIn totalOut assetRef numRef : ZMod p} {blockRef : Digest4 p}
    {slots : List (SlotF p)}
    (hdummy : ∀ r ∈ rows, DummyCheckL r)
    (hdnull : ∀ r ∈ rows, DummyNullCheck perm r)
    (hlen : rows.length ≤ 64)
    (h32 : ∀ q ∈ rows.map LeafRow.leaf, inRange 32 q.inputAmount)
    -- the scanned references and the consistency constraints against them
    (hblockRef : blockRef = blockRefL rows)
    (hnumRef : numRef = scanRefL rows fun r => r.blockNumber)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef)
    (hcBlock : ∀ r ∈ rows, DigestConsistencyCheck r.isDummy r.blockHash blockRef)
    -- the header outputs
    (hoAsset : out.assetId = assetRef.val)
    (hoFee : out.volumeFeeBps = fee.val)
    (hoBlock : out.blockHash = valDigest blockRef)
    (hoNum : out.blockNumber = numRef.val)
    (hoSlots : out.numExitSlots = out.exitSlots.length)
    -- the masked input total, the exit slots and the fee comparator
    (hin : totalIn = (rows.map fun r => bselect r.isDummy 0 r.inputAmount).sum)
    (hslots32 : ∀ s ∈ slots, rangeCheck s.1 32)
    (hslotsLen : slots.length ≤ 64)
    (hout : totalOut = (slots.map Prod.fst).sum)
    (hfc : FeeCheck fee totalIn totalOut)
    (hoExits : out.exitSlots = slots.map valSlot)
    -- the nullifier region and the uniqueness constraints
    (hoNulls : out.nullifiers = (rows.map fun r => r.slot.sel).map valDigest)
    (hcol : rows.Pairwise fun r s => UniqCheck r.nullifier s.nullifier r.isDummy s.isDummy) :
    PrivateBatchConstraints perm (rows.map LeafRow.toSlotRow) fee totalIn totalOut out := by
  have hb : ∀ r ∈ rows, IsBool r.isDummy := fun r hr => (dummyCheckL_spec (hdummy r hr)).1
  have hd : ∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf :=
    fun r hr => (dummyCheckL_spec (hdummy r hr)).2
  have h32' : ∀ r ∈ rows, r.inputAmount.val < 2 ^ 32 :=
    fun r hr => h32 r.leaf (List.mem_map_of_mem hr)
  have hpG : 2 ^ 40 < p := lt_of_lt_of_le (by decide) hpg
  have hs32 : ∀ s ∈ slots, s.1.val < 2 ^ 32 :=
    fun s hs => rangeCheck_sound (le_of_lt (lt_trans (by decide) hpG)) (hslots32 s hs)
  have hmap : (rows.map LeafRow.toSlotRow).map (fun t => t.2.1) = rows.map LeafRow.leaf := by
    rw [List.map_map]; rfl
  refine
    { hb := List.forall_mem_map.mpr hb
      hd := List.forall_mem_map.mpr hd
      hdnull := List.forall_mem_map.mpr fun r hr => dummyNullCheck_val (hdnull r hr)
      hreal := List.forall_mem_map.mpr fun _ _ => rfl
      hnull := by rw [hoNulls, List.map_map, List.map_map, List.map_map]; rfl
      hcol := ?_
      hlen := by rw [List.length_map]; exact hlen
      h32 := by rw [hmap]; exact h32
      hslots := ?_
      hslotsLen := by rw [hoExits, List.length_map]; exact hslotsLen
      hfee := hoFee.symm
      hin := ?_
      hout := ?_
      hfc := hfc
      hmeta := by
        rw [hmap]
        exact metadataL_val_bridge rows hb hd hcAsset hcBlock hoAsset hoBlock
      href := by
        rw [hmap, hblockRef, hnumRef] at *
        exact referenceL_val_bridge rows hb hd hoBlock hoNum hoAsset hcAsset
      hnum := hoSlots }
  · -- pairwise uniqueness
    intro i j hi hj hij
    rw [List.length_map] at hi hj
    obtain ⟨m, hm, hz⟩ := List.pairwise_iff_getElem.mp hcol i j hi hj hij
    refine ⟨m, ?_, ?_⟩
    · simp only [List.getElem_map]
      exact (digestEqW_spec hm).2
    · simp only [List.getElem_map]
      exact hz
  · -- the exit amounts are 32-bit
    intro s hs
    rw [hoExits] at hs
    obtain ⟨f, hf, rfl⟩ := List.mem_map.1 hs
    exact hs32 f hf
  · -- dummy-masked input total
    rw [hin, (val_sum_of_le (2 ^ 32) _ ?_ ?_).1, maskedInputTotal_val rows hb hd, hmap]
    · intro x hx
      obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hx
      rw [val_mask (hb r hr)]
      have := h32' r hr
      split <;> omega
    · rw [List.length_map]
      have : rows.length * 2 ^ 32 ≤ 64 * 2 ^ 32 := Nat.mul_le_mul_right _ hlen
      omega
  · -- exit-slot total
    rw [hout, (val_sum_of_le (2 ^ 32) _ ?_ ?_).1]
    · unfold outputExitTotal
      rw [hoExits, slotsTotal_map_valSlot]
    · intro x hx
      obtain ⟨f, hf, rfl⟩ := List.mem_map.1 hx
      exact Nat.le_of_lt (hs32 f hf)
    · rw [List.length_map]
      have : slots.length * 2 ^ 32 ≤ 64 * 2 ^ 32 := Nat.mul_le_mul_right _ hslotsLen
      omega

/-- **The private-batch wrapper satisfies `RPrivateBatch`** on the decoded children,
    preimages and output (`private_batch_constraints_rows` through
    `PrivateBatchConstraints.sound`). -/
theorem private_batch_val_rows (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (rows : List (LeafRow p)) {out : PrivateBatchOutput}
    {fee totalIn totalOut assetRef numRef : ZMod p} {blockRef : Digest4 p}
    {slots : List (SlotF p)}
    (hdummy : ∀ r ∈ rows, DummyCheckL r)
    (hdnull : ∀ r ∈ rows, DummyNullCheck perm r)
    (hlen : rows.length ≤ 64)
    (h32 : ∀ q ∈ rows.map LeafRow.leaf, inRange 32 q.inputAmount)
    (hblockRef : blockRef = blockRefL rows)
    (hnumRef : numRef = scanRefL rows fun r => r.blockNumber)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef)
    (hcBlock : ∀ r ∈ rows, DigestConsistencyCheck r.isDummy r.blockHash blockRef)
    (hoAsset : out.assetId = assetRef.val)
    (hoFee : out.volumeFeeBps = fee.val)
    (hoBlock : out.blockHash = valDigest blockRef)
    (hoNum : out.blockNumber = numRef.val)
    (hoSlots : out.numExitSlots = out.exitSlots.length)
    (hin : totalIn = (rows.map fun r => bselect r.isDummy 0 r.inputAmount).sum)
    (hslots32 : ∀ s ∈ slots, rangeCheck s.1 32)
    (hslotsLen : slots.length ≤ 64)
    (hout : totalOut = (slots.map Prod.fst).sum)
    (hfc : FeeCheck fee totalIn totalOut)
    (hoExits : out.exitSlots = slots.map valSlot)
    (hoNulls : out.nullifiers = (rows.map fun r => r.slot.sel).map valDigest)
    (hcol : rows.Pairwise fun r s => UniqCheck r.nullifier s.nullifier r.isDummy s.isDummy) :
    RPrivateBatch (spongeRO perm) (rows.map LeafRow.leaf) (rows.map LeafRow.u) out := by
  have h := (private_batch_constraints_rows perm hpg rows hdummy hdnull hlen h32
    hblockRef hnumRef hcAsset hcBlock hoAsset hoFee hoBlock hoNum hoSlots hin hslots32 hslotsLen
    hout hfc hoExits hoNulls hcol).sound hpg
  rwa [List.map_map, List.map_map] at h

end Plonky2Bridge
