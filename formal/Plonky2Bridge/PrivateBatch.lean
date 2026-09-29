/-
  The private-batch wrapper across the `.val` seam, for any number of leaves (PLAN.md
  Step 8e).

  `build_private_batch_constraints` (wormhole/aggregator/src/private_batch/circuit/
  circuit_logic.rs) does, per child leaf proof: the dummy check
  `bytes_digest_eq(block_hash_i, 0)`, a first-real prefix scan selecting the header from the
  first non-dummy child, the consistency constraints against the scanned references, the
  dummy-masked totals and the fee comparator, the exit-slot grouping loop over the `2n`
  masked `(exit, amount)` candidates, the pairwise nullifier-uniqueness constraints, the
  dummy-nullifier double hash and the odd-even switch network on the selected nullifiers.

  This module states each of those as a list-shaped hypothesis over `LeafRow`s — the field
  wires of one child together with the `is_equal` witnesses of its dummy check — and lifts
  them into `PrivateBatchConstraints` (`private_batch_constraints_rows`), hence into
  `RPrivateBatch` (`private_batch_val_rows`). The exporter-generated
  `Plonky2Bridge/Generated/Wrapper{N}.lean` discharges every hypothesis for a recorded
  `n = N` trace by rewriting with the facts of `privateBatchWrapper{N}_decode`.
-/
import Plonky2Bridge.Complete
import Plonky2Bridge.PublicBatch

namespace Plonky2Bridge

open Plonky2Spec (IsBool bselect band bnot bor bnot_isBool bor_isBool Digest4 network digestEq
  bytesDigestEq_spec IsEqual scanStep firstRealVal scanFirst_correct real_block_matches
  isEqual_iff isEqual_isBool bselect_true bselect_false match_contribution dedup_select
  FeeCheck bnot_eq_one bor_eq_one)
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash)
open WormholeSpec (Digest LeafPublic PrivateBatchOutput ExitSlot isDummyPrivateBatch isRealB
  Felt matchSum maskedChildPairs groupExits groupAux maskedInputTotal maskedOutputTotal
  metadataConsistent referenceFromFirstReal inRange RPrivateBatch isRealB_true_iff)

variable {p : ℕ} [Fact p.Prime]

/-! ### Field witness for one child -/

/-- The public-input wires of one child leaf proof the wrapper reads, the `is_equal`
    witnesses (`equal` flag and auxiliary inverse per limb) of its
    `bytes_digest_eq(block_hash, 0)` dummy check, its dummy-nullifier preimage and the
    outer hash output. The dummy flag itself is the gadget's output, `LeafRow.isDummy`. -/
structure LeafRow (p : ℕ) where
  assetId : ZMod p
  out1 : ZMod p
  out2 : ZMod p
  fee : ZMod p
  nullifier : Digest4 p
  exit1 : Digest4 p
  exit2 : Digest4 p
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
  { assetId := r.assetId.val, outputAmount1 := r.out1.val, outputAmount2 := r.out2.val,
    volumeFeeBps := r.fee.val, nullifier := valDigest r.nullifier,
    exitAccount1 := valDigest r.exit1, exitAccount2 := valDigest r.exit2,
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
    (hfee : out.volumeFeeBps = (scanRefL rows fun r => r.fee).val)
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
  rw [hblock, hnum, hfee, hblockRef, scanRefL_eq rows _ hb hd, scanRefL_eq rows _ hb hd]
  cases hf : rows.find? (fun r => isRealB r.leaf) with
  | none =>
      simp only [Option.map_none]
      exact ⟨valDigest_zero, ZMod.val_zero, ZMod.val_zero⟩
  | some r =>
      have hr : r ∈ rows := List.mem_of_find?_eq_some hf
      simp only [Option.map_some]
      exact ⟨rfl, rfl, by rw [hasset, ← hcAsset r hr]; rfl, rfl⟩

/-- **Metadata bridge.** Every non-dummy child agrees with the header. -/
theorem metadataL_val_bridge (rows : List (LeafRow p)) {out : PrivateBatchOutput}
    {assetRef feeRef : ZMod p} {blockRef : Digest4 p}
    (hb : ∀ r ∈ rows, IsBool r.isDummy)
    (hd : ∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef)
    (hcFee : ∀ r ∈ rows, ConsistencyCheck r.isDummy r.fee feeRef)
    (hcBlock : ∀ r ∈ rows, DigestConsistencyCheck r.isDummy r.blockHash blockRef)
    (hoAsset : out.assetId = assetRef.val)
    (hoFee : out.volumeFeeBps = feeRef.val)
    (hoBlock : out.blockHash = valDigest blockRef) :
    metadataConsistent (rows.map LeafRow.leaf) out := by
  intro q hq hreal
  obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hq
  have hnd : r.isDummy ≠ 1 := fun h1 => hreal ((hd r hr).mp h1)
  refine ⟨?_, ?_, ?_⟩
  · rw [hoAsset, ← hcAsset r hr]; rfl
  · rw [hoFee]; exact consistency_val (hb r hr) hnd (hcFee r hr)
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

theorem val_mask2 {d x y : ZMod p} (hb : IsBool d) (hxy : x.val + y.val < p) :
    (bselect d 0 x + bselect d 0 y).val = if d = 1 then 0 else x.val + y.val := by
  rcases hb with h0 | h1
  · rw [bselect_false h0, bselect_false h0, if_neg (by rw [h0]; exact zero_ne_one),
      ZMod.val_add_of_lt hxy]
  · rw [bselect_true h1, bselect_true h1, if_pos h1]; simp

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

theorem maskedOutputTotal_val :
    ∀ rows : List (LeafRow p), (∀ r ∈ rows, IsBool r.isDummy) →
      (∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) →
      (∀ r ∈ rows, r.out1.val + r.out2.val < p) →
      ((rows.map fun r => bselect r.isDummy 0 r.out1 + bselect r.isDummy 0 r.out2).map
          ZMod.val).sum
        = maskedOutputTotal (rows.map LeafRow.leaf)
  | [], _, _, _ => rfl
  | r :: rest, hb, hd, hlt => by
      simp only [List.map_cons, List.sum_cons, maskedOutputTotal]
      rw [maskedOutputTotal_val rest (fun q hq => hb q (List.mem_cons_of_mem _ hq))
        (fun q hq => hd q (List.mem_cons_of_mem _ hq))
        (fun q hq => hlt q (List.mem_cons_of_mem _ hq)),
        val_mask2 (hb r List.mem_cons_self) (hlt r List.mem_cons_self)]
      by_cases h : r.isDummy = 1
      · rw [if_pos h, if_pos ((hd r List.mem_cons_self).mp h)]
      · rw [if_neg h, if_neg (fun h' => h ((hd r List.mem_cons_self).mpr h'))]; rfl

/-! ### The exit-grouping loop -/

/-- The `2n` masked `(exit, amount)` candidates, in slot order. -/
def candsL (rows : List (LeafRow p)) : List (Digest4 p × ZMod p) :=
  rows.flatMap fun r =>
    [(fun j => bselect r.isDummy 0 (r.exit1 j), bselect r.isDummy 0 r.out1),
     (fun j => bselect r.isDummy 0 (r.exit2 j), bselect r.isDummy 0 r.out2)]

theorem candsL_val :
    ∀ rows : List (LeafRow p), (∀ r ∈ rows, IsBool r.isDummy) →
      (∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf) →
      (candsL rows).map (fun q => (valDigest q.1, q.2.val))
        = maskedChildPairs (rows.map LeafRow.leaf)
  | [], _, _ => rfl
  | r :: rest, hb, hd => by
      have hbr := hb r List.mem_cons_self
      simp only [candsL, List.flatMap_cons, List.map_cons, List.cons_append, List.nil_append,
        maskedChildPairs]
      rw [← candsL, candsL_val rest (fun q hq => hb q (List.mem_cons_of_mem _ hq))
        (fun q hq => hd q (List.mem_cons_of_mem _ hq)),
        valDigest_mask hbr, valDigest_mask hbr, val_mask hbr, val_mask hbr]
      by_cases h : r.isDummy = 1
      · simp only [if_pos h, if_pos ((hd r List.mem_cons_self).mp h)]
      · simp only [if_neg h, if_neg (fun h' => h ((hd r List.mem_cons_self).mpr h'))]
        rfl

theorem candsL_length (rows : List (LeafRow p)) : (candsL rows).length = 2 * rows.length := by
  induction rows with
  | nil => rfl
  | cons r rest ih =>
      simp only [candsL, List.flatMap_cons, List.length_append, List.length_cons,
        List.length_nil] at ih ⊢
      omega

theorem candsL_bound (rows : List (LeafRow p)) (hb : ∀ r ∈ rows, IsBool r.isDummy)
    (h32 : ∀ r ∈ rows, r.out1.val < 2 ^ 32 ∧ r.out2.val < 2 ^ 32) :
    ∀ q ∈ candsL rows, q.2.val < 2 ^ 32 := by
  intro q hq
  obtain ⟨r, hr, hmem⟩ := List.mem_flatMap.1 hq
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
  have hbr := hb r hr
  obtain ⟨h1, h2⟩ := h32 r hr
  rcases hmem with rfl | rfl
  · show (bselect r.isDummy 0 r.out1).val < 2 ^ 32
    rw [val_mask hbr]; split <;> omega
  · show (bselect r.isDummy 0 r.out2).val < 2 ^ 32
    rw [val_mask hbr]; split <;> omega

/-- The wires of one slot of the grouping loop: the duplicate flag, the accumulated sum,
    the self-match flag, the `bytes_digest_eq` flags against the earlier slots (dedup) and
    against the earlier and later candidates (matching). -/
structure SlotW (p : ℕ) where
  dup : ZMod p
  acc : ZMod p
  eqk : ZMod p
  dupFlags : List (ZMod p)
  mEarlier : List (ZMod p)
  mLater : List (ZMod p)

/-- The constraints of one slot with key `Ek` and amount `Ak`, between the `earlier` and
    `later` candidates: every flag is a recorded `bytes_digest_eq`, the duplicate flag is
    the `or`-fold of the dedup flags, the accumulator the `add`-fold of every candidate's
    `select(match, amount, 0)` (self included), and the emitted slot is the masked pair. -/
def SlotCheck (earlier : List (Digest4 p × ZMod p)) (Ek : Digest4 p) (Ak : ZMod p)
    (later : List (Digest4 p × ZMod p)) (w : SlotW p) (s : SlotF p) : Prop :=
  List.Forall₂ (fun q m => DigestEqW q.1 Ek m) earlier w.dupFlags ∧
  List.Forall₂ (fun q m => DigestEqW q.1 Ek m) earlier w.mEarlier ∧
  DigestEqW Ek Ek w.eqk ∧
  List.Forall₂ (fun q m => DigestEqW q.1 Ek m) later w.mLater ∧
  w.dup = w.dupFlags.foldl bor 0 ∧
  w.acc = (List.zipWith (fun m (q : Digest4 p × ZMod p) => bselect m q.2 0) w.mEarlier earlier
      ++ bselect w.eqk Ak 0
      :: List.zipWith (fun m (q : Digest4 p × ZMod p) => bselect m q.2 0) w.mLater later).foldl
        (· + ·) 0 ∧
  s.1 = bselect w.dup 0 w.acc ∧ ∀ j, s.2 j = bselect w.dup 0 (Ek j)

/-- The slots of the candidates `rest`, each checked after the `earlier` ones. -/
def SlotsOk : List (Digest4 p × ZMod p) → List (Digest4 p × ZMod p) → List (SlotF p) → Prop
  | _, [], [] => True
  | earlier, q :: rest, s :: slots =>
      (∃ w, SlotCheck earlier q.1 q.2 rest w s) ∧ SlotsOk (earlier ++ [q]) rest slots
  | _, _, _ => False

theorem foldl_add_eq : ∀ (l : List (ZMod p)) (b : ZMod p), l.foldl (· + ·) b = b + l.sum
  | [], _ => by simp
  | x :: rest, b => by rw [List.foldl_cons, List.sum_cons, foldl_add_eq rest (b + x), add_assoc]

theorem foldl_bor_isBool : ∀ (l : List (ZMod p)) (b : ZMod p), IsBool b →
    (∀ m ∈ l, IsBool m) → IsBool (l.foldl bor b)
  | [], _, hb, _ => hb
  | m :: rest, _, hb, hl =>
      foldl_bor_isBool rest _ (bor_isBool hb (hl m List.mem_cons_self))
        (fun x hx => hl x (List.mem_cons_of_mem _ hx))

theorem foldl_bor_eq_one : ∀ (l : List (ZMod p)) (b : ZMod p), IsBool b →
    (∀ m ∈ l, IsBool m) → (l.foldl bor b = 1 ↔ b = 1 ∨ ∃ m ∈ l, m = 1)
  | [], _, _, _ => by simp
  | m :: rest, b, hb, hl => by
      rw [List.foldl_cons, foldl_bor_eq_one rest _ (bor_isBool hb (hl m List.mem_cons_self))
        (fun x hx => hl x (List.mem_cons_of_mem _ hx)), bor_eq_one hb (hl m List.mem_cons_self)]
      simp only [List.mem_cons, exists_eq_or_imp, or_assoc]

theorem forall₂_flags_isBool {Ek : Digest4 p} :
    ∀ {qs : List (Digest4 p × ZMod p)} {ms : List (ZMod p)},
      List.Forall₂ (fun q m => DigestEqW q.1 Ek m) qs ms → ∀ m ∈ ms, IsBool m
  | _, _, List.Forall₂.nil, _, h => nomatch h
  | _, _, List.Forall₂.cons hqm hrest, m, hm => by
      rcases List.mem_cons.1 hm with rfl | hm
      · exact (digestEqW_spec hqm).1
      · exact forall₂_flags_isBool hrest m hm

theorem forall₂_flags_exists {Ek : Digest4 p} :
    ∀ {qs : List (Digest4 p × ZMod p)} {ms : List (ZMod p)},
      List.Forall₂ (fun q m => DigestEqW q.1 Ek m) qs ms →
      ((∃ m ∈ ms, m = 1) ↔ ∃ q ∈ qs, q.1 = Ek)
  | _, _, List.Forall₂.nil => by simp
  | _, _, List.Forall₂.cons hqm hrest => by
      rw [List.exists_mem_cons_iff, List.exists_mem_cons_iff, forall₂_flags_exists hrest,
        (digestEqW_spec hqm).2]

theorem earlier_sum_zero {Ek : Digest4 p} :
    ∀ {qs : List (Digest4 p × ZMod p)} {ms : List (ZMod p)},
      List.Forall₂ (fun q m => DigestEqW q.1 Ek m) qs ms → (¬ ∃ q ∈ qs, q.1 = Ek) →
      (List.zipWith (fun m (q : Digest4 p × ZMod p) => bselect m q.2 0) ms qs).sum = 0
  | _, _, List.Forall₂.nil, _ => rfl
  | q :: _, m :: _, List.Forall₂.cons hqm hrest, hn => by
      simp only [List.zipWith_cons_cons, List.sum_cons]
      have ⟨hb, hiff⟩ := digestEqW_spec hqm
      have hne : q.1 ≠ Ek := fun h => hn ⟨q, List.mem_cons_self, h⟩
      rw [bselect_false (hb.resolve_right (fun h1 => hne (hiff.mp h1))), zero_add]
      exact earlier_sum_zero hrest (fun ⟨q', hq', h⟩ => hn ⟨q', List.mem_cons_of_mem _ hq', h⟩)

theorem later_sum_eq {Ek : Digest4 p} :
    ∀ {qs : List (Digest4 p × ZMod p)} {ms : List (ZMod p)},
      List.Forall₂ (fun q m => DigestEqW q.1 Ek m) qs ms →
      (List.zipWith (fun m (q : Digest4 p × ZMod p) => bselect m q.2 0) ms qs).sum
        = (qs.map fun q => if q.1 = Ek then q.2 else 0).sum
  | _, _, List.Forall₂.nil => rfl
  | _ :: _, _ :: _, List.Forall₂.cons hqm hrest => by
      simp only [List.zipWith_cons_cons, List.sum_cons, List.map_cons]
      have ⟨hb, hiff⟩ := digestEqW_spec hqm
      rw [match_contribution hb hiff, later_sum_eq hrest]

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
    ∀ l : List (Digest4 p × ZMod p), (l.map fun q : Digest4 p × ZMod p => q.2.val).sum < p →
      ((l.map fun q : Digest4 p × ZMod p => if q.1 = Ek then q.2 else 0).sum).val
        = matchSum (valDigest Ek) (l.map fun q : Digest4 p × ZMod p => (valDigest q.1, q.2.val))
  | [], _ => by simp [matchSum]
  | q :: rest, hb => by
      simp only [List.map_cons, List.sum_cons] at hb ⊢
      have ih := matchSum_val Ek rest (by omega)
      have hle := matchSum_le (valDigest Ek) (rest.map fun q : Digest4 p × ZMod p => (valDigest q.1, q.2.val))
      simp only [List.map_map, Function.comp_def] at hle
      have hrest : ((rest.map fun q : Digest4 p × ZMod p => if q.1 = Ek then q.2 else 0).sum).val
          ≤ (rest.map fun q : Digest4 p × ZMod p => q.2.val).sum := by
        rw [ih]; exact hle
      have ht : (if q.1 = Ek then q.2 else 0).val
          = if valDigest q.1 = valDigest Ek then q.2.val else 0 := by
        by_cases h : q.1 = Ek
        · rw [if_pos h, if_pos (by rw [h])]
        · rw [if_neg h, if_neg (fun hv => h (valDigest_injective hv)), ZMod.val_zero]
      have hlt : (if q.1 = Ek then q.2 else 0).val
          + ((rest.map fun q : Digest4 p × ZMod p => if q.1 = Ek then q.2 else 0).sum).val < p := by
        rw [ht]
        split <;> omega
      rw [ZMod.val_add_of_lt hlt, ht, ih, matchSum]

theorem matchSum_val_le (Ek : Digest4 p) (l : List (Digest4 p × ZMod p))
    (hb : (l.map fun q : Digest4 p × ZMod p => q.2.val).sum < p) :
    ((l.map fun q : Digest4 p × ZMod p => if q.1 = Ek then q.2 else 0).sum).val ≤ (l.map fun q : Digest4 p × ZMod p => q.2.val).sum := by
  rw [matchSum_val Ek l hb]
  have hle := matchSum_le (valDigest Ek) (l.map fun q : Digest4 p × ZMod p => (valDigest q.1, q.2.val))
  simp only [List.map_map, Function.comp_def] at hle
  exact hle

theorem mem_seen_iff (Ek : Digest4 p) (earlier : List (Digest4 p × ZMod p)) :
    valDigest Ek ∈ earlier.map (fun q => valDigest q.1) ↔ ∃ q ∈ earlier, q.1 = Ek := by
  rw [List.mem_map]
  constructor
  · rintro ⟨q, hq, h⟩; exact ⟨q, hq, valDigest_injective h⟩
  · rintro ⟨q, hq, h⟩; exact ⟨q, hq, by rw [h]⟩

/-- **One exit slot of the grouping loop, read through `.val`.** The emitted
    `(select(dup, 0, acc), select(dup, 0, exit))` is `groupAux`'s slot for `seen` the
    earlier keys and `rest` the later pairs. -/
theorem slotCheck_val {earlier later : List (Digest4 p × ZMod p)} {Ek : Digest4 p}
    {Ak : ZMod p} {w : SlotW p} {s : SlotF p} (h : SlotCheck earlier Ek Ak later w s)
    (hbound : Ak.val + (later.map fun q => q.2.val).sum < p)
    {seen : List Digest} (hseen : ∀ k, k ∈ seen ↔ k ∈ earlier.map fun q => valDigest q.1) :
    valSlot s
      = if valDigest Ek ∈ seen then ⟨0, Digest.zero⟩
        else ⟨Ak.val + matchSum (valDigest Ek) (later.map fun q => (valDigest q.1, q.2.val)),
          valDigest Ek⟩ := by
  obtain ⟨hdupF, hmE, heqk, hmL, hdup, hacc, hs1, hs2⟩ := h
  have hdupB : IsBool w.dup := by
    rw [hdup]; exact foldl_bor_isBool _ _ (Or.inl rfl) (forall₂_flags_isBool hdupF)
  have hP : w.dup = 1 ↔ valDigest Ek ∈ seen := by
    rw [hseen, mem_seen_iff, hdup,
      foldl_bor_eq_one _ _ (Or.inl rfl) (forall₂_flags_isBool hdupF), forall₂_flags_exists hdupF]
    simp
  have hs2' : s.2 = fun j => bselect w.dup 0 (Ek j) := funext hs2
  show (⟨s.1.val, valDigest s.2⟩ : ExitSlot) = _
  rw [hs1, hs2', dedup_select hdupB hP, valDigest_mask hdupB]
  by_cases hin : valDigest Ek ∈ seen
  · simp only [if_pos hin, if_pos (hP.mpr hin), ZMod.val_zero]
  · simp only [if_neg hin, if_neg (fun h1 => hin (hP.mp h1))]
    have hnone : ¬ ∃ q ∈ earlier, q.1 = Ek :=
      fun h => hin ((hseen _).mpr ((mem_seen_iff _ _).mpr h))
    have hk : bselect w.eqk Ak 0 = Ak := by
      rw [match_contribution (digestEqW_spec heqk).1 (digestEqW_spec heqk).2, if_pos rfl]
    rw [hacc, foldl_add_eq, zero_add, List.sum_append, List.sum_cons, earlier_sum_zero hmE hnone,
      zero_add, hk, later_sum_eq hmL]
    have hb' : (later.map fun q => q.2.val).sum < p :=
      lt_of_le_of_lt (Nat.le_add_left _ _) hbound
    rw [ZMod.val_add_of_lt (lt_of_le_of_lt (Nat.add_le_add_left (matchSum_val_le Ek later hb') _)
      hbound), matchSum_val Ek later hb']

theorem sum_map_le_length_mul {α : Type} (f : α → ℕ) (B : ℕ) :
    ∀ l : List α, (∀ x ∈ l, f x ≤ B) → (l.map f).sum ≤ l.length * B
  | [], _ => by simp
  | x :: rest, h => by
      simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.succ_mul]
      have := sum_map_le_length_mul f B rest (fun y hy => h y (List.mem_cons_of_mem _ hy))
      have := h x List.mem_cons_self
      omega

/-- **The grouping loop, read through `.val`.** With every candidate amount below `2^32` and
    at most 128 candidates, the emitted slots are `groupExits` of the decoded pairs. -/
theorem slotsOk_val (hpg : WormholeSpec.goldilocks ≤ p) (all : List (Digest4 p × ZMod p))
    (hb : ∀ q ∈ all, q.2.val < 2 ^ 32) (hlen : all.length ≤ 128) :
    ∀ (earlier rest : List (Digest4 p × ZMod p)) (slots : List (SlotF p)) (seen : List Digest),
      all = earlier ++ rest → (∀ k, k ∈ seen ↔ k ∈ earlier.map fun q => valDigest q.1) →
      SlotsOk earlier rest slots →
      slots.map valSlot = groupAux seen (rest.map fun q => (valDigest q.1, q.2.val))
  | _, [], [], _, _, _, _ => rfl
  | _, [], _ :: _, _, _, _, h => nomatch h
  | _, _ :: _, [], _, _, _, h => nomatch h
  | earlier, q :: rest, s :: slots, seen, hall, hseen, ⟨⟨w, hw⟩, hrest⟩ => by
      have hpG : 2 ^ 39 < p := lt_of_lt_of_le (by decide) hpg
      have hsum : ((q :: rest).map fun q => q.2.val).sum ≤ 128 * 2 ^ 32 := by
        have hle := sum_map_le_length_mul (fun q : Digest4 p × ZMod p => q.2.val) (2 ^ 32)
          (q :: rest) (fun x hx => Nat.le_of_lt_succ (Nat.lt_succ_of_lt
            (hb x (by rw [hall]; exact List.mem_append_right _ hx))))
        have hl : (q :: rest).length ≤ 128 := by
          rw [hall, List.length_append] at hlen; omega
        exact le_trans hle (Nat.mul_le_mul_right _ hl)
      simp only [List.map_cons, List.sum_cons] at hsum
      simp only [List.map_cons, groupAux]
      rw [slotCheck_val hw (by omega) hseen]
      congr 1
      exact slotsOk_val hpg all hb hlen (earlier ++ [q]) rest slots _
        (by rw [hall, List.append_assoc]; rfl)
        (by
          intro k
          simp only [List.mem_cons, List.map_append, List.map_cons, List.map_nil,
            List.mem_append, hseen]
          tauto)
        hrest

/-! ### Nullifier uniqueness -/

/-- One pairwise constraint: `and(and(not d_i, not d_j), bytes_digest_eq(null_i, null_j))`
    connected to zero. -/
def UniqCheck (x y : Digest4 p) (dx dy : ZMod p) : Prop :=
  ∃ m, DigestEqW x y m ∧ band (band (bnot dx) (bnot dy)) m = 0

/-! ### Composition -/

/-- **Wrapper-logic composition (private batch).** From the recorded gadget constraints —
    stated per child, per candidate slot and per pair — the wrapper's field witness satisfies
    `PrivateBatchConstraints` on the decoded rows. Booleanity of the dummy flags and every
    equality flag's meaning are derived from the gadget lemmas, not assumed. -/
theorem private_batch_constraints_rows (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (rows : List (LeafRow p)) (rounds : List (List (ZMod p))) {out : PrivateBatchOutput}
    {feeRef totalIn totalOut assetRef numRef : ZMod p} {blockRef : Digest4 p}
    {cands : List (Digest4 p × ZMod p)} {slots : List (SlotF p)}
    (hsw : ∀ s ∈ rounds.flatten, IsBool s)
    (hdummy : ∀ r ∈ rows, DummyCheckL r)
    (hdnull : ∀ r ∈ rows, DummyNullCheck perm r)
    (hlen : rows.length ≤ 64)
    (h32 : ∀ q ∈ rows.map LeafRow.leaf, inRange 32 q.inputAmount ∧
      inRange 32 q.outputAmount1 ∧ inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps)
    -- the scanned references and the consistency constraints against them
    (hblockRef : blockRef = blockRefL rows)
    (hnumRef : numRef = scanRefL rows fun r => r.blockNumber)
    (hfeeRef : feeRef = scanRefL rows fun r => r.fee)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef)
    (hcFee : ∀ r ∈ rows, ConsistencyCheck r.isDummy r.fee feeRef)
    (hcBlock : ∀ r ∈ rows, DigestConsistencyCheck r.isDummy r.blockHash blockRef)
    -- the header outputs
    (hoAsset : out.assetId = assetRef.val)
    (hoFee : out.volumeFeeBps = feeRef.val)
    (hoBlock : out.blockHash = valDigest blockRef)
    (hoNum : out.blockNumber = numRef.val)
    (hoSlots : out.numExitSlots = 2 * rows.length)
    -- the masked totals and the fee comparator
    (hin : totalIn = (rows.map fun r => bselect r.isDummy 0 r.inputAmount).sum)
    (hout : totalOut =
      (rows.map fun r => bselect r.isDummy 0 r.out1 + bselect r.isDummy 0 r.out2).sum)
    (hfc : FeeCheck feeRef totalIn totalOut)
    -- the grouping loop
    (hcands : cands = candsL rows)
    (hslots : SlotsOk [] cands slots)
    (hoExits : out.exitSlots = slots.map valSlot)
    -- the nullifier region and the uniqueness constraints
    (hoNulls : out.nullifiers = (network rounds (rows.map fun r => r.slot.sel)).map valDigest)
    (hcol : rows.Pairwise fun r s => UniqCheck r.nullifier s.nullifier r.isDummy s.isDummy) :
    PrivateBatchConstraints perm rounds (rows.map LeafRow.toSlotRow) feeRef totalIn totalOut
      out := by
  have hb : ∀ r ∈ rows, IsBool r.isDummy := fun r hr => (dummyCheckL_spec (hdummy r hr)).1
  have hd : ∀ r ∈ rows, r.isDummy = 1 ↔ isDummyPrivateBatch r.leaf :=
    fun r hr => (dummyCheckL_spec (hdummy r hr)).2
  have h32' : ∀ r ∈ rows, r.inputAmount.val < 2 ^ 32 ∧ r.out1.val < 2 ^ 32 ∧
      r.out2.val < 2 ^ 32 ∧ r.fee.val < 2 ^ 32 :=
    fun r hr => h32 r.leaf (List.mem_map_of_mem hr)
  have hpG : 2 ^ 40 < p := lt_of_lt_of_le (by decide) hpg
  have hmap : (rows.map LeafRow.toSlotRow).map (fun t => t.2.1) = rows.map LeafRow.leaf := by
    rw [List.map_map]; rfl
  refine
    { hsw := fun ss hss s hs => hsw s (List.mem_flatten.mpr ⟨ss, hss, hs⟩)
      hb := List.forall_mem_map.mpr hb
      hd := List.forall_mem_map.mpr hd
      hdnull := List.forall_mem_map.mpr fun r hr => dummyNullCheck_val (hdnull r hr)
      hreal := List.forall_mem_map.mpr fun _ _ => rfl
      hnull := by rw [hoNulls, List.map_map]; rfl
      hcol := ?_
      hlen := by rw [List.length_map]; exact hlen
      h32 := by rw [hmap]; exact h32
      hfee := hoFee.symm
      hin := ?_
      hout := ?_
      hfc := hfc
      hexits := ?_
      hmeta := by
        rw [hmap]
        exact metadataL_val_bridge rows hb hd hcAsset hcFee hcBlock hoAsset hoFee hoBlock
      href := by
        rw [hmap, hblockRef, hnumRef] at *
        exact referenceL_val_bridge rows hb hd hoBlock hoNum (hfeeRef ▸ hoFee) hoAsset hcAsset
      hnum := by rw [hoSlots, List.length_map] }
  · -- pairwise uniqueness
    intro i j hi hj hij
    rw [List.length_map] at hi hj
    obtain ⟨m, hm, hz⟩ := List.pairwise_iff_getElem.mp hcol i j hi hj hij
    refine ⟨m, ?_, ?_⟩
    · simp only [List.getElem_map]
      exact (digestEqW_spec hm).2
    · simp only [List.getElem_map]
      exact hz
  · -- dummy-masked input total
    rw [hin, (val_sum_of_le (2 ^ 32) _ ?_ ?_).1, maskedInputTotal_val rows hb hd, hmap]
    · intro x hx
      obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hx
      rw [val_mask (hb r hr)]
      have := (h32' r hr).1
      split <;> omega
    · rw [List.length_map]
      have : rows.length * 2 ^ 32 ≤ 64 * 2 ^ 32 := Nat.mul_le_mul_right _ hlen
      omega
  · -- dummy-masked output total
    rw [hout, (val_sum_of_le (2 ^ 33) _ ?_ ?_).1,
      maskedOutputTotal_val rows hb hd (fun r hr => by have := h32' r hr; omega), hmap]
    · intro x hx
      obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hx
      have := h32' r hr
      rw [val_mask2 (hb r hr) (by omega)]
      split <;> omega
    · rw [List.length_map]
      have : rows.length * 2 ^ 33 ≤ 64 * 2 ^ 33 := Nat.mul_le_mul_right _ hlen
      omega
  · -- exit grouping
    rw [hoExits, hmap, ← candsL_val rows hb hd]
    unfold groupExits
    subst hcands
    exact slotsOk_val hpg (candsL rows) (candsL_bound rows hb fun r hr => by
        have := h32' r hr; omega)
      (by rw [candsL_length]; omega) [] (candsL rows) slots [] rfl (by simp) hslots

/-- **The private-batch wrapper satisfies `RPrivateBatch`** on the decoded children,
    preimages and output (`private_batch_constraints_rows` through
    `PrivateBatchConstraints.sound`). -/
theorem private_batch_val_rows (perm : St p → St p) (hpg : WormholeSpec.goldilocks ≤ p)
    (rows : List (LeafRow p)) (rounds : List (List (ZMod p))) {out : PrivateBatchOutput}
    {feeRef totalIn totalOut assetRef numRef : ZMod p} {blockRef : Digest4 p}
    {cands : List (Digest4 p × ZMod p)} {slots : List (SlotF p)}
    (hsw : ∀ s ∈ rounds.flatten, IsBool s)
    (hdummy : ∀ r ∈ rows, DummyCheckL r)
    (hdnull : ∀ r ∈ rows, DummyNullCheck perm r)
    (hlen : rows.length ≤ 64)
    (h32 : ∀ q ∈ rows.map LeafRow.leaf, inRange 32 q.inputAmount ∧
      inRange 32 q.outputAmount1 ∧ inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps)
    (hblockRef : blockRef = blockRefL rows)
    (hnumRef : numRef = scanRefL rows fun r => r.blockNumber)
    (hfeeRef : feeRef = scanRefL rows fun r => r.fee)
    (hcAsset : ∀ r ∈ rows, r.assetId = assetRef)
    (hcFee : ∀ r ∈ rows, ConsistencyCheck r.isDummy r.fee feeRef)
    (hcBlock : ∀ r ∈ rows, DigestConsistencyCheck r.isDummy r.blockHash blockRef)
    (hoAsset : out.assetId = assetRef.val)
    (hoFee : out.volumeFeeBps = feeRef.val)
    (hoBlock : out.blockHash = valDigest blockRef)
    (hoNum : out.blockNumber = numRef.val)
    (hoSlots : out.numExitSlots = 2 * rows.length)
    (hin : totalIn = (rows.map fun r => bselect r.isDummy 0 r.inputAmount).sum)
    (hout : totalOut =
      (rows.map fun r => bselect r.isDummy 0 r.out1 + bselect r.isDummy 0 r.out2).sum)
    (hfc : FeeCheck feeRef totalIn totalOut)
    (hcands : cands = candsL rows)
    (hslots : SlotsOk [] cands slots)
    (hoExits : out.exitSlots = slots.map valSlot)
    (hoNulls : out.nullifiers = (network rounds (rows.map fun r => r.slot.sel)).map valDigest)
    (hcol : rows.Pairwise fun r s => UniqCheck r.nullifier s.nullifier r.isDummy s.isDummy) :
    RPrivateBatch (spongeRO perm) (rows.map LeafRow.leaf) (rows.map LeafRow.u) out := by
  have h := (private_batch_constraints_rows perm hpg rows rounds hsw hdummy hdnull hlen h32
    hblockRef hnumRef hfeeRef hcAsset hcFee hcBlock hoAsset hoFee hoBlock hoNum hoSlots hin hout
    hfc hcands hslots hoExits hoNulls hcol).sound hpg
  rwa [List.map_map, List.map_map] at h

end Plonky2Bridge
