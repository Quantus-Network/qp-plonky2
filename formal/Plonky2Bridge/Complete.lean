/-
  Completeness of the private-batch wrapper across the `.val` seam (PLAN.md Step 7b).

  `private_batch_val` says: every field witness satisfying the wrapper constraints decodes
  to an `RPrivateBatch` instance (soundness). This module proves the converse: every
  `RPrivateBatch` instance — under an explicit `CompletenessAssumptions` record — is
  realized by *some* field witness satisfying those same constraints. The two directions
  share one predicate, `PrivateBatchConstraints`, so the constraint list cannot drift.

  The assumptions record is the point of the exercise. It is the complete list of things
  an honest prover must arrange that the spec relation itself does not force:
  the modulus, the 64-leaf cap, one preimage per slot, the children's 32-bit ranges,
  canonical (`< p`) nullifier lanes, and a routing witness for the permutation network.
-/
import Plonky2Bridge

namespace Plonky2Bridge

open Plonky2Spec (IsBool bselect band bnot bselect_true bselect_false Digest4 network
  FeeCheck feeCheck_complete)
open Plonky2Spec.Poseidon2 (St)
open WormholeSpec (Digest RandomOracle LeafPublic PrivateBatchOutput isDummyPrivateBatch
  buildNullifiers nullifiersReplaced RPrivateBatch realNullifiersDistinct maskedInputTotal
  maskedOutputTotal privateBatchFeeOk feeDenominator inRange groupExits maskedChildPairs
  metadataConsistent referenceFromFirstReal goldilocks Felt)

variable {p : ℕ} [Fact p.Prime]

/-! ### Canonical digests and the input-side cast -/

/-- Every lane below the modulus: exactly the digests `valDigest` can reproduce. -/
def DigestCanonical (p : ℕ) (d : Digest) : Prop :=
  d.x0 < p ∧ d.x1 < p ∧ d.x2 < p ∧ d.x3 < p

/-- Cast a spec digest into four field lanes (the input side of the seam). -/
def castDigest (d : Digest) : Digest4 p
  | 0 => (d.x0 : ZMod p)
  | 1 => (d.x1 : ZMod p)
  | 2 => (d.x2 : ZMod p)
  | 3 => (d.x3 : ZMod p)

theorem valDigest_castDigest {d : Digest} (h : DigestCanonical p d) :
    valDigest (castDigest (p := p) d) = d := by
  obtain ⟨h0, h1, h2, h3⟩ := h
  show (⟨((d.x0 : ℕ) : ZMod p).val, ((d.x1 : ℕ) : ZMod p).val,
      ((d.x2 : ℕ) : ZMod p).val, ((d.x3 : ℕ) : ZMod p).val⟩ : Digest) = d
  rw [ZMod.val_natCast_of_lt h0, ZMod.val_natCast_of_lt h1, ZMod.val_natCast_of_lt h2,
    ZMod.val_natCast_of_lt h3]

theorem castDigest_injOn {a b : Digest} (ha : DigestCanonical p a) (hb : DigestCanonical p b)
    (h : castDigest (p := p) a = castDigest b) : a = b := by
  rw [← valDigest_castDigest ha, ← valDigest_castDigest hb, h]

theorem valDigest_canonical (f : Digest4 p) : DigestCanonical p (valDigest f) :=
  ⟨ZMod.val_lt _, ZMod.val_lt _, ZMod.val_lt _, ZMod.val_lt _⟩

/-- The sponge's outputs are `.val`s, hence canonical. -/
theorem spongeH_canonical (perm : St p → St p) (l : List ℕ) :
    DigestCanonical p (spongeH perm l) :=
  ⟨ZMod.val_lt _, ZMod.val_lt _, ZMod.val_lt _, ZMod.val_lt _⟩

theorem spongeRO_dummyNull_canonical (perm : St p → St p) (u : List ℕ) :
    DigestCanonical p ((spongeRO perm).dummyNull u) := by
  rw [spongeRO_dummyNull]; exact spongeH_canonical perm _

theorem map_valDigest_castDigest {ds : List Digest} (h : ∀ d ∈ ds, DigestCanonical p d) :
    (ds.map (castDigest (p := p))).map valDigest = ds := by
  rw [List.map_map]
  conv_rhs => rw [← List.map_id ds]
  apply List.map_congr_left
  intro d hd
  exact valDigest_castDigest (h d hd)

/-- Every pre-permutation nullifier is a dummy replacement or a child's nullifier. -/
theorem mem_of_nullifiersReplaced (ro : RandomOracle) :
    ∀ (ls : List LeafPublic) (us : List (List Felt)) (raw : List Digest),
      nullifiersReplaced ro ls us raw → ∀ d ∈ raw,
        (∃ u ∈ us, d = ro.dummyNull u) ∨ (∃ q ∈ ls, d = q.nullifier)
  | [], [], [], _, _, hd => absurd hd List.not_mem_nil
  | q :: qs, u :: us, n :: ns, ⟨hn, hrest⟩, d, hd => by
      rcases List.mem_cons.1 hd with rfl | hd'
      · rw [hn]
        split
        · exact Or.inl ⟨u, List.mem_cons_self, rfl⟩
        · exact Or.inr ⟨q, List.mem_cons_self, rfl⟩
      · rcases mem_of_nullifiersReplaced ro qs us ns hrest d hd' with ⟨u', hu, e⟩ | ⟨q', hq, e⟩
        · exact Or.inl ⟨u', List.mem_cons_of_mem _ hu, e⟩
        · exact Or.inr ⟨q', List.mem_cons_of_mem _ hq, e⟩
  | [], [], _ :: _, h, _, _ => nomatch h
  | [], _ :: _, _, h, _, _ => nomatch h
  | _ :: _, [], _, h, _, _ => nomatch h
  | _ :: _, _ :: _, [], h, _, _ => nomatch h

/-- The output nullifier region of an `RPrivateBatch` instance over the sponge oracle is
    canonical whenever the children's nullifiers are. -/
theorem RPrivateBatch_nullifiers_canonical (perm : St p → St p) {leaves : List LeafPublic}
    {us : List (List Felt)} {out : PrivateBatchOutput}
    (hcanon : ∀ q ∈ leaves, DigestCanonical p q.nullifier)
    (h : RPrivateBatch (spongeRO perm) leaves us out) :
    ∀ d ∈ out.nullifiers, DigestCanonical p d := by
  obtain ⟨raw, hrep, hperm⟩ := h.2.2.1
  intro d hd
  rcases mem_of_nullifiersReplaced _ leaves us raw hrep d (hperm.mem_iff.mp hd) with
    ⟨u, _, rfl⟩ | ⟨q, hq, rfl⟩
  · exact spongeRO_dummyNull_canonical perm u
  · exact hcanon q hq

/-! ### The shared constraint predicate -/

/-- The private-batch wrapper's constraints on a field witness, exactly as `private_batch_val`
    consumes them: boolean switches and dummy flags, the slot decodes, the network output as
    the public nullifier region, the pairwise uniqueness constraints, the fee comparator with
    its accumulator decodes, and the remaining decoded wrapper outputs. -/
structure PrivateBatchConstraints (perm : St p → St p) (rounds : List (List (ZMod p)))
    (rows : List (SlotRow p)) (fee totalIn totalOut : ZMod p) (out : PrivateBatchOutput) :
    Prop where
  hsw : ∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s
  hb : ∀ t ∈ rows, IsBool t.1.isDummy
  hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1
  hdnull : ∀ t ∈ rows, valDigest t.1.dnull = (spongeRO perm).dummyNull t.2.2
  hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier
  hnull : out.nullifiers = (network rounds (rows.map (fun t => t.1.sel))).map valDigest
  hcol : ∀ (i j : ℕ) (hi : i < rows.length) (hj : j < rows.length), i < j →
    ∃ eq : ZMod p, (eq = 1 ↔ rows[i].1.real = rows[j].1.real) ∧
      band (band (bnot rows[i].1.isDummy) (bnot rows[j].1.isDummy)) eq = 0
  hlen : rows.length ≤ 64
  h32 : ∀ q ∈ rows.map (fun t => t.2.1), inRange 32 q.inputAmount ∧
    inRange 32 q.outputAmount1 ∧ inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps
  hfee : fee.val = out.volumeFeeBps
  hin : totalIn.val = maskedInputTotal (rows.map (fun t => t.2.1))
  hout : totalOut.val = maskedOutputTotal (rows.map (fun t => t.2.1))
  hfc : FeeCheck fee totalIn totalOut
  hexits : out.exitSlots = groupExits (maskedChildPairs (rows.map (fun t => t.2.1)))
  hmeta : metadataConsistent (rows.map (fun t => t.2.1)) out
  href : referenceFromFirstReal (rows.map (fun t => t.2.1)) out
  hnum : out.numExitSlots = 2 * rows.length

/-- Soundness, restated over the shared predicate (this is `private_batch_val`). -/
theorem PrivateBatchConstraints.sound {perm : St p → St p} (hpg : goldilocks ≤ p)
    {rounds : List (List (ZMod p))} {rows : List (SlotRow p)} {fee totalIn totalOut : ZMod p}
    {out : PrivateBatchOutput}
    (h : PrivateBatchConstraints perm rounds rows fee totalIn totalOut out) :
    RPrivateBatch (spongeRO perm) (rows.map (fun t => t.2.1)) (rows.map (fun t => t.2.2)) out :=
  private_batch_val perm hpg rounds rows h.hsw h.hb h.hd h.hdnull h.hreal h.hnull h.hcol h.hlen
    h.h32 h.hfee h.hin h.hout h.hfc h.hexits h.hmeta h.href h.hnum

/-! ### Witness construction -/

/-- The honest field slot for one child: flag from the sentinel, both digests cast in. -/
def honestSlot (perm : St p → St p) (q : LeafPublic) (u : List Felt) : NullSlot p :=
  { isDummy := if isDummyPrivateBatch q then 1 else 0
    dnull := castDigest ((spongeRO perm).dummyNull u)
    real := castDigest q.nullifier }

/-- The honest witness rows, one per `(leaf, preimage)` pair. -/
def honestRows (perm : St p → St p) (leaves : List LeafPublic) (us : List (List Felt)) :
    List (SlotRow p) :=
  (leaves.zip us).map fun qu => (honestSlot perm qu.1 qu.2, qu.1, qu.2)

theorem honestSlot_isBool (perm : St p → St p) (q : LeafPublic) (u : List Felt) :
    IsBool (honestSlot (p := p) perm q u).isDummy := by
  unfold honestSlot IsBool
  split <;> simp

theorem honestSlot_flag (perm : St p → St p) (q : LeafPublic) (u : List Felt) :
    (honestSlot (p := p) perm q u).isDummy = 1 ↔ isDummyPrivateBatch q := by
  unfold honestSlot
  split
  · next h => exact ⟨fun _ => h, fun _ => rfl⟩
  · next h => exact ⟨fun h0 => absurd h0 (by simp), fun hq => absurd hq h⟩

/-- The honest slot's selection is the cast of the spec's per-slot nullifier. -/
theorem honestSlot_sel (perm : St p → St p) (q : LeafPublic) (u : List Felt) :
    (honestSlot (p := p) perm q u).sel
      = castDigest (if isDummyPrivateBatch q then (spongeRO perm).dummyNull u else q.nullifier) := by
  funext j
  unfold NullSlot.sel honestSlot
  split
  · simp only [bselect_true rfl]
  · simp only [bselect_false rfl]

theorem honestRows_leaves (perm : St p → St p) {leaves : List LeafPublic} {us : List (List Felt)}
    (hlen : us.length = leaves.length) :
    (honestRows (p := p) perm leaves us).map (fun t => t.2.1) = leaves := by
  unfold honestRows
  rw [List.map_map]
  exact List.map_fst_zip (le_of_eq hlen.symm)

theorem honestRows_us (perm : St p → St p) {leaves : List LeafPublic} {us : List (List Felt)}
    (hlen : us.length = leaves.length) :
    (honestRows (p := p) perm leaves us).map (fun t => t.2.2) = us := by
  unfold honestRows
  rw [List.map_map]
  exact List.map_snd_zip (le_of_eq hlen)

theorem honestRows_length (perm : St p → St p) {leaves : List LeafPublic} {us : List (List Felt)}
    (hlen : us.length = leaves.length) :
    (honestRows (p := p) perm leaves us).length = leaves.length := by
  unfold honestRows
  rw [List.length_map, List.length_zip, hlen, Nat.min_self]

theorem honestRows_sel (perm : St p → St p) (leaves : List LeafPublic) (us : List (List Felt)) :
    (honestRows (p := p) perm leaves us).map (fun t => t.1.sel)
      = (buildNullifiers (spongeRO perm) leaves us).map castDigest := by
  induction leaves generalizing us with
  | nil => cases us <;> rfl
  | cons q qs ih =>
      cases us with
      | nil => rfl
      | cons u us =>
          simp only [honestRows, List.zip_cons_cons, List.map_cons, buildNullifiers]
          rw [honestSlot_sel]
          congr 1
          exact ih us

/-- The pairwise uniqueness constraints are satisfiable from `realNullifiersDistinct`: the
    honest equality flag is `1` exactly on equal real lanes, and two real slots never have
    equal lanes. (The converse of `uniqueness_val_bridge`.) -/
theorem hcol_of_distinct (rows : List (SlotRow p))
    (hb : ∀ t ∈ rows, IsBool t.1.isDummy)
    (hd : ∀ t ∈ rows, t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1)
    (hreal : ∀ t ∈ rows, valDigest t.1.real = t.2.1.nullifier)
    (hdist : realNullifiersDistinct (rows.map (fun t => t.2.1))) :
    ∀ (i j : ℕ) (hi : i < rows.length) (hj : j < rows.length), i < j →
      ∃ eq : ZMod p, (eq = 1 ↔ rows[i].1.real = rows[j].1.real) ∧
        band (band (bnot rows[i].1.isDummy) (bnot rows[j].1.isDummy)) eq = 0 := by
  intro i j hi hj hij
  unfold realNullifiersDistinct at hdist
  rw [List.pairwise_iff_getElem] at hdist
  have hpair := hdist i j (by simpa using hi) (by simpa using hj) hij
  simp only [List.getElem_map] at hpair
  have hmi := List.getElem_mem hi
  have hmj := List.getElem_mem hj
  refine ⟨if rows[i].1.real = rows[j].1.real then 1 else 0, by split <;> simp_all, ?_⟩
  rcases hb _ hmi with hi0 | hi1
  · rcases hb _ hmj with hj0 | hj1
    · have hri : ¬ isDummyPrivateBatch rows[i].2.1 := fun h =>
        absurd ((hd _ hmi).mpr h) (by rw [hi0]; exact zero_ne_one)
      have hrj : ¬ isDummyPrivateBatch rows[j].2.1 := fun h =>
        absurd ((hd _ hmj).mpr h) (by rw [hj0]; exact zero_ne_one)
      have hne : rows[i].1.real ≠ rows[j].1.real := fun h =>
        hpair hri hrj (by rw [← hreal _ hmi, ← hreal _ hmj, h])
      rw [if_neg hne]
      simp [band]
    · simp [band, bnot, hj1]
  · simp [band, bnot, hi1]

/-! ### The assumptions record and the completeness theorem -/

/-- What an honest prover must arrange, beyond `RPrivateBatch` itself, for the wrapper
    constraints to be satisfiable. Each field is a real operational requirement:
    * `modulus` — the field is (at least) Goldilocks; the fee no-wrap bounds are stated
      against `goldilocks`;
    * `leafCap` — at most `MAX_PROOF_COUNT = 64` children;
    * `uslen` — one dummy-nullifier preimage per slot;
    * `ranges32` — each child's amounts and fee are 32-bit (guaranteed for children with
      accepted leaf proofs by `Rleaf_ranges`);
    * `nullCanon` — each child's nullifier lanes are canonical (`< p`), as hash outputs are;
    * `routable` — the odd-even switch network can route the pre-permutation list into the
      claimed output order. `nullsPerm` says the two lists are a permutation; that the
      `n`-round odd-even transposition network realizes *every* permutation (the
      `permutation_switches` witness in `common/src/gadgets.rs`) is the sorting-network
      theorem, left here as the explicit hypothesis. -/
structure CompletenessAssumptions (perm : St p → St p) (leaves : List LeafPublic)
    (us : List (List Felt)) (out : PrivateBatchOutput) : Prop where
  modulus : goldilocks ≤ p
  leafCap : leaves.length ≤ 64
  uslen : us.length = leaves.length
  ranges32 : ∀ q ∈ leaves, inRange 32 q.inputAmount ∧ inRange 32 q.outputAmount1 ∧
    inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps
  nullCanon : ∀ q ∈ leaves, DigestCanonical p q.nullifier
  routable : ∃ rounds : List (List (ZMod p)), (∀ ss ∈ rounds, ∀ s ∈ ss, IsBool s) ∧
    network rounds ((buildNullifiers (spongeRO perm) leaves us).map castDigest)
      = out.nullifiers.map castDigest

omit [Fact p.Prime] in
/-- The three fee-comparator wires, cast in from the spec totals. -/
theorem fee_wires_canonical (hpg : goldilocks ≤ p) {leaves : List LeafPublic}
    {out : PrivateBatchOutput} (hlen : leaves.length ≤ 64)
    (h32 : ∀ q ∈ leaves, inRange 32 q.inputAmount ∧ inRange 32 q.outputAmount1 ∧
      inRange 32 q.outputAmount2 ∧ inRange 32 q.volumeFeeBps)
    (href : referenceFromFirstReal leaves out) :
    out.volumeFeeBps < p ∧ maskedInputTotal leaves < p ∧ maskedOutputTotal leaves < p := by
  have hfee : out.volumeFeeBps < 2 ^ 32 :=
    WormholeSpec.referenceFromFirstReal_volumeFeeBps_lt (by decide) href
      (fun q hq => (h32 q hq).2.2.2)
  have hin : maskedInputTotal leaves ≤ 64 * 2 ^ 32 :=
    Nat.le_trans (WormholeSpec.maskedInputTotal_le_rawInputTotal leaves)
      (Nat.le_trans (WormholeSpec.rawInputTotal_le_linear (fun q hq => Nat.le_of_lt (h32 q hq).1))
        (Nat.mul_le_mul_right _ hlen))
  have hout : maskedOutputTotal leaves ≤ 64 * (2 * 2 ^ 32) :=
    Nat.le_trans (WormholeSpec.maskedOutputTotal_le_rawOutputTotal leaves)
      (Nat.le_trans (WormholeSpec.rawOutputTotal_le_linear
          (fun q hq => ⟨Nat.le_of_lt (h32 q hq).2.1, Nat.le_of_lt (h32 q hq).2.2.1⟩))
        (Nat.mul_le_mul_right _ hlen))
  refine ⟨?_, ?_, ?_⟩
  · exact Nat.lt_of_lt_of_le hfee (le_trans (by decide) hpg)
  · exact Nat.lt_of_le_of_lt hin (Nat.lt_of_lt_of_le (by decide) hpg)
  · exact Nat.lt_of_le_of_lt hout (Nat.lt_of_lt_of_le (by decide) hpg)

/-- **Completeness.** Every `RPrivateBatch` instance over the sponge oracle, under the
    `CompletenessAssumptions`, is realized by a field witness satisfying the wrapper
    constraints — the honest rows, the routing switches, and the cast-in fee wires. -/
theorem private_batch_complete (perm : St p → St p) {leaves : List LeafPublic}
    {us : List (List Felt)} {out : PrivateBatchOutput}
    (ha : CompletenessAssumptions perm leaves us out)
    (h : RPrivateBatch (spongeRO perm) leaves us out) :
    ∃ (rounds : List (List (ZMod p))) (rows : List (SlotRow p)) (fee totalIn totalOut : ZMod p),
      rows.map (fun t => t.2.1) = leaves ∧ rows.map (fun t => t.2.2) = us ∧
      PrivateBatchConstraints perm rounds rows fee totalIn totalOut out := by
  obtain ⟨rounds, hsw, hroute⟩ := ha.routable
  have hL := honestRows_leaves (p := p) perm ha.uslen
  have hU := honestRows_us (p := p) perm ha.uslen
  have hmem : ∀ t ∈ honestRows (p := p) perm leaves us,
      t.1 = honestSlot perm t.2.1 t.2.2 ∧ t.2.1 ∈ leaves := by
    intro t ht
    obtain ⟨⟨q, u⟩, hqu, rfl⟩ := List.mem_map.1 ht
    exact ⟨rfl, (List.of_mem_zip hqu).1⟩
  obtain ⟨hfeeLt, hinLt, houtLt⟩ :=
    fee_wires_canonical ha.modulus ha.leafCap ha.ranges32 h.2.1
  have hp53 : 2 ^ 53 ≤ p := le_trans (by decide) ha.modulus
  have hb : ∀ t ∈ honestRows (p := p) perm leaves us, IsBool t.1.isDummy := by
    intro t ht; rw [(hmem t ht).1]; exact honestSlot_isBool perm _ _
  have hd : ∀ t ∈ honestRows (p := p) perm leaves us,
      t.1.isDummy = 1 ↔ isDummyPrivateBatch t.2.1 := by
    intro t ht; rw [(hmem t ht).1]; exact honestSlot_flag perm _ _
  have hreal : ∀ t ∈ honestRows (p := p) perm leaves us, valDigest t.1.real = t.2.1.nullifier := by
    intro t ht; rw [(hmem t ht).1]; exact valDigest_castDigest (ha.nullCanon _ (hmem t ht).2)
  have hfeeOk := h.2.2.2.2.1
  have hfeeVal : ((out.volumeFeeBps : ℕ) : ZMod p).val = out.volumeFeeBps :=
    ZMod.val_natCast_of_lt hfeeLt
  have hinVal : ((maskedInputTotal leaves : ℕ) : ZMod p).val = maskedInputTotal leaves :=
    ZMod.val_natCast_of_lt hinLt
  have houtVal : ((maskedOutputTotal leaves : ℕ) : ZMod p).val = maskedOutputTotal leaves :=
    ZMod.val_natCast_of_lt houtLt
  refine ⟨rounds, honestRows perm leaves us, (out.volumeFeeBps : ZMod p),
    (maskedInputTotal leaves : ZMod p), (maskedOutputTotal leaves : ZMod p), hL, hU, ?_⟩
  refine
    { hsw := hsw
      hb := hb
      hd := hd
      hdnull := ?_
      hreal := hreal
      hnull := ?_
      hcol := hcol_of_distinct _ hb hd hreal (by rw [hL]; exact h.2.2.2.2.2.2.1)
      hlen := by rw [honestRows_length perm ha.uslen]; exact ha.leafCap
      h32 := by rw [hL]; exact ha.ranges32
      hfee := hfeeVal
      hin := by rw [hL]; exact hinVal
      hout := by rw [hL]; exact houtVal
      hfc := ?_
      hexits := by rw [hL]; exact h.2.2.2.2.2.1
      hmeta := by rw [hL]; exact h.1
      href := by rw [hL]; exact h.2.1
      hnum := by rw [honestRows_length perm ha.uslen]; exact h.2.2.2.2.2.2.2 }
  · intro t ht
    rw [(hmem t ht).1]
    exact valDigest_castDigest (spongeRO_dummyNull_canonical perm _)
  · rw [honestRows_sel, hroute,
      map_valDigest_castDigest (RPrivateBatch_nullifiers_canonical perm ha.nullCanon h)]
  · have hass : FeeCheck.Assumptions ((out.volumeFeeBps : ℕ) : ZMod p)
        ((maskedInputTotal leaves : ℕ) : ZMod p) ((maskedOutputTotal leaves : ℕ) : ZMod p) :=
      { fee32 := by
          rw [hfeeVal]
          exact WormholeSpec.referenceFromFirstReal_volumeFeeBps_lt (by decide) h.2.1
            (fun q hq => (ha.ranges32 q hq).2.2.2)
        rhsBound := by
          rw [hinVal, hfeeVal]
          have := WormholeSpec.privateBatchFeeRhs_lt_two_pow_52 (out := out) ha.leafCap
            (fun q hq => (ha.ranges32 q hq).1)
          unfold WormholeSpec.feeDenominator at this
          exact this
        lhsBound := by
          rw [houtVal]
          have := WormholeSpec.privateBatchFeeLhs_add_two_pow_52_lt_modulus ha.leafCap
            (fun q hq => ⟨(ha.ranges32 q hq).2.1, (ha.ranges32 q hq).2.2.1⟩)
          unfold WormholeSpec.feeDenominator at this
          exact Nat.lt_of_lt_of_le this ha.modulus }
    apply feeCheck_complete hp53 hass
    · rw [hfeeVal]; exact hfeeOk.1
    · rw [houtVal, hinVal, hfeeVal]
      have := hfeeOk.2
      unfold WormholeSpec.feeDenominator at this
      exact this

/-- Soundness and completeness together: over the honest witness shape, the wrapper
    constraints are satisfiable exactly when the spec relation holds. -/
theorem private_batch_iff (perm : St p → St p) {leaves : List LeafPublic}
    {us : List (List Felt)} {out : PrivateBatchOutput}
    (ha : CompletenessAssumptions perm leaves us out) :
    RPrivateBatch (spongeRO perm) leaves us out ↔
      ∃ (rounds : List (List (ZMod p))) (rows : List (SlotRow p)) (fee totalIn totalOut : ZMod p),
        rows.map (fun t => t.2.1) = leaves ∧ rows.map (fun t => t.2.2) = us ∧
        PrivateBatchConstraints perm rounds rows fee totalIn totalOut out := by
  constructor
  · exact private_batch_complete perm ha
  · rintro ⟨rounds, rows, fee, totalIn, totalOut, hL, hU, hc⟩
    have := hc.sound ha.modulus
    rwa [hL, hU] at this

end Plonky2Bridge
