/-
  The leaf circuit across the `.val` seam (PLAN.md Step 9).

  `build_leaf_constraints` (wormhole/circuit/src/circuit.rs) is, in gadget calls: the
  unspendable-account double hash `H(H(salt ‖ secret))` pinned to `account_id`; the seven
  32-bit range checks and the leaf hash; the depth bound `enforce_target_less_than_const
  (depth, 17, 5)`; sixteen Merkle levels, each `is_const_less_than(level, depth, 5)`, a
  2-bit `range_check` of the position, four `is_equal`s against `0..3`, the child selects,
  a 16-felt node hash and the `is_active` select; the root binding gated on `is_not_dummy`;
  the block-number range check; the shared-target `connect`s; the dummy detection; and the
  gated nullifier, block-hash and zk-root bindings.

  This module gives each of those pieces its `WormholeSpec` meaning: the bit comparator as
  `ltLoop` (the circuit's own operand order, equal to `RangeCheck.cmp`), one Merkle level
  as `stepUp`, the gated walk as `computeRoot` over the first `depth` levels, the sponge
  calls as `spongeRO`'s `H`, and the dummy flag as `LeafPublic.isDummy`. Digests are
  passed component-wise (`D4 x0 x1 x2 x3`), the shape the generated per-call lemmas have.
  The exporter-generated `Plonky2Bridge/Generated/Leaf.lean` instantiates these on the
  recorded wiring and assembles `Rleaf`.

  The hash lemmas are stated on the spec's concrete `wormholeSalt` / `nullifierSalt`
  (`string_to_felts("wormhole")`, `string_to_felts("~nullif~")`); the exporter checks the
  circuit bakes in the same constants.
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Tauto
import Plonky2Bridge.Complete

namespace Plonky2Bridge.Leaf

open Plonky2Spec (IsBool IsEqual bselect band bnot bor BaseSum bitsVal bitsVal_cons cmp
  cmp_spec cmp_isBool rangeCheck rangeCheck_sound isEqual_iff isEqual_isBool bselect_true
  bselect_false band_eq_one limbRangeProduct_eq_zero_iff reconstructF_eq_cast bitsVal_lt)
open Plonky2Spec.Poseidon2 (St)
open Plonky2Spec.Sponge (spongeHash)
open WormholeSpec (Digest LeafPublic LeafWitness MerkleLevel RandomOracle stepUp computeRoot
  goldilocks wormholeSalt nullifierSalt)

variable {p : ℕ} [Fact p.Prime]

/-! ### Field bounds -/

omit [Fact p.Prime] in
theorem two_pow_32_le (hpg : goldilocks ≤ p) : 2 ^ 32 ≤ p :=
  le_trans (by decide) hpg

omit [Fact p.Prime] in
theorem lt_of_goldilocks (hpg : goldilocks ≤ p) {n : ℕ} (hn : n < 2 ^ 32) : n < p :=
  lt_of_lt_of_le hn (two_pow_32_le hpg)

/-- A 32-bit `range_check` is the spec's `inRange 32` on the canonical value. -/
theorem inRange32 (hpg : goldilocks ≤ p) {x : ZMod p} (h : rangeCheck x 32) :
    WormholeSpec.inRange 32 x.val :=
  rangeCheck_sound (two_pow_32_le hpg) h

theorem pos_lt_four (hpg : goldilocks ≤ p) {x : ZMod p} (h : rangeCheck x 2) : x.val < 4 :=
  rangeCheck_sound (le_trans (by decide) hpg) h

/-- The canonical value of a small numeral. -/
theorem val_lit (hpg : goldilocks ≤ p) (n : ℕ) [n.AtLeastTwo] (hn : n < 2 ^ 32) :
    (OfNat.ofNat n : ZMod p).val = n :=
  ZMod.val_natCast_of_lt (lt_of_goldilocks hpg hn)

theorem val_one' (hpg : goldilocks ≤ p) : (1 : ZMod p).val = 1 :=
  ZMod.val_one_eq_one_mod p ▸ Nat.mod_eq_of_lt (lt_of_goldilocks hpg (by norm_num))

/-! ### Digests -/

/-- A digest read component-wise through `.val`. -/
def D4 (x0 x1 x2 x3 : ZMod p) : Digest := ⟨x0.val, x1.val, x2.val, x3.val⟩

omit [Fact p.Prime] in
theorem D4_toList (x0 x1 x2 x3 : ZMod p) :
    (D4 x0 x1 x2 x3).toList = [x0.val, x1.val, x2.val, x3.val] := rfl

theorem D4_eq_zero_iff {x0 x1 x2 x3 : ZMod p} :
    D4 x0 x1 x2 x3 = Digest.zero ↔ x0 = 0 ∧ x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := by
  simp only [D4, WormholeSpec.Digest.zero, WormholeSpec.Digest.mk.injEq, ZMod.val_eq_zero]

theorem D4_eq_iff {x0 x1 x2 x3 y0 y1 y2 y3 : ZMod p} :
    D4 x0 x1 x2 x3 = D4 y0 y1 y2 y3 ↔ x0 = y0 ∧ x1 = y1 ∧ x2 = y2 ∧ x3 = y3 := by
  simp only [D4, WormholeSpec.Digest.mk.injEq, ZMod.val_injective p |>.eq_iff]

/-- Four `select`s on a flag that is the indicator of `P`. -/
theorem D4_gate {b : ZMod p} {P : Prop} [Decidable P] (hb : b = if P then 1 else 0)
    {x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3 : ZMod p}
    (h0 : z0 = bselect b x0 y0) (h1 : z1 = bselect b x1 y1)
    (h2 : z2 = bselect b x2 y2) (h3 : z3 = bselect b x3 y3) :
    D4 z0 z1 z2 z3 = if P then D4 x0 x1 x2 x3 else D4 y0 y1 y2 y3 := by
  split_ifs with hP
  · rw [if_pos hP] at hb
    rw [h0, h1, h2, h3, bselect_true hb, bselect_true hb, bselect_true hb, bselect_true hb]
  · rw [if_neg hP] at hb
    rw [h0, h1, h2, h3, bselect_false hb, bselect_false hb, bselect_false hb, bselect_false hb]

theorem emb_map_val : ∀ l : List (ZMod p), emb (l.map ZMod.val) = l
  | [] => rfl
  | a :: l => by simp only [List.map_cons, emb_cons, ZMod.natCast_zmod_val, emb_map_val l]

/-- A recorded sponge call (the exporter's four-conjunct shape), read through `.val`, is
    the realized oracle's `H` of the inputs' canonical values. -/
theorem spongeH_of_hash (perm : St p → St p) {inputs : List (ZMod p)} {o0 o1 o2 o3 : ZMod p}
    (h : o0 = spongeHash perm inputs 0 ∧ o1 = spongeHash perm inputs 1 ∧
      o2 = spongeHash perm inputs 2 ∧ o3 = spongeHash perm inputs 3) :
    D4 o0 o1 o2 o3 = spongeH perm (inputs.map ZMod.val) := by
  obtain ⟨h0, h1, h2, h3⟩ := h
  simp only [spongeH, emb_map_val, D4, h0, h1, h2, h3]

/-! ### Bits of a `split_le` row -/

theorem baseSum_bits {x : ZMod p} {bits : List (ZMod p)} (hp : 2 ^ bits.length ≤ p)
    (h : BaseSum 2 x bits) : (∀ b ∈ bits, IsBool b) ∧ x.val = bitsVal bits := by
  have hbool : ∀ b ∈ bits, IsBool b := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := limbRangeProduct_eq_zero_iff.mp (h.range b hb)
    interval_cases j
    · exact Or.inl (by simp)
    · exact Or.inr (by simp)
  refine ⟨hbool, ?_⟩
  have hcast : x = ((bitsVal bits : ℕ) : ZMod p) := by
    unfold bitsVal; rw [← h.recon, reconstructF_eq_cast]
  rw [hcast, ZMod.val_natCast_of_lt (lt_of_lt_of_le (bitsVal_lt hbool) hp)]

/-! ### The `is_const_less_than` bit loop -/

/-- The comparator loop of `is_const_less_than` (common/gadgets.rs) with the circuit's
    operand order, over `(constant bit, target bit)` pairs least-significant first:
    `lt' = lt ∨ ((¬c ∧ b) ∧ eq)`, `eq' = eq ∧ ¬(c + b - 2cb)`, from `(0, 1)`. -/
def ltLoop : List (ZMod p × ZMod p) → ZMod p × ZMod p
  | [] => (0, 1)
  | (c, b) :: rest =>
    let r := ltLoop rest
    (bor r.1 (band (band (bnot c) b) r.2), band r.2 (bnot (c + b - 2 * (c * b))))

theorem ltLoop_eq_cmp : ∀ (cs bs : List (ZMod p)), cs.length = bs.length →
    ltLoop (cs.zip bs) = cmp cs bs
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | c :: cs, b :: bs, h => by
    have ih := ltLoop_eq_cmp cs bs (by simpa using h)
    simp only [List.zip_cons_cons, ltLoop, ih]
    refine Prod.ext ?_ rfl
    show bor _ (band (bnot c) b * (cmp cs bs).2) = bor _ ((cmp cs bs).2 * band (bnot c) b)
    rw [mul_comm]

/-- A constant bit as a field boolean. -/
def cb (x : Bool) : ZMod p := cond x 1 0

theorem cb_isBool (x : Bool) : IsBool (cb (p := p) x) := by
  cases x <;> simp [cb, IsBool]

theorem cb_val (x : Bool) : (cb (p := p) x).val = x.toNat := by
  cases x <;> simp [cb, ZMod.val_one]

/-- `is_const_less_than(c, x, 5)` on a `split_le(x, 5)` row: the loop output is the
    indicator of `c < x`, `c` given by its five bits. -/
theorem isActive5_bits (c0 c1 c2 c3 c4 : Bool) (hpg : goldilocks ≤ p)
    {x b0 b1 b2 b3 b4 : ZMod p} (hb : BaseSum 2 x [b0, b1, b2, b3, b4]) :
    (ltLoop [(cb c0, b0), (cb c1, b1), (cb c2, b2), (cb c3, b3), (cb c4, b4)]).1 =
      if c0.toNat + 2 * c1.toNat + 4 * c2.toNat + 8 * c3.toNat + 16 * c4.toNat < x.val
      then 1 else 0 := by
  obtain ⟨hbool, hval⟩ := baseSum_bits (le_trans (by simp [goldilocks]) hpg) hb
  have hcbool : ∀ y ∈ [cb c0, cb c1, cb c2, cb c3, cb c4], IsBool (p := p) y := by
    simp only [List.mem_cons, List.not_mem_nil, or_false]
    rintro y (rfl | rfl | rfl | rfl | rfl) <;> exact cb_isBool _
  have hz : [(cb c0, b0), (cb c1, b1), (cb c2, b2), (cb c3, b3), (cb c4, b4)] =
      [cb (p := p) c0, cb c1, cb c2, cb c3, cb c4].zip [b0, b1, b2, b3, b4] := rfl
  rw [hz, ltLoop_eq_cmp [cb c0, cb c1, cb c2, cb c3, cb c4] [b0, b1, b2, b3, b4] rfl]
  obtain ⟨hlt, -⟩ :=
    cmp_spec [cb c0, cb c1, cb c2, cb c3, cb c4] [b0, b1, b2, b3, b4] rfl hcbool hbool
  obtain ⟨hltB, -⟩ := cmp_isBool _ _ hcbool hbool
  have hc : bitsVal [cb (p := p) c0, cb c1, cb c2, cb c3, cb c4] =
      c0.toNat + 2 * c1.toNat + 4 * c2.toNat + 8 * c3.toNat + 16 * c4.toNat := by
    simp only [bitsVal_cons, Plonky2Spec.bitsVal_nil, cb_val]; ring
  rw [hc, ← hval] at hlt
  split_ifs with h
  · exact hlt.mpr h
  · rcases hltB with h0 | h1
    · exact h0
    · exact absurd (hlt.mp h1) h

/-- `isActive5_bits` with the constant spelled as a numeral (`hc` is `by decide`). -/
theorem isActive5 (c : ℕ) (c0 c1 c2 c3 c4 : Bool)
    (hc : c0.toNat + 2 * c1.toNat + 4 * c2.toNat + 8 * c3.toNat + 16 * c4.toNat = c)
    (hpg : goldilocks ≤ p) {x b0 b1 b2 b3 b4 : ZMod p} (hb : BaseSum 2 x [b0, b1, b2, b3, b4]) :
    (ltLoop [(cb c0, b0), (cb c1, b1), (cb c2, b2), (cb c3, b3), (cb c4, b4)]).1 =
      if c < x.val then 1 else 0 := by
  rw [isActive5_bits c0 c1 c2 c3 c4 hpg hb, hc]

/-- `enforce_target_less_than_const(x, 17, 5)`: the `is_const_less_than(16, x)` output is
    connected to zero. -/
theorem depth_le_of_loop (hpg : goldilocks ≤ p) {x b0 b1 b2 b3 b4 : ZMod p}
    (hb : BaseSum 2 x [b0, b1, b2, b3, b4])
    (h : (ltLoop [(cb false, b0), (cb false, b1), (cb false, b2), (cb false, b3),
      (cb true, b4)]).1 = 0) : x.val ≤ 16 := by
  rw [isActive5_bits false false false false true hpg hb] at h
  by_contra hlt
  simp only [Bool.toNat_false, Bool.toNat_true] at h
  rw [if_pos (by omega)] at h
  exact one_ne_zero h

/-! ### The dummy flag and the gated bindings -/

theorem isEqual_indicator {x y e i : ZMod p} (h : IsEqual x y e i) :
    e = if x = y then 1 else 0 := by
  split_ifs with hxy
  · exact (isEqual_iff h).mpr hxy
  · rcases isEqual_isBool h with h0 | h1
    · exact h0
    · exact absurd ((isEqual_iff h).mp h1) hxy

/-- `is_dummy = and(and(and(e0, e1), and(e2, e3)), and(e4, e5))` over the `is_equal(·, 0)`
    flags of the block-hash limbs and the two output amounts, and
    `is_not_dummy = 1 - is_dummy`. -/
theorem notDummy_spec {h0 h1 h2 h3 o1 o2 e0 e1 e2 e3 e4 e5 i0 i1 i2 i3 i4 i5 a01 a23 a45 a03
    d nd : ZMod p}
    (q0 : IsEqual h0 0 e0 i0) (q1 : IsEqual h1 0 e1 i1) (q2 : IsEqual h2 0 e2 i2)
    (q3 : IsEqual h3 0 e3 i3) (q4 : IsEqual o1 0 e4 i4) (q5 : IsEqual o2 0 e5 i5)
    (ha01 : a01 = band e0 e1) (ha23 : a23 = band e2 e3) (ha03 : a03 = band a01 a23)
    (ha45 : a45 = band e4 e5) (hd : d = band a03 a45) (hnd : nd = 1 - d) :
    IsBool nd ∧ (nd = 1 ↔ ¬ (D4 h0 h1 h2 h3 = Digest.zero ∧ o1.val = 0 ∧ o2.val = 0)) := by
  have b0 := isEqual_isBool q0
  have b1 := isEqual_isBool q1
  have b2 := isEqual_isBool q2
  have b3 := isEqual_isBool q3
  have b4 := isEqual_isBool q4
  have b5 := isEqual_isBool q5
  have b01 : IsBool a01 := ha01 ▸ Plonky2Spec.band_isBool b0 b1
  have b23 : IsBool a23 := ha23 ▸ Plonky2Spec.band_isBool b2 b3
  have b03 : IsBool a03 := ha03 ▸ Plonky2Spec.band_isBool b01 b23
  have b45 : IsBool a45 := ha45 ▸ Plonky2Spec.band_isBool b4 b5
  have hdb : IsBool d := hd ▸ Plonky2Spec.band_isBool b03 b45
  have hd1 : d = 1 ↔ D4 h0 h1 h2 h3 = Digest.zero ∧ o1.val = 0 ∧ o2.val = 0 := by
    rw [hd, band_eq_one b03 b45, ha03, band_eq_one b01 b23, ha01, ha23, ha45,
      band_eq_one b0 b1, band_eq_one b2 b3, band_eq_one b4 b5,
      isEqual_iff q0, isEqual_iff q1, isEqual_iff q2, isEqual_iff q3, isEqual_iff q4,
      isEqual_iff q5, D4_eq_zero_iff, ZMod.val_eq_zero, ZMod.val_eq_zero]
    tauto
  refine ⟨?_, ?_⟩
  · rw [hnd]; exact Plonky2Spec.bnot_isBool hdb
  · rw [hnd, ← hd1]
    rcases hdb with h | h <;> rw [h] <;> norm_num

/-- `(x - y) * flag` connected to zero, with the flag `1`: `x = y`. -/
theorem bind_of_gate {x y diff prod flag : ZMod p} (hd : diff = x - y) (hm : prod = diff * flag)
    (hz : prod = 0) (hf : flag = 1) : x = y := by
  rw [hf, mul_one, hd] at hm
  exact sub_eq_zero.mp (hm.symm.trans hz)

/-! ### The hash derivations -/

theorem WA_of_hashes (perm : St p → St p) (hpg : goldilocks ≤ p)
    {s0 s1 s2 s3 m0 m1 m2 m3 o0 o1 o2 o3 : ZMod p}
    (hm : m0 = spongeHash perm [1836216183, 1701605224, 1, s0, s1, s2, s3] 0 ∧
      m1 = spongeHash perm [1836216183, 1701605224, 1, s0, s1, s2, s3] 1 ∧
      m2 = spongeHash perm [1836216183, 1701605224, 1, s0, s1, s2, s3] 2 ∧
      m3 = spongeHash perm [1836216183, 1701605224, 1, s0, s1, s2, s3] 3)
    (ho : o0 = spongeHash perm [m0, m1, m2, m3] 0 ∧ o1 = spongeHash perm [m0, m1, m2, m3] 1 ∧
      o2 = spongeHash perm [m0, m1, m2, m3] 2 ∧ o3 = spongeHash perm [m0, m1, m2, m3] 3) :
    D4 o0 o1 o2 o3 = (spongeRO perm).WA (D4 s0 s1 s2 s3) := by
  have hm' := spongeH_of_hash perm hm
  simp only [List.map_cons, List.map_nil, val_lit hpg 1836216183 (by norm_num),
    val_lit hpg 1701605224 (by norm_num), val_one' hpg] at hm'
  rw [RandomOracle.WA, RandomOracle.hh, spongeRO_H, spongeRO_H, wormholeSalt, D4_toList,
    spongeH_of_hash perm ho, List.cons_append, List.cons_append, List.cons_append,
    List.nil_append, ← hm', D4_toList]
  simp only [List.map_cons, List.map_nil]

theorem Null_of_hashes (perm : St p → St p) (hpg : goldilocks ≤ p)
    {s0 s1 s2 s3 t0 t1 m0 m1 m2 m3 o0 o1 o2 o3 : ZMod p}
    (hm : m0 = spongeHash perm [1819635326, 2120640876, 1, s0, s1, s2, s3, t0, t1] 0 ∧
      m1 = spongeHash perm [1819635326, 2120640876, 1, s0, s1, s2, s3, t0, t1] 1 ∧
      m2 = spongeHash perm [1819635326, 2120640876, 1, s0, s1, s2, s3, t0, t1] 2 ∧
      m3 = spongeHash perm [1819635326, 2120640876, 1, s0, s1, s2, s3, t0, t1] 3)
    (ho : o0 = spongeHash perm [m0, m1, m2, m3] 0 ∧ o1 = spongeHash perm [m0, m1, m2, m3] 1 ∧
      o2 = spongeHash perm [m0, m1, m2, m3] 2 ∧ o3 = spongeHash perm [m0, m1, m2, m3] 3) :
    D4 o0 o1 o2 o3 = (spongeRO perm).Null (D4 s0 s1 s2 s3) [t0.val, t1.val] := by
  have hm' := spongeH_of_hash perm hm
  simp only [List.map_cons, List.map_nil, val_lit hpg 1819635326 (by norm_num),
    val_lit hpg 2120640876 (by norm_num), val_one' hpg] at hm'
  rw [RandomOracle.Null, RandomOracle.hh, spongeRO_H, spongeRO_H, nullifierSalt, D4_toList,
    spongeH_of_hash perm ho]
  simp only [List.cons_append, List.nil_append]
  rw [← hm', D4_toList]
  simp only [List.map_cons, List.map_nil]

theorem leafHash_of_hash (perm : St p → St p)
    {t0 t1 t2 t3 c0 c1 asset amt o0 o1 o2 o3 : ZMod p}
    (h : o0 = spongeHash perm [t0, t1, t2, t3, c0, c1, asset, amt] 0 ∧
      o1 = spongeHash perm [t0, t1, t2, t3, c0, c1, asset, amt] 1 ∧
      o2 = spongeHash perm [t0, t1, t2, t3, c0, c1, asset, amt] 2 ∧
      o3 = spongeHash perm [t0, t1, t2, t3, c0, c1, asset, amt] 3) :
    D4 o0 o1 o2 o3 =
      (spongeRO perm).leafHash (D4 t0 t1 t2 t3) [c0.val, c1.val] asset.val amt.val := by
  rw [RandomOracle.leafHash, spongeRO_H, D4_toList, spongeH_of_hash perm h]
  simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append]

/-- `spongeH` of a recorded call whose inputs are spelled out, for the header hash: the
    generated proof states the flattened `headerPreimage` list and closes by `rfl`. -/
theorem H_of_hash (perm : St p → St p) {inputs : List (ZMod p)} {o0 o1 o2 o3 : ZMod p}
    (h : o0 = spongeHash perm inputs 0 ∧ o1 = spongeHash perm inputs 1 ∧
      o2 = spongeHash perm inputs 2 ∧ o3 = spongeHash perm inputs 3) :
    D4 o0 o1 o2 o3 = (spongeRO perm).H (inputs.map ZMod.val) := by
  rw [spongeRO_H, spongeH_of_hash perm h]

/-! ### One Merkle level -/

theorem bselect_one (x y : ZMod p) : bselect 1 x y = x := bselect_true rfl
theorem bselect_zero (x y : ZMod p) : bselect 0 x y = y := bselect_false rfl

/-- With `pos < 4`, `is_equal(pos, k)` is the indicator of `pos.val = k`. -/
theorem posFlag (hpg : goldilocks ≤ p) {pos e i : ZMod p} (k : ℕ) (hk : k < 4)
    (h : IsEqual pos (k : ZMod p) e i) : e = if pos.val = k then 1 else 0 := by
  rw [isEqual_indicator h]
  have hkp : k < p := lt_of_goldilocks hpg (by omega)
  congr 1
  apply propext
  constructor
  · intro hpk; rw [hpk, ZMod.val_natCast_of_lt hkp]
  · intro hv; rw [← hv, ZMod.natCast_zmod_val]

/-- `ZkMerkleProofData::constraints`, one level: the four children are `cur` inserted at
    `pos` among the sorted siblings (`a`, `b`, `c`, `d`, with `m`/`n` the inner selects),
    and their 16-felt hash is `stepUp`. -/
theorem stepUp_of_level (perm : St p → St p) (hpg : goldilocks ≤ p)
    {pos e0 e1 e2 e3 i0 i1 i2 i3 o01 : ZMod p}
    {u0 u1 u2 u3 s00 s01 s02 s03 s10 s11 s12 s13 s20 s21 s22 s23 : ZMod p}
    {a0 a1 a2 a3 m0 m1 m2 m3 b0 b1 b2 b3 n0 n1 n2 n3 c0 c1 c2 c3 d0 d1 d2 d3 h0 h1 h2 h3 :
      ZMod p}
    (hr : rangeCheck pos 2)
    (he0 : IsEqual pos 0 e0 i0) (he1 : IsEqual pos 1 e1 i1)
    (he2 : IsEqual pos 2 e2 i2) (he3 : IsEqual pos 3 e3 i3)
    (ha0 : a0 = bselect e0 u0 s00) (ha1 : a1 = bselect e0 u1 s01)
    (ha2 : a2 = bselect e0 u2 s02) (ha3 : a3 = bselect e0 u3 s03)
    (hm0 : m0 = bselect e0 s00 s10) (hb0 : b0 = bselect e1 u0 m0)
    (hm1 : m1 = bselect e0 s01 s11) (hb1 : b1 = bselect e1 u1 m1)
    (hm2 : m2 = bselect e0 s02 s12) (hb2 : b2 = bselect e1 u2 m2)
    (hm3 : m3 = bselect e0 s03 s13) (hb3 : b3 = bselect e1 u3 m3)
    (h01 : o01 = bor e0 e1)
    (hn0 : n0 = bselect o01 s10 s20) (hc0 : c0 = bselect e2 u0 n0)
    (hn1 : n1 = bselect o01 s11 s21) (hc1 : c1 = bselect e2 u1 n1)
    (hn2 : n2 = bselect o01 s12 s22) (hc2 : c2 = bselect e2 u2 n2)
    (hn3 : n3 = bselect o01 s13 s23) (hc3 : c3 = bselect e2 u3 n3)
    (hd0 : d0 = bselect e3 u0 s20) (hd1 : d1 = bselect e3 u1 s21)
    (hd2 : d2 = bselect e3 u2 s22) (hd3 : d3 = bselect e3 u3 s23)
    (hh : h0 = spongeHash perm [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3] 0 ∧
      h1 = spongeHash perm [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3] 1 ∧
      h2 = spongeHash perm [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3] 2 ∧
      h3 = spongeHash perm [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3] 3) :
    D4 h0 h1 h2 h3 = stepUp (spongeRO perm) (D4 u0 u1 u2 u3)
      ⟨pos.val, D4 s00 s01 s02 s03, D4 s10 s11 s12 s13, D4 s20 s21 s22 s23⟩ := by
  have hpos := pos_lt_four hpg hr
  have f0 := posFlag hpg 0 (by omega) (by simpa using he0)
  have f1 := posFlag hpg 1 (by omega) (by simpa using he1)
  have f2 := posFlag hpg 2 (by omega) (by simpa using he2)
  have f3 := posFlag hpg 3 (by omega) (by simpa using he3)
  rw [spongeH_of_hash perm hh]
  simp only [stepUp, RandomOracle.nodeHash, spongeRO_H, D4_toList]
  subst ha0 ha1 ha2 ha3 hm0 hm1 hm2 hm3 hb0 hb1 hb2 hb3 h01 hn0 hn1 hn2 hn3 hc0 hc1 hc2 hc3
    hd0 hd1 hd2 hd3
  interval_cases hv : pos.val <;>
    simp only [Nat.zero_ne_one, Nat.succ_ne_self, OfNat.ofNat_ne_zero, OfNat.ofNat_ne_one,
      Nat.reduceEqDiff, if_true, if_false] at f0 f1 f2 f3 <;>
    subst f0 f1 f2 f3 <;>
    simp only [bselect_one, bselect_zero, bor, mul_zero, mul_one, add_zero, zero_add, sub_zero,
      List.map_cons, List.map_nil, List.cons_append, List.nil_append]

/-! ### The gated walk -/

/-- The Merkle walk over all `MAX_DEPTH` levels, level `i` applied only when `i < d`. -/
def gatedWalk (ro : RandomOracle) (d : ℕ) : ℕ → Digest → List MerkleLevel → Digest
  | _, cur, [] => cur
  | i, cur, l :: ls => gatedWalk ro d (i + 1) (if i < d then stepUp ro cur l else cur) ls

theorem gatedWalk_nil (ro : RandomOracle) (d i : ℕ) (cur : Digest) :
    gatedWalk ro d i cur [] = cur := rfl

theorem gatedWalk_cons (ro : RandomOracle) (d i : ℕ) (cur : Digest) (l : MerkleLevel)
    (ls : List MerkleLevel) :
    gatedWalk ro d i cur (l :: ls) =
      gatedWalk ro d (i + 1) (if i < d then stepUp ro cur l else cur) ls := rfl

theorem gatedWalk_step (ro : RandomOracle) (d i : ℕ) {cur nxt : Digest} (l : MerkleLevel)
    (ls : List MerkleLevel) (h : nxt = if i < d then stepUp ro cur l else cur) :
    gatedWalk ro d i cur (l :: ls) = gatedWalk ro d (i + 1) nxt ls := by
  rw [gatedWalk_cons, h]

/-- The gated walk from level `i` is `computeRoot` over the next `d - i` levels. -/
theorem gatedWalk_eq (ro : RandomOracle) (d : ℕ) :
    ∀ (ls : List MerkleLevel) (i : ℕ) (cur : Digest),
      gatedWalk ro d i cur ls = computeRoot ro cur (ls.take (d - i))
  | [], _, _ => by simp [gatedWalk_nil, computeRoot]
  | l :: ls, i, cur => by
    rw [gatedWalk_cons, gatedWalk_eq ro d ls (i + 1)]
    by_cases h : i < d
    · rw [if_pos h]
      obtain ⟨k, hk⟩ : ∃ k, d - i = k + 1 := ⟨d - i - 1, by omega⟩
      rw [hk, List.take_succ_cons, show d - (i + 1) = k by omega]
      simp only [computeRoot, List.foldl_cons]
    · rw [if_neg h, show d - i = 0 by omega, show d - (i + 1) = 0 by omega, List.take_zero,
        List.take_zero]

/-- One gated level: the `is_active` select on top of `stepUp_of_level`. -/
theorem gatedStep (ro : RandomOracle) {d i : ℕ} {act : ZMod p} (hact : act = if i < d then 1 else 0)
    {cur nxt : Digest} {c0 c1 c2 c3 h0 h1 h2 h3 n0 n1 n2 n3 : ZMod p}
    (hcur : cur = D4 c0 c1 c2 c3) (hnxt : nxt = D4 n0 n1 n2 n3) {lvl : MerkleLevel}
    (hstep : D4 h0 h1 h2 h3 = stepUp ro cur lvl)
    (hs0 : n0 = bselect act h0 c0) (hs1 : n1 = bselect act h1 c1)
    (hs2 : n2 = bselect act h2 c2) (hs3 : n3 = bselect act h3 c3) :
    nxt = (if i < d then stepUp ro cur lvl else cur) := by
  rw [hnxt, D4_gate hact hs0 hs1 hs2 hs3, hstep, hcur]

/-- Levels beyond `depth` do not matter: `take d` of the sixteen recorded levels. -/
theorem levels_length {ls : List MerkleLevel} {d : ℕ} (hl : ls.length = 16) (hd : d ≤ 16) :
    (ls.take d).length = d := by
  rw [List.length_take, hl]; omega

theorem levels_pos {ls : List MerkleLevel} (h : ∀ lvl ∈ ls, lvl.pos < 4) (d : ℕ) :
    ∀ lvl ∈ ls.take d, lvl.pos < 4 :=
  fun lvl hm => h lvl (List.mem_of_mem_take hm)

end Plonky2Bridge.Leaf
