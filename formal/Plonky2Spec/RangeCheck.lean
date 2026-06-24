/-
  T2 — the range primitive over the native field `ZMod p`.

  This is the field-native model: wire values are `ZMod p` and the gate constraints
  are stated exactly as `BaseSumGate::eval_unfiltered` emits them. Following Zellic's
  *Formal Verification of a Plonky2 Gate*, the *specification* is phrased over
  `ZMod.val` (the canonical `ℕ` representative), black-boxing the field; the
  positional-arithmetic core lives in `Basic.lean` over `ℕ`.

  Generality (à la Zellic): theorems are stated for an *arbitrary* prime field
  `[Fact p.Prime]` with explicit no-wraparound size bounds (`Bᴸ ≤ p`). This avoids a
  Goldilocks-primality certificate and is strictly more general; the concrete
  modulus `goldilocks` satisfies the bounds with vast margin (`2^48 ≪ p ≈ 2^64`).

  Rust references (qp-plonky2):
  * `BaseSumGate::eval_unfiltered`            `plonky2/src/gates/base_sum.rs:68-81`
  * `range_check` → `split_le` → BaseSumGate  `plonky2/src/gadgets/range_check.rs:21`
  * `enforce_target_less_than_const`          `qp-zk-circuits/common/src/gadgets.rs:66`

  The gate decomposes `sum` into little-endian base-`B` limbs and emits:
    (1) `reduce_with_powers(limbs, B) − sum = 0`          (reconstruction), and
    (2) for each limb: `∏_{j<B}(limb − j) = 0`            (limb ∈ {0,…,B-1}).
  Over a prime field (2) forces the limb into `{0,…,B-1}` because `ZMod p` is an
  integral domain — the step that core Lean could not express and that is now
  *discharged* (`limb_val_lt_of_product`).
-/
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.Algebra.BigOperators.GroupWithZero.Finset
import Plonky2Spec.Basic
import Plonky2Spec.Boolean

namespace Plonky2Spec

-- Several helper lemmas are generic in `p` and don't use primality; the instance
-- is carried by the section `variable` but harmlessly unused in those.
set_option linter.unusedSectionVars false

variable {p : ℕ} [Fact p.Prime]

/-! ### Field reconstruction (the `recon` constraint) -/

/-- Little-endian base-`B` reconstruction over the field, mirroring
    `reduce_with_powers(limbs, B)` (base_sum.rs:71). -/
def reconstructF (B : ℕ) : List (ZMod p) → ZMod p
  | []      => 0
  | d :: ds => d + (B : ZMod p) * reconstructF B ds

@[simp] theorem reconstructF_nil (B : ℕ) : reconstructF (p := p) B [] = 0 := rfl

@[simp] theorem reconstructF_cons (B : ℕ) (d : ZMod p) (ds : List (ZMod p)) :
    reconstructF B (d :: ds) = d + (B : ZMod p) * reconstructF B ds := rfl

/-- The field reconstruction is the cast of the `ℕ` reconstruction of the limb
    values. This is the bridge that lets the `ℕ` place-value lemmas in `Basic.lean`
    govern the field computation. -/
theorem reconstructF_eq_cast (B : ℕ) (limbs : List (ZMod p)) :
    reconstructF B limbs = ((reconstruct B (limbs.map ZMod.val) : ℕ) : ZMod p) := by
  induction limbs with
  | nil => simp [reconstructF]
  | cons d ds ih =>
    have hd : ((d.val : ℕ) : ZMod p) = d := by rw [ZMod.natCast_val, ZMod.cast_id]
    rw [reconstructF_cons, ih, List.map_cons, reconstruct_cons, Nat.cast_add, Nat.cast_mul, hd]

/-! ### The per-limb range constraint and the integral-domain fact -/

/-- The field product range constraint for one limb: `∏_{j<B}(x − j)`, exactly the
    per-limb constraint emitted by `BaseSumGate::eval_unfiltered` (base_sum.rs:73-79). -/
def limbRangeProduct (B : ℕ) (x : ZMod p) : ZMod p :=
  Finset.prod (Finset.range B) (fun j => x - (j : ZMod p))

/-- Over a prime field the product is zero iff `x` equals one of `0,…,B-1`
    (`ZMod p` is an integral domain). -/
theorem limbRangeProduct_eq_zero_iff {B : ℕ} {x : ZMod p} :
    limbRangeProduct B x = 0 ↔ ∃ j, j < B ∧ x = (j : ZMod p) := by
  rw [limbRangeProduct, Finset.prod_eq_zero_iff]
  constructor
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, Finset.mem_range.mp ha, sub_eq_zero.mp hx⟩
  · rintro ⟨j, hj, hx⟩
    exact ⟨j, Finset.mem_range.mpr hj, sub_eq_zero.mpr hx⟩

/-- **The previously-deferred fact, now discharged.** A satisfied degree-`B` product
    constraint forces the limb's canonical value below `B` (needs `B ≤ p`, i.e. the
    digits are genuine field-distinct residues). -/
theorem limb_val_lt_of_product {B : ℕ} (hBp : B ≤ p) {x : ZMod p}
    (h : limbRangeProduct B x = 0) : x.val < B := by
  obtain ⟨j, hj, rfl⟩ := limbRangeProduct_eq_zero_iff.mp h
  rw [ZMod.val_natCast_of_lt (lt_of_lt_of_le hj hBp)]
  exact hj

/-! ### `BaseSumGate<B>` structured constraints (field-native) -/

/-- Structured `BaseSumGate<B>` constraints over `ZMod p` with
    `num_limbs = limbs.length` (base_sum.rs:68-81):
    * `recon` — `reduce_with_powers(limbs, B) = sum`, and
    * `range` — every limb satisfies the degree-`B` product constraint. -/
structure BaseSum (B : ℕ) (sum : ZMod p) (limbs : List (ZMod p)) : Prop where
  recon : reconstructF B limbs = sum
  range : ∀ x ∈ limbs, limbRangeProduct B x = 0

/-- **Soundness of `BaseSumGate<B>`** (field-native): given no field wraparound
    (`Bᴸ ≤ p`, with `B ≤ p`), a satisfied gate forces the sum wire's canonical value
    to be a genuine `num_limbs`-digit base-`B` value, i.e. `sum.val < B ^ num_limbs`. -/
theorem baseSum_sound {B : ℕ} {sum : ZMod p} {limbs : List (ZMod p)}
    (hB : 0 < B) (hBp : B ≤ p) (hwrap : B ^ limbs.length ≤ p)
    (c : BaseSum B sum limbs) : sum.val < B ^ limbs.length := by
  have hlimbs : ∀ d ∈ limbs.map ZMod.val, d < B := by
    intro d hd
    rw [List.mem_map] at hd
    obtain ⟨x, hx, rfl⟩ := hd
    exact limb_val_lt_of_product hBp (c.range x hx)
  have hlen : (limbs.map ZMod.val).length = limbs.length := List.length_map ..
  have hrec_lt : reconstruct B (limbs.map ZMod.val) < B ^ limbs.length := by
    have h := reconstruct_lt hB hlimbs
    rwa [hlen] at h
  have hcast : sum = ((reconstruct B (limbs.map ZMod.val) : ℕ) : ZMod p) := by
    rw [← c.recon, reconstructF_eq_cast]
  rw [hcast, ZMod.val_natCast_of_lt (lt_of_lt_of_le hrec_lt hwrap)]
  exact hrec_lt

/-- For a list of `ℕ`s all below the modulus, casting to `ZMod p` and taking `.val`
    is the identity — the round-trip used by completeness. -/
theorem val_cast_map :
    ∀ {ns : List ℕ}, (∀ n ∈ ns, n < p) →
      (ns.map (fun n => ((n : ℕ) : ZMod p))).map ZMod.val = ns := by
  intro ns
  induction ns with
  | nil => intro _; rfl
  | cons a as ih =>
    intro h
    simp only [List.map_cons]
    rw [ZMod.val_natCast_of_lt (h a (List.mem_cons_self ..)),
        ih (fun n hn => h n (List.mem_cons_of_mem _ hn))]

/-- **Completeness of `BaseSumGate<B>`** (field-native): any field element whose
    canonical value is below `B ^ L` has an `L`-limb witness (the base-`B` digits,
    cast into the field) satisfying the gate constraints. -/
theorem baseSum_complete {B : ℕ} {sum : ZMod p} {L : ℕ}
    (hB : 0 < B) (hBp : B ≤ p) (h : sum.val < B ^ L) :
    ∃ limbs : List (ZMod p), limbs.length = L ∧ BaseSum B sum limbs := by
  refine ⟨(baseDigits B L sum.val).map (fun n => ((n : ℕ) : ZMod p)), ?_, ?_, ?_⟩
  · rw [List.length_map, baseDigits_length]
  · -- recon
    rw [reconstructF_eq_cast,
        val_cast_map (fun n hn => lt_of_lt_of_le (baseDigits_lt hB L sum.val n hn) hBp),
        baseDigits_reconstruct hB L sum.val h, ZMod.natCast_val, ZMod.cast_id]
  · -- range
    intro x hx
    rw [List.mem_map] at hx
    obtain ⟨n, hn, rfl⟩ := hx
    rw [limbRangeProduct_eq_zero_iff]
    exact ⟨n, baseDigits_lt hB L sum.val n hn, rfl⟩

/-! ### `range_check(x, n)` — the `B = 2` instantiation -/

/-- `range_check(x, n)` (range_check.rs:21 → split_le → `BaseSumGate<2>`): there is
    a base-2 decomposition of `x` into `n` bits, each in `{0,1}`. -/
def rangeCheck (x : ZMod p) (n : ℕ) : Prop :=
  ∃ limbs : List (ZMod p), limbs.length = n ∧ BaseSum 2 x limbs

/-- The spec-level range predicate, matching `WormholeSpec.inRange` lifted through
    `ZMod.val`: `x`'s canonical value is a `bits`-bit natural. -/
def inRange (bits : ℕ) (x : ZMod p) : Prop := x.val < 2 ^ bits

/-- **Soundness of `range_check`**: with `2^n ≤ p` (no wraparound — true for the
    Goldilocks field and `n ≤ 48`), a satisfied `range_check(x, n)` proves the
    spec-level bound `x.val < 2^n`. -/
theorem rangeCheck_sound {x : ZMod p} {n : ℕ} (hn : 2 ^ n ≤ p)
    (h : rangeCheck x n) : x.val < 2 ^ n := by
  obtain ⟨limbs, hlen, c⟩ := h
  have h2p : (2 : ℕ) ≤ p := Nat.Prime.two_le Fact.out
  have hwrap : 2 ^ limbs.length ≤ p := by rw [hlen]; exact hn
  have hb := baseSum_sound (by omega) h2p hwrap c
  rwa [hlen] at hb

/-- **Completeness of `range_check`**: every value whose canonical representative is
    `< 2^n` is accepted. (No wraparound bound needed: the reconstructed value is the
    canonical `sum.val < p`.) -/
theorem rangeCheck_complete {x : ZMod p} {n : ℕ}
    (h : x.val < 2 ^ n) : rangeCheck x n :=
  baseSum_complete (by omega) (Nat.Prime.two_le Fact.out) h

/-- **Bridge lemma.** A satisfied `range_check(x, n)` *discharges* the spec's
    `inRange n x` (a theorem, not an assumption). This is the rung that lets
    `qp-zk-circuits/formal` stop *assuming* `inRange 32 …` for the leaf circuit. -/
theorem rangeCheck_implies_inRange {x : ZMod p} {n : ℕ} (hn : 2 ^ n ≤ p)
    (h : rangeCheck x n) : inRange n x := rangeCheck_sound hn h

/-! ### `enforce_target_less_than_const` (the `is_const_less_than` bit comparator)

  `enforce_target_less_than_const(t, U, n_log)` (common/gadgets.rs:66) `split_le`s `t`
  to `n_log` bits (the range core above) and forces the comparator
  `is_const_less_than(U-1, t, n_log)` (gadgets.rs:32) to output `0`. We model that
  comparator (`cmp`) and prove it computes `<` on the bit values, so the gadget's
  *constraints* imply `t < U` — rather than positing the conclusion. -/

/-- Little-endian bit value: the `ℕ` a boolean bit list (least-significant first)
    reconstructs to (`= reduce_with_powers(·, 2)`). -/
def bitsVal (l : List (ZMod p)) : ℕ := reconstruct 2 (l.map ZMod.val)

@[simp] theorem bitsVal_nil : bitsVal ([] : List (ZMod p)) = 0 := rfl

theorem bitsVal_cons (a : ZMod p) (l : List (ZMod p)) :
    bitsVal (a :: l) = a.val + 2 * bitsVal l := by
  simp [bitsVal, List.map_cons, reconstruct_cons]

/-- A boolean field bit decodes to `0`/`1`, both as a field element and as its `.val`. -/
theorem isBool_val_cases {a : ZMod p} (h : IsBool a) :
    (a = 0 ∧ a.val = 0) ∨ (a = 1 ∧ a.val = 1) := by
  have hp1 : 1 < p := Nat.Prime.one_lt Fact.out
  rcases h with h | h <;> subst h
  · exact Or.inl ⟨rfl, by rw [show (0 : ZMod p) = ((0 : ℕ) : ZMod p) by simp,
      ZMod.val_natCast_of_lt (by omega)]⟩
  · exact Or.inr ⟨rfl, by rw [show (1 : ZMod p) = ((1 : ℕ) : ZMod p) by simp,
      ZMod.val_natCast_of_lt hp1]⟩

theorem bit_val_lt_two {a : ZMod p} (h : IsBool a) : a.val < 2 := by
  rcases isBool_val_cases h with ⟨_, hv⟩ | ⟨_, hv⟩ <;> omega

/-- An `n`-bit boolean list's value is `< 2^n` (no field wraparound). -/
theorem bitsVal_lt {l : List (ZMod p)} (hb : ∀ x ∈ l, IsBool x) :
    bitsVal l < 2 ^ l.length := by
  unfold bitsVal
  have hd : ∀ d ∈ l.map ZMod.val, d < 2 := by
    intro d hd
    rw [List.mem_map] at hd
    obtain ⟨x, hx, rfl⟩ := hd
    exact bit_val_lt_two (hb x hx)
  have h := reconstruct_lt (by norm_num : 0 < 2) hd
  rwa [List.length_map] at h

/-- Per-bit "less than" through `.val`: with boolean bits, `¬a ∧ b` is `1` iff
    `a.val < b.val`. -/
theorem ltbit_val {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) :
    band (bnot a) b = 1 ↔ a.val < b.val := by
  rw [band_eq_one (bnot_isBool ha) hb, bnot_eq_one]
  rcases isBool_val_cases ha with ⟨ha0, hav⟩ | ⟨ha1, hav⟩ <;>
    rcases isBool_val_cases hb with ⟨hb0, hbv⟩ | ⟨hb1, hbv⟩ <;> simp_all

/-- Per-bit equality through `.val`. -/
theorem bxnor_val {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) :
    bxnor a b = 1 ↔ a.val = b.val := by
  rw [bxnor_eq_one ha hb]
  rcases isBool_val_cases ha with ⟨ha0, hav⟩ | ⟨ha1, hav⟩ <;>
    rcases isBool_val_cases hb with ⟨hb0, hbv⟩ | ⟨hb1, hbv⟩ <;> simp_all

/-- The MSB-priority bit comparator `is_const_less_than` (gadgets.rs:32-61), as a
    recursion over the bit lists (least-significant first; computes the same `lt`/`eq`
    as the gadget's MSB-first scan). Each step is the gadget's per-bit logic:
    `lt' = lt ∨ (eq ∧ ¬a ∧ b)` and `eq' = eq ∧ ¬(a ⊕ b)`. -/
def cmp : List (ZMod p) → List (ZMod p) → ZMod p × ZMod p
  | a :: xs, b :: ys =>
      (bor (cmp xs ys).1 (band (cmp xs ys).2 (band (bnot a) b)),
       band (cmp xs ys).2 (bxnor a b))
  | _, _ => (0, 1)

theorem cmp_fst (a b : ZMod p) (xs ys : List (ZMod p)) :
    (cmp (a :: xs) (b :: ys)).1
      = bor (cmp xs ys).1 (band (cmp xs ys).2 (band (bnot a) b)) := rfl

theorem cmp_snd (a b : ZMod p) (xs ys : List (ZMod p)) :
    (cmp (a :: xs) (b :: ys)).2 = band (cmp xs ys).2 (bxnor a b) := rfl

theorem cmp_isBool : ∀ (xs ys : List (ZMod p)),
    (∀ x ∈ xs, IsBool x) → (∀ y ∈ ys, IsBool y) →
    IsBool (cmp xs ys).1 ∧ IsBool (cmp xs ys).2 := by
  intro xs
  induction xs with
  | nil => intro ys _ _; cases ys <;> exact ⟨Or.inl rfl, Or.inr rfl⟩
  | cons a xs ih =>
    intro ys ha hb
    cases ys with
    | nil => exact ⟨Or.inl rfl, Or.inr rfl⟩
    | cons b ys =>
      have hatl : ∀ x ∈ xs, IsBool x := fun x hx => ha x (List.mem_cons_of_mem _ hx)
      have hbtl : ∀ y ∈ ys, IsBool y := fun y hy => hb y (List.mem_cons_of_mem _ hy)
      obtain ⟨hlt, heq⟩ := ih ys hatl hbtl
      have ha0 : IsBool a := ha a List.mem_cons_self
      have hb0 : IsBool b := hb b List.mem_cons_self
      rw [cmp_fst, cmp_snd]
      exact ⟨bor_isBool hlt (band_isBool heq (band_isBool (bnot_isBool ha0) hb0)),
             band_isBool heq (bxnor_isBool ha0 hb0)⟩

/-- **Comparator correctness.** On equal-length boolean bit lists, `cmp` computes the
    value comparison: `lt = 1 ↔ as < bs` and `eq = 1 ↔ as = bs` (on `.val`). -/
theorem cmp_spec : ∀ (xs ys : List (ZMod p)), xs.length = ys.length →
    (∀ x ∈ xs, IsBool x) → (∀ y ∈ ys, IsBool y) →
    ((cmp xs ys).1 = 1 ↔ bitsVal xs < bitsVal ys) ∧
    ((cmp xs ys).2 = 1 ↔ bitsVal xs = bitsVal ys) := by
  intro xs
  induction xs with
  | nil =>
    intro ys hlen _ _
    cases ys with
    | nil =>
      have h1 : (cmp ([] : List (ZMod p)) []).1 = 0 := rfl
      have h2 : (cmp ([] : List (ZMod p)) []).2 = 1 := rfl
      rw [h1, h2, bitsVal_nil]
      exact ⟨by simp, by simp⟩
    | cons b ys => simp at hlen
  | cons a xs ih =>
    intro ys hlen ha hb
    cases ys with
    | nil => simp at hlen
    | cons b ys =>
      have hlen' : xs.length = ys.length := by simpa using hlen
      have hatl : ∀ x ∈ xs, IsBool x := fun x hx => ha x (List.mem_cons_of_mem _ hx)
      have hbtl : ∀ y ∈ ys, IsBool y := fun y hy => hb y (List.mem_cons_of_mem _ hy)
      have ha0 : IsBool a := ha a List.mem_cons_self
      have hb0 : IsBool b := hb b List.mem_cons_self
      obtain ⟨ihlt, iheq⟩ := ih ys hlen' hatl hbtl
      obtain ⟨hltB, heqB⟩ := cmp_isBool xs ys hatl hbtl
      have hav := bit_val_lt_two ha0
      have hbv := bit_val_lt_two hb0
      rw [bitsVal_cons, bitsVal_cons]
      refine ⟨?_, ?_⟩
      · rw [cmp_fst, bor_eq_one hltB (band_isBool heqB (band_isBool (bnot_isBool ha0) hb0)),
            band_eq_one heqB (band_isBool (bnot_isBool ha0) hb0), ihlt, iheq, ltbit_val ha0 hb0]
        omega
      · rw [cmp_snd, band_eq_one heqB (bxnor_isBool ha0 hb0), iheq, bxnor_val ha0 hb0]
        omega

/-- The constraints `enforce_target_less_than_const(t, U, nlog)` imposes (gadgets.rs:66):
    `t` `split_le`s into `nlog` bits, the constant `U-1` into the same width, and the
    comparator's overflow flag `((U-1) < t)` is connected to `0`. -/
def EnforceLessThanConst (t U : ZMod p) (nlog : ℕ) : Prop :=
  ∃ tbits ubits : List (ZMod p),
    (∀ x ∈ tbits, IsBool x) ∧ (∀ x ∈ ubits, IsBool x) ∧
      tbits.length = nlog ∧ ubits.length = nlog ∧
      reconstructF 2 tbits = t ∧ bitsVal ubits = U.val - 1 ∧ (cmp ubits tbits).1 = 0

/-- **Soundness of `enforce_target_less_than_const`.** With no field wraparound
    (`2^nlog ≤ p`) and a positive bound, the gadget's constraints *force* `t < U` on
    canonical values: the comparator output being `0` rules out `U-1 < t`. -/
theorem enforceLessThanConst_sound {t U : ZMod p} {nlog : ℕ}
    (hp : 2 ^ nlog ≤ p) (hU : 0 < U.val) (h : EnforceLessThanConst t U nlog) :
    t.val < U.val := by
  obtain ⟨tbits, ubits, tbool, ubool, tlen, ulen, trecon, ubound, overflow⟩ := h
  have htlt : bitsVal tbits < 2 ^ nlog := by
    have h := bitsVal_lt tbool; rwa [tlen] at h
  have htval : t.val = bitsVal tbits := by
    have hcast : t = ((bitsVal tbits : ℕ) : ZMod p) := by
      unfold bitsVal; rw [← trecon, reconstructF_eq_cast]
    rw [hcast, ZMod.val_natCast_of_lt (lt_of_lt_of_le htlt hp)]
  have hlen : ubits.length = tbits.length := by rw [ulen, tlen]
  obtain ⟨hlt_iff, _⟩ := cmp_spec ubits tbits hlen ubool tbool
  have hnot : ¬ bitsVal ubits < bitsVal tbits := by
    intro hh
    have hone : (cmp ubits tbits).1 = 1 := hlt_iff.mpr hh
    rw [overflow] at hone
    exact zero_ne_one hone
  have hle : bitsVal tbits ≤ bitsVal ubits := Nat.not_lt.mp hnot
  rw [ubound] at hle
  rw [htval]; omega

/-- When the exclusive bound fits in `nlog` bits (`U.val ≤ 2^nlog`, the
    `assert_comparison_width` precondition), the enforced `t < U` also yields the spec
    range bound `inRange nlog t`. -/
theorem enforceLessThanConst_inRange {t U : ZMod p} {nlog : ℕ}
    (hp : 2 ^ nlog ≤ p) (hU : 0 < U.val) (hUb : U.val ≤ 2 ^ nlog)
    (h : EnforceLessThanConst t U nlog) : inRange nlog t :=
  lt_of_lt_of_le (enforceLessThanConst_sound hp hU h) hUb

end Plonky2Spec
