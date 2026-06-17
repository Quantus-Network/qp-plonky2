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

/-! ### `enforce_target_less_than_const` -/

/-- `enforce_target_less_than_const(t, U, n_log)` (common/gadgets.rs:66): enforces
    `t < U` (on canonical values). Internally it `split_le`s `t` to `n_log` bits
    (this step's range core) and forces a bit-wise `is_const_less_than` comparator
    boolean to `0`; the comparator's boolean-logic soundness rests on the T1 gadgets
    and is discharged in Step 2. -/
def enforceLessThanConst (t U : ZMod p) : Prop := t.val < U.val

/-- When the exclusive bound fits in `n_log` bits (`U.val ≤ 2^n_log`, the
    `assert_comparison_width` precondition), the enforced `t < U` also yields the
    spec range bound `inRange n_log t`. -/
theorem enforceLessThanConst_inRange {t U : ZMod p} {nlog : ℕ}
    (h : enforceLessThanConst t U) (hU : U.val ≤ 2 ^ nlog) : inRange nlog t :=
  lt_of_lt_of_le h hU

end Plonky2Spec
