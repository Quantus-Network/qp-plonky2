/-
  T1 — booleanity, the logical connectives, the multiplexers, and `is_equal`.

  These are the gadgets the aggregation wrapper logic (`R_L0`) is built from: dummy
  flags (`assert_bool`), flag combination (`and`/`or`/`not`), dummy-nullifier
  replacement and value selection (`select`/`_if`), and nullifier/exit matching
  (`is_equal`). All are stated over `ZMod p` for a prime field `[Fact p.Prime]`;
  primality is what turns the booleanity constraint `b² = b` into the genuine
  two-point set `{0,1}` (integral domain) and lets `is_equal` use field inverses.

  Rust references (qp-plonky2, `gadgets/`):
  * `assert_bool`        `range_check.rs:66-70`   `b·b − b = 0`
  * `not`                `arithmetic.rs:345-349`   `1 − b`
  * `and`                `arithmetic.rs:352-354`   `a · b`
  * `or`                 `arithmetic.rs:356-360`   `a + b − a·b`
  * `_if`                `arithmetic.rs:362-367`   `b·x + (1−b)·y`
  * `select`             `select.rs:33-36`         `b·(x−y) + y`
  * `is_equal`           `arithmetic.rs:370-388`   two constraints + `equal, inv`
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Tactic.Ring
import Plonky2Spec.Basic

namespace Plonky2Spec

variable {p : ℕ} [Fact p.Prime]

/-! ### Booleanity -/

/-- A wire is boolean iff its value is `0` or `1`. -/
def IsBool (b : ZMod p) : Prop := b = 0 ∨ b = 1

/-- `assert_bool` (range_check.rs:66-70) emits `b·b − b = 0`. Over a prime field this
    constraint is *exactly* booleanity (`ZMod p` is an integral domain). -/
theorem isBool_iff_assertBool {b : ZMod p} : IsBool b ↔ b * b - b = 0 := by
  have hfac : b * b - b = b * (b - 1) := by ring
  rw [IsBool, hfac, mul_eq_zero, sub_eq_zero]

/-! ### Logical connectives (as the field polynomials the gadgets emit) -/

def bnot (b : ZMod p) : ZMod p := 1 - b
def band (a b : ZMod p) : ZMod p := a * b
def bor (a b : ZMod p) : ZMod p := a + b - a * b

/-- Booleanity is preserved by the connectives. -/
theorem bnot_isBool {b : ZMod p} (h : IsBool b) : IsBool (bnot b) := by
  rcases h with h | h <;> subst h
  · exact Or.inr (by simp [bnot])
  · exact Or.inl (by simp [bnot])

theorem band_isBool {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) : IsBool (band a b) := by
  rcases ha with ha | ha <;> rcases hb with hb | hb <;> subst ha <;> subst hb <;>
    simp [band, IsBool]

theorem bor_isBool {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) : IsBool (bor a b) := by
  rcases ha with ha | ha <;> rcases hb with hb | hb <;> subst ha <;> subst hb <;>
    simp [bor, IsBool]

/-- Truth tables (given boolean inputs). -/
theorem bnot_eq_one {b : ZMod p} : bnot b = 1 ↔ b = 0 := by
  rw [bnot, sub_eq_self]

theorem band_eq_one {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) :
    band a b = 1 ↔ a = 1 ∧ b = 1 := by
  rcases ha with ha | ha <;> rcases hb with hb | hb <;> subst ha <;> subst hb <;>
    simp [band, zero_ne_one]

theorem bor_eq_one {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) :
    bor a b = 1 ↔ a = 1 ∨ b = 1 := by
  rcases ha with ha | ha <;> rcases hb with hb | hb <;> subst ha <;> subst hb <;>
    simp [bor, zero_ne_one]

/-! ### XOR and bit-equality (the comparator's `not_xor`, gadgets.rs:83-105) -/

/-- `xor(a, b)` (gadgets.rs:93-105): `a + b − 2·a·b`. -/
def bxor (a b : ZMod p) : ZMod p := a + b - 2 * (a * b)

/-- `¬(a ⊕ b)`: the `not_xor` flag `is_const_less_than` threads as `eq` (gadgets.rs:56-57);
    `1` iff the (boolean) bits agree. -/
def bxnor (a b : ZMod p) : ZMod p := bnot (bxor a b)

theorem bxor_isBool {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) : IsBool (bxor a b) := by
  unfold IsBool bxor
  rcases ha with h | h <;> rcases hb with h' | h' <;> subst h <;> subst h'
  · left; ring
  · right; ring
  · right; ring
  · left; ring

theorem bxnor_isBool {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) : IsBool (bxnor a b) :=
  bnot_isBool (bxor_isBool ha hb)

/-- `bxnor` is bit-equality: `1` iff the bits are equal. Uses `(a−b)² = a+b−2ab` on
    booleans (`a² = a`) and that a prime field has no zero divisors. -/
theorem bxnor_eq_one {a b : ZMod p} (ha : IsBool a) (hb : IsBool b) :
    bxnor a b = 1 ↔ a = b := by
  have ha2 : a * a = a := by rcases ha with h | h <;> subst h <;> ring
  have hb2 : b * b = b := by rcases hb with h | h <;> subst h <;> ring
  have hsq : a + b - 2 * (a * b) = (a - b) * (a - b) := by
    have h0 : (a - b) * (a - b) = a * a - 2 * (a * b) + b * b := by ring
    rw [h0, ha2, hb2]; ring
  rw [bxnor, bnot_eq_one, bxor, hsq, mul_self_eq_zero]
  exact sub_eq_zero

/-! ### Multiplexers: `_if` and `select` -/

/-- `_if(b, x, y)` (arithmetic.rs:362-367): `b·x + (1−b)·y`. (`bif` is a Lean
    keyword, so the model is named `bmux`.) -/
def bmux (b x y : ZMod p) : ZMod p := b * x + (1 - b) * y

/-- `select(b, x, y)` (select.rs:33-36): `b·(x−y) + y`. -/
def bselect (b x y : ZMod p) : ZMod p := b * (x - y) + y

/-- The two multiplexers compute the same polynomial. -/
theorem bselect_eq_bmux (b x y : ZMod p) : bselect b x y = bmux b x y := by
  rw [bselect, bmux]; ring

theorem bmux_true {b x y : ZMod p} (h : b = 1) : bmux b x y = x := by rw [bmux, h]; ring
theorem bmux_false {b x y : ZMod p} (h : b = 0) : bmux b x y = y := by rw [bmux, h]; ring
theorem bselect_true {b x y : ZMod p} (h : b = 1) : bselect b x y = x := by rw [bselect, h]; ring
theorem bselect_false {b x y : ZMod p} (h : b = 0) : bselect b x y = y := by rw [bselect, h]; ring

/-! ### `is_equal` -/

/-- The two constraints `is_equal(x, y)` emits (arithmetic.rs:378-385), with the
    witnessed result bool `equal` and auxiliary inverse `inv`:
    * `c1 : equal · (x − y) = 0`            (`not_equal_check`), and
    * `c2 : (x − y) · inv − (1 − equal) = 0` (`equal_check`). -/
structure IsEqual (x y equal inv : ZMod p) : Prop where
  c1 : equal * (x - y) = 0
  c2 : (x - y) * inv - (1 - equal) = 0

/-- **Soundness, part 1:** the constraints force `equal` to be boolean, even though
    it is witnessed by `add_virtual_bool_target_unsafe` (no separate `assert_bool`). -/
theorem isEqual_isBool {x y equal inv : ZMod p} (h : IsEqual x y equal inv) :
    IsBool equal := by
  by_cases hxy : x = y
  · refine Or.inr ?_
    have hc2 := h.c2
    rw [sub_eq_zero.mpr hxy, zero_mul, zero_sub, neg_eq_zero, sub_eq_zero] at hc2
    exact hc2.symm
  · refine Or.inl ?_
    rcases mul_eq_zero.mp h.c1 with he | he
    · exact he
    · exact absurd he (sub_ne_zero.mpr hxy)

/-- **Soundness, part 2:** `equal = 1` iff `x = y`. (`⟸` uses `c2`; `⟹` uses `c1`.) -/
theorem isEqual_iff {x y equal inv : ZMod p} (h : IsEqual x y equal inv) :
    equal = 1 ↔ x = y := by
  constructor
  · intro he
    have hc1 := h.c1
    rw [he, one_mul] at hc1
    exact sub_eq_zero.mp hc1
  · intro hxy
    have hc2 := h.c2
    rw [sub_eq_zero.mpr hxy, zero_mul, zero_sub, neg_eq_zero, sub_eq_zero] at hc2
    exact hc2.symm

/-- **Completeness:** for any `x, y` there is an honest witness (`equal, inv`)
    satisfying the constraints and computing the correct boolean — `equal = 1` when
    `x = y` (with `inv = 0`), else `equal = 0` and `inv = (x − y)⁻¹`. -/
theorem isEqual_complete (x y : ZMod p) :
    ∃ equal inv, IsEqual x y equal inv ∧ (equal = 1 ↔ x = y) := by
  by_cases hxy : x = y
  · refine ⟨1, 0, ⟨?_, ?_⟩, ?_⟩
    · rw [sub_eq_zero.mpr hxy]; ring
    · rw [sub_eq_zero.mpr hxy]; ring
    · simp [hxy]
  · refine ⟨0, (x - y)⁻¹, ⟨?_, ?_⟩, ?_⟩
    · ring
    · rw [mul_inv_cancel₀ (sub_ne_zero.mpr hxy)]; ring
    · exact ⟨fun h => absurd h zero_ne_one, fun h => absurd h hxy⟩

end Plonky2Spec
