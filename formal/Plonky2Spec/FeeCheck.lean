/-
  The private-batch aggregate fee comparator (qp-zk-circuits
  `private_batch/circuit/circuit_logic.rs`, "Enforce the fee once over the private
  settlement segment"):

      fee_complement = 10000 − volume_fee_bps_ref
      range_check(fee_complement, 14)
      lhs  = total_output · 10000
      rhs  = total_input  · fee_complement
      diff = rhs − lhs
      range_check(diff, 52)

  The spec relation this realizes is `WormholeSpec.privateBatchFeeOk`:

      fee ≤ 10000  ∧  out · 10000 ≤ in · (10000 − fee)          (over ℕ)

  Two field-vs-ℕ subtleties make the gadget sound, and both are `Assumptions` the
  surrounding circuit must supply (the Zellic-style split):

  * `range_check(10000 − fee, 14)` forces `fee ≤ 10000` only because `fee` is itself
    small: a `fee > 10000` wraps the complement to `p + 10000 − fee`, which the 14-bit
    check rejects *provided* `fee < 2^32` (true: the leaf circuit range-checks
    `volume_fee_bps` to 32 bits, and the reference is a leaf's value or the scan's 0).
  * `range_check(rhs − lhs, 52)` forces `lhs ≤ rhs` by a two-sided argument: an honest
    difference is `≤ rhs < 2^52`, and a dishonest one wraps to `p − (lhs − rhs) ≥ p − lhs`,
    which must land *above* `2^52`, i.e. `lhs + 2^52 < p`. Note `lhs` itself may exceed
    `2^52`; the spec supplies exactly these two bounds
    (`WormholeSpec.privateBatchFeeRhs_lt_two_pow_52`,
    `WormholeSpec.privateBatchFeeLhs_add_two_pow_52_lt_modulus`).

  Both directions are proved: soundness (constraints ⟹ the ℕ inequalities on `.val`)
  and completeness (the ℕ inequalities ⟹ the range checks are satisfiable).
-/
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod
import Plonky2Spec.RangeCheck

namespace Plonky2Spec

variable {p : ℕ} [Fact p.Prime]

/-! ### `.val` of a field subtraction of naturals -/

/-- `a − b` in the field, for canonical `a ≥ b`: no wrap. -/
theorem val_natCast_sub_of_le {a b : ℕ} (ha : a < p) (hab : b ≤ a) :
    ((a : ZMod p) - (b : ZMod p)).val = a - b := by
  rw [← Nat.cast_sub hab, ZMod.val_natCast_of_lt (by omega)]

/-- `a − b` in the field, for canonical `a < b < p`: wraps to `p + a − b`. -/
theorem val_natCast_sub_of_lt {a b : ℕ} (hb : b < p) (hab : a < b) :
    ((a : ZMod p) - (b : ZMod p)).val = p + a - b := by
  have hle : b ≤ p + a := by omega
  have hcast : ((a : ZMod p) - (b : ZMod p)) = ((p + a - b : ℕ) : ZMod p) := by
    rw [Nat.cast_sub hle, Nat.cast_add, ZMod.natCast_self, zero_add]
  rw [hcast, ZMod.val_natCast_of_lt (by omega)]

/-- `.val` of a product with no wrap. -/
theorem val_mul_of_lt {a b : ZMod p} (h : a.val * b.val < p) :
    (a * b).val = a.val * b.val := by
  rw [ZMod.val_mul, Nat.mod_eq_of_lt h]

/-! ### The gadget -/

/-- The basis-point denominator, as the circuit constant `from_canonical_u32(10_000)`. -/
def feeDen : ZMod p := ((10000 : ℕ) : ZMod p)

/-- The two range checks the fee comparator emits. -/
structure FeeCheck (fee totalIn totalOut : ZMod p) : Prop where
  /-- `range_check(10000 − fee, 14)`. -/
  comp : rangeCheck (feeDen - fee) 14
  /-- `range_check(rhs − lhs, 52)` with `rhs = in · (10000 − fee)`, `lhs = out · 10000`. -/
  diff : rangeCheck (totalIn * (feeDen - fee) - totalOut * feeDen) 52

/-- What the *surrounding* circuit guarantees about the comparator's inputs. Each is
    discharged elsewhere: `fee32` by the leaf's `range_check(volume_fee_bps, 32)` through
    the first-real scan; `rhsBound`/`lhsBound` by the spec's ≤64-leaf, 32-bit-amount
    bounds. -/
structure FeeCheck.Assumptions (fee totalIn totalOut : ZMod p) : Prop where
  fee32 : fee.val < 2 ^ 32
  rhsBound : totalIn.val * (10000 - fee.val) < 2 ^ 52
  lhsBound : totalOut.val * 10000 + 2 ^ 52 < p

/-! ### Soundness -/

/-- **Fee-complement soundness.** With `fee < 2^32` (and `p ≥ 2^33`), the 14-bit check on
    `10000 − fee` forces `fee ≤ 10000`: a larger fee wraps the complement to
    `p + 10000 − fee > 2^14`. -/
theorem feeComplement_sound (hp : 2 ^ 33 ≤ p) {fee : ZMod p} (hfee : fee.val < 2 ^ 32)
    (h : rangeCheck (feeDen - fee) 14) : fee.val ≤ 10000 := by
  have h14 : (feeDen - fee).val < 2 ^ 14 :=
    rangeCheck_sound (le_trans (by decide) hp) h
  by_contra hgt
  have hgt' : 10000 < fee.val := Nat.lt_of_not_le hgt
  have hrepr : feeDen - fee = ((10000 : ℕ) : ZMod p) - ((fee.val : ℕ) : ZMod p) := by
    unfold feeDen; rw [ZMod.natCast_zmod_val]
  rw [hrepr, val_natCast_sub_of_lt (ZMod.val_lt fee) hgt'] at h14
  omega

/-- **Difference soundness (ℕ form).** With `R < 2^52` and `L + 2^52 < p`, a satisfied
    52-bit check on the field difference `R − L` forces `L ≤ R`. -/
theorem feeDiff_sound {R L : ℕ} (hR : R < 2 ^ 52) (hL : L + 2 ^ 52 < p)
    (h : rangeCheck ((R : ZMod p) - (L : ZMod p)) 52) : L ≤ R := by
  have h52 : ((R : ZMod p) - (L : ZMod p)).val < 2 ^ 52 := rangeCheck_sound (by omega) h
  by_contra hlt
  have hlt' : R < L := Nat.lt_of_not_le hlt
  rw [val_natCast_sub_of_lt (by omega) hlt'] at h52
  omega

/-- With `fee ≤ 10000` and the no-wrap `Assumptions`, the field difference the circuit
    range-checks is the cast difference of the two ℕ sides. -/
theorem feeCheck_diff_repr (hp : 2 ^ 53 ≤ p) {fee totalIn totalOut : ZMod p}
    (ha : FeeCheck.Assumptions fee totalIn totalOut) (hfee : fee.val ≤ 10000) :
    totalIn * (feeDen - fee) - totalOut * feeDen
      = ((totalIn.val * (10000 - fee.val) : ℕ) : ZMod p)
        - ((totalOut.val * 10000 : ℕ) : ZMod p) := by
  -- `omega` is kept away from goals of the shape `x * 10000` (it recurses on large
  -- right-hand literal coefficients in this toolchain); those steps use explicit lemmas.
  have hR := ha.rhsBound
  have hp52 : 2 ^ 52 ≤ p := le_trans (by decide) hp
  have h10 : 10000 < p := by omega
  have hR' : totalIn.val * (10000 - fee.val) < p := Nat.lt_of_lt_of_le hR hp52
  have hL' : totalOut.val * 10000 < p := Nat.lt_of_le_of_lt (Nat.le_add_right _ _) ha.lhsBound
  have hcomp : (feeDen - fee).val = 10000 - fee.val := by
    have hrepr : feeDen - fee = ((10000 : ℕ) : ZMod p) - ((fee.val : ℕ) : ZMod p) := by
      unfold feeDen; rw [ZMod.natCast_zmod_val]
    rw [hrepr, val_natCast_sub_of_le h10 hfee]
  have hden : (feeDen : ZMod p).val = 10000 := ZMod.val_natCast_of_lt h10
  have hrhs : (totalIn * (feeDen - fee)).val = totalIn.val * (10000 - fee.val) := by
    rw [val_mul_of_lt (lt_of_eq_of_lt (congrArg (totalIn.val * ·) hcomp) hR'), hcomp]
  have hlhs : (totalOut * feeDen).val = totalOut.val * 10000 := by
    rw [val_mul_of_lt (lt_of_eq_of_lt (congrArg (totalOut.val * ·) hden) hL'), hden]
  rw [← hrhs, ← hlhs, ZMod.natCast_zmod_val, ZMod.natCast_zmod_val]

/-- **Fee-comparator soundness.** The two range checks, under the `Assumptions`, enforce
    exactly `WormholeSpec.privateBatchFeeOk` read through `.val`:
    `fee ≤ 10000 ∧ out · 10000 ≤ in · (10000 − fee)`. -/
theorem feeCheck_sound (hp : 2 ^ 53 ≤ p) {fee totalIn totalOut : ZMod p}
    (ha : FeeCheck.Assumptions fee totalIn totalOut) (h : FeeCheck fee totalIn totalOut) :
    fee.val ≤ 10000 ∧ totalOut.val * 10000 ≤ totalIn.val * (10000 - fee.val) := by
  have hp33 : 2 ^ 33 ≤ p := le_trans (by decide) hp
  have hfee := feeComplement_sound hp33 ha.fee32 h.comp
  refine ⟨hfee, ?_⟩
  have hdiff := feeCheck_diff_repr hp ha hfee
  have hd := h.diff
  rw [hdiff] at hd
  exact feeDiff_sound ha.rhsBound ha.lhsBound hd

/-! ### Completeness -/

/-- **Fee-comparator completeness.** If the ℕ inequalities hold (and the no-wrap
    `Assumptions`), both range checks are satisfiable — no honest batch is locked out. -/
theorem feeCheck_complete (hp : 2 ^ 53 ≤ p) {fee totalIn totalOut : ZMod p}
    (ha : FeeCheck.Assumptions fee totalIn totalOut)
    (hfee : fee.val ≤ 10000)
    (hle : totalOut.val * 10000 ≤ totalIn.val * (10000 - fee.val)) :
    FeeCheck fee totalIn totalOut := by
  have hR := ha.rhsBound
  have hp52 : 2 ^ 52 ≤ p := le_trans (by decide) hp
  have h10 : 10000 < p := by omega
  have hcomp : (feeDen - fee).val = 10000 - fee.val := by
    have hrepr : feeDen - fee = ((10000 : ℕ) : ZMod p) - ((fee.val : ℕ) : ZMod p) := by
      unfold feeDen; rw [ZMod.natCast_zmod_val]
    rw [hrepr, val_natCast_sub_of_le h10 hfee]
  have hdval : (totalIn * (feeDen - fee) - totalOut * feeDen).val
      = totalIn.val * (10000 - fee.val) - totalOut.val * 10000 := by
    rw [feeCheck_diff_repr hp ha hfee, val_natCast_sub_of_le (Nat.lt_of_lt_of_le hR hp52) hle]
  exact ⟨rangeCheck_complete (lt_of_eq_of_lt hcomp (by omega)),
    rangeCheck_complete (lt_of_eq_of_lt hdval (Nat.lt_of_le_of_lt (Nat.sub_le _ _) hR))⟩

end Plonky2Spec
