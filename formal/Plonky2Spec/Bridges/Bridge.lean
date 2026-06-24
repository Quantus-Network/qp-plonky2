/-
  2b — pinning the **auto-extracted** gate constraints to the hand models.

  `Generated/Gates.lean` is produced by the `qp-plonky2-constraint-exporter` Rust
  tool, which symbolically executes each gate's real `Gate::eval_unfiltered` and
  prints the constraint polynomials verbatim. This module proves that each
  extracted constraint is *the same polynomial* (up to commutativity, `ring`) as
  the corresponding hand-written model that the soundness/completeness proofs in
  `Arithmetic.lean` / `RangeCheck.lean` are stated against.

  Why this matters: it closes the transcription gap. The hand models are trusted
  by eye against the Rust today; these lemmas make `lake build` fail if the
  extractor (i.e. the gate code) and the hand model ever disagree — so the spec
  cannot silently drift from `eval_unfiltered`. The exporter's own differential
  test (`cargo test -p qp-plonky2-constraint-exporter`) independently checks the
  extraction against the real `GoldilocksField` gate at random points; this is
  the Lean-side half of the same guarantee.

  Correspondence (see `PLAN.md` Step 2b):
  | extracted def (Gates.lean) | hand model                                  |
  |----------------------------|---------------------------------------------|
  | `arithmeticGate_c0`        | `ArithmeticConstraint`  (Arithmetic.lean)   |
  | `baseSum2_c0`              | `reconstructF 2 [·,·]`  (RangeCheck.lean)   |
  | `baseSum2_c1 / _c2`        | `limbRangeProduct 2 ·`  (RangeCheck.lean)   |
-/
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.Generated.Gates
import Plonky2Spec.Arithmetic
import Plonky2Spec.RangeCheck

namespace Plonky2Spec

set_option linter.unusedSectionVars false

variable {p : ℕ} [Fact p.Prime]

/-- The exported `ArithmeticGate` constraint is *exactly* the hand model
    `ArithmeticConstraint` (only the factors are commuted), so the `add/sub/mul/…`
    specs proved against `ArithmeticConstraint` govern the real gate. -/
theorem arithmeticGate_c0_matches (w0 w1 w2 w3 c0 c1 : ZMod p) :
    Generated.arithmeticGate_c0 w0 w1 w2 w3 c0 c1 = 0
      ↔ ArithmeticConstraint c0 c1 w0 w1 w2 w3 := by
  unfold Generated.arithmeticGate_c0 ArithmeticConstraint
  constructor <;> intro h <;> linear_combination h

/-- The exported `BaseSumGate<2>` reconstruction constraint says exactly that the
    sum wire equals the field reconstruction of the two limbs — the `recon`
    component of `BaseSum` in `RangeCheck.lean`. -/
theorem baseSum2_recon_matches (w0 w1 w2 : ZMod p) :
    Generated.baseSum2_c0 w0 w1 w2 = 0 ↔ reconstructF 2 [w1, w2] = w0 := by
  have key : Generated.baseSum2_c0 w0 w1 w2 = reconstructF 2 [w1, w2] - w0 := by
    unfold Generated.baseSum2_c0
    simp only [reconstructF, Nat.cast_ofNat]
    ring
  rw [key, sub_eq_zero]

/-- The exported first-limb range product is exactly `limbRangeProduct 2`. -/
theorem baseSum2_range0_matches (w0 w1 w2 : ZMod p) :
    Generated.baseSum2_c1 w0 w1 w2 = limbRangeProduct 2 w1 := by
  unfold Generated.baseSum2_c1 limbRangeProduct
  rw [Finset.prod_range_succ, Finset.prod_range_succ, Finset.prod_range_zero]
  simp only [Nat.cast_zero, Nat.cast_one]
  ring

/-- The exported second-limb range product is exactly `limbRangeProduct 2`. -/
theorem baseSum2_range1_matches (w0 w1 w2 : ZMod p) :
    Generated.baseSum2_c2 w0 w1 w2 = limbRangeProduct 2 w2 := by
  unfold Generated.baseSum2_c2 limbRangeProduct
  rw [Finset.prod_range_succ, Finset.prod_range_succ, Finset.prod_range_zero]
  simp only [Nat.cast_zero, Nat.cast_one]
  ring

end Plonky2Spec
