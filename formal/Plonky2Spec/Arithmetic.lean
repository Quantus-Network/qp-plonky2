/-
  T0 — the arithmetic gate (`ArithmeticGate`) and the field-arithmetic builder ops.

  `ArithmeticGate::eval_unfiltered` (arithmetic_base.rs:77-93) emits one constraint
  per packed op:  `output − (c0·m0·m1 + c1·addend) = 0`,  i.e. the output wire is
  forced to the weighted multiply-add `c0·m0·m1 + c1·addend`. Every base-field
  builder primitive (`add`, `sub`, `mul`, `mul_add`, `mul_sub`) is just a choice of
  the two constants `c0, c1` and the wiring of `m0, m1, addend` (arithmetic.rs:160-220),
  so each reduces to this single constraint.

  These are field identities (no primality needed): the gate *defines* the output,
  so "soundness" here is the bridge "constraint ⟹ output = <the op>" that lets the
  wrapper-logic model name a wire by the value it must hold.
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Tactic.Ring
import Plonky2Spec.Basic

namespace Plonky2Spec

variable {p : ℕ}

/-- The single constraint emitted by `ArithmeticGate` for one op
    (arithmetic_base.rs:87-89): `output = c0·m0·m1 + c1·addend`. -/
def ArithmeticConstraint (c0 c1 m0 m1 addend output : ZMod p) : Prop :=
  output - (c0 * m0 * m1 + c1 * addend) = 0

/-- The gate constraint is exactly "the output wire equals the multiply-add". -/
theorem arithmetic_iff {c0 c1 m0 m1 addend output : ZMod p} :
    ArithmeticConstraint c0 c1 m0 m1 addend output
      ↔ output = c0 * m0 * m1 + c1 * addend := by
  rw [ArithmeticConstraint, sub_eq_zero]

/-- `add(x, y) = arithmetic(1, 1, x, 1, y)` (arithmetic.rs:193-196) ⟹ `output = x + y`. -/
theorem add_spec {x y output : ZMod p}
    (h : ArithmeticConstraint 1 1 x 1 y output) : output = x + y := by
  rw [arithmetic_iff] at h; rw [h]; ring

/-- `sub(x, y) = arithmetic(1, -1, x, 1, y)` (arithmetic.rs:210-213) ⟹ `output = x - y`. -/
theorem sub_spec {x y output : ZMod p}
    (h : ArithmeticConstraint 1 (-1) x 1 y output) : output = x - y := by
  rw [arithmetic_iff] at h; rw [h]; ring

/-- `mul(x, y) = arithmetic(1, 0, x, y, x)` (arithmetic.rs:217-219) ⟹ `output = x * y`. -/
theorem mul_spec {x y output : ZMod p}
    (h : ArithmeticConstraint 1 0 x y x output) : output = x * y := by
  rw [arithmetic_iff] at h; rw [h]; ring

/-- `mul_add(x, y, z) = arithmetic(1, 1, x, y, z)` ⟹ `output = x * y + z`. -/
theorem mulAdd_spec {x y z output : ZMod p}
    (h : ArithmeticConstraint 1 1 x y z output) : output = x * y + z := by
  rw [arithmetic_iff] at h; rw [h]; ring

/-- `mul_sub(x, y, z) = arithmetic(1, -1, x, y, z)` (arithmetic.rs:188-189) ⟹
    `output = x * y - z`. -/
theorem mulSub_spec {x y z output : ZMod p}
    (h : ArithmeticConstraint 1 (-1) x y z output) : output = x * y - z := by
  rw [arithmetic_iff] at h; rw [h]; ring

end Plonky2Spec
