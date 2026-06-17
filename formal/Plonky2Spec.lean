/-
  Formal model of the qp-plonky2 gadget constraint semantics (layer-2 bridge).

  This package formalizes what each constraint-emitting plonky2 *gadget* enforces,
  as a predicate over wire assignments, with `sound`/`complete` lemmas relating the
  predicate to its arithmetic meaning. The spec relations in `qp-zk-circuits/formal`
  (`R_leaf`, `R_L0`, `R_L1`) can then be proved to be *what the circuit constraints
  actually enforce*, rather than merely asserted. See `PLAN.md` for the full plan,
  the trust stack, and the gadget inventory.

  Module map:
  * `Plonky2Spec.Basic`      field/element model and shared constants
  * `Plonky2Spec.RangeCheck` the `BaseSumGate<B>` range primitive (T2): the
                             `range_check(x,n)` gadget and `enforce_target_less_than_const`,
                             with soundness and completeness
  * `Plonky2Spec.Arithmetic` the `ArithmeticGate` (T0): `add/sub/mul/mul_add/mul_sub`
                             reduced to the weighted multiply-add constraint
  * `Plonky2Spec.Boolean`    booleanity, `not/and/or/_if/select`, and the `is_equal`
                             gadget (T1), with soundness and completeness

  Methodology follows Zellic's *Formal Verification of a Plonky2 Gate*: each gadget
  separates an `Assumptions` side-condition (what the surrounding circuit must
  enforce) from the `Spec` it establishes, and we prove both directions —
  soundness (constraints ⇒ spec) and completeness (spec ⇒ a satisfying witness).
-/
import Plonky2Spec.Basic
import Plonky2Spec.RangeCheck
import Plonky2Spec.Arithmetic
import Plonky2Spec.Boolean
