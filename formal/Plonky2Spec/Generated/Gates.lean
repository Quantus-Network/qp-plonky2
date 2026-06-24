/-
  AUTO-GENERATED — do not edit by hand.

  Produced by the `qp-plonky2-constraint-exporter` dev tool, which symbolically
  executes each gate's real `Gate::eval_unfiltered` (over a symbolic field) and
  prints the constraint polynomials it emits. Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints

  Each `def …_c{i}` is the i-th constraint the gate forces to zero, with `w{j}`
  the j-th `local_wires` entry and `c{j}` the j-th `local_constants` entry.
  `Bridges/Bridge.lean` proves each of these equals the corresponding
  hand-written model in `Arithmetic.lean` / `RangeCheck.lean` (by `ring`), so a
  drift between the gate code and the spec breaks `lake build`.
-/
import Mathlib.Algebra.Field.ZMod

namespace Plonky2Spec.Generated

-- Extracted defs carry every gate wire/constant as a parameter, so some are
-- unused in a given constraint; that is intentional and not a code smell.
set_option linter.unusedVariables false

variable {p : ℕ}

/-- `arithmeticGate` constraint #0, extracted verbatim from `arithmeticGate::eval_unfiltered`. -/
def arithmeticGate_c0 (w0 w1 w2 w3 c0 c1 : ZMod p) : ZMod p :=
  (w3 - (((w0 * w1) * c0) + (w2 * c1)))

/-- `baseSum2` constraint #0, extracted verbatim from `baseSum2::eval_unfiltered`. -/
def baseSum2_c0 (w0 w1 w2 : ZMod p) : ZMod p :=
  (((w2 * 2) + w1) - w0)

/-- `baseSum2` constraint #1, extracted verbatim from `baseSum2::eval_unfiltered`. -/
def baseSum2_c1 (w0 w1 w2 : ZMod p) : ZMod p :=
  (w1 * (w1 - 1))

/-- `baseSum2` constraint #2, extracted verbatim from `baseSum2::eval_unfiltered`. -/
def baseSum2_c2 (w0 w1 w2 : ZMod p) : ZMod p :=
  (w2 * (w2 - 1))

end Plonky2Spec.Generated
