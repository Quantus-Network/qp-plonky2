/-
  Field-element model and shared constants for the gadget layer.

  STEP-1 MODELING CHOICE (field representation)
  ---------------------------------------------
  A wire value `Felt` is modeled as `Nat` — the *canonical representative* in
  `[0, p)` of a Goldilocks field element (the same role mathlib's `ZMod.val`
  plays). The gadget facts we prove here are about magnitudes of small scalars
  (range decompositions, place-value bounds), which are stated directly over `Nat`
  with the explicit modulus `goldilocks` where field wraparound matters.

  Why not `ZMod goldilocks` yet? The one genuinely field-algebraic step — that the
  degree-`B` product constraint `∏_{j<B}(limb − j) = 0` forces a limb into
  `{0,…,B-1}` — needs the prime field to be an integral domain (Euclid's lemma),
  which lives in mathlib (`Nat.Prime` is not even in core Lean). That step is
  deferred to the `ZMod p` phase; everything else (the place-value reconstruction
  bound and the witness construction) is proved here in core Lean and is fully
  machine-checked. See the FIELD-FIDELITY NOTE in `RangeCheck.lean`.
-/

namespace Plonky2Spec

/-- A plonky2 wire value: the canonical `Nat` representative of a Goldilocks field
    element (Step-1 placeholder; see module header). -/
abbrev Felt : Type := Nat

/-- The Goldilocks prime `p = 2^64 − 2^32 + 1`, the native field of qp-plonky2
    (`GoldilocksField`). Used as the explicit modulus by the no-wraparound
    side-conditions of the range primitive. -/
def goldilocks : Nat := 0xFFFFFFFF00000001

/-- `inRange bits x` is the spec-level range predicate, matching
    `WormholeSpec.inRange`: the value is the canonical representative of a
    `bits`-bit natural (no field wraparound). The point of this package is to turn
    `inRange` from an *assumption* of the spec into a *consequence* of a satisfied
    `range_check` gadget (see `RangeCheck.rangeCheck_implies_inRange`). -/
def inRange (bits : Nat) (x : Felt) : Prop := x < 2 ^ bits

end Plonky2Spec
