/-
  AUTO-GENERATED — do not edit by hand.

  The three Poseidon2 permutation primitives, each extracted by running the real
  `plonky2/src/gates/poseidon2.rs` helper over the symbolic field. Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints

  * `sbox7 w0`            = `sbox7_base(w0)`               (the `x^7` S-box)
  * `mdsLight w0..w11`    = `mds_light_base([w0..w11])`    (external light MDS)
  * `internalMix w0..w23` = `internal_mix_base([w0..w11], diag=[w12..w23])`

  `Bridges/Poseidon2Bridge.lean` proves each equals the opaque hand model
  `Plonky2Spec.Poseidon2.{sbox7,mdsLight,internalMix}` by `ring`.
-/
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fin.VecNotation

namespace Plonky2Spec.Generated

set_option linter.unusedVariables false

variable {p : ℕ}

/-- `sbox7_base(w0) = w0^7`, extracted verbatim. -/
def sbox7 (w0 : ZMod p) : ZMod p :=
  ((w0 * (w0 * w0)) * ((w0 * w0) * (w0 * w0)))

/-- `mds_light_base([w0..w11])`, lane outputs, extracted verbatim. -/
def mdsLight (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 : ZMod p) : Fin 12 → ZMod p :=
  ![(((((((w0 + w1) + w2) + w3) + w0) + w1) + w1) + ((((((((w0 + w1) + w2) + w3) + w0) + w1) + w1) + ((((((w4 + w5) + w6) + w7) + w4) + w5) + w5)) + ((((((w8 + w9) + w10) + w11) + w8) + w9) + w9))),
    (((((((w0 + w1) + w2) + w3) + w1) + w2) + w2) + ((((((((w0 + w1) + w2) + w3) + w1) + w2) + w2) + ((((((w4 + w5) + w6) + w7) + w5) + w6) + w6)) + ((((((w8 + w9) + w10) + w11) + w9) + w10) + w10))),
    (((((((w0 + w1) + w2) + w3) + w2) + w3) + w3) + ((((((((w0 + w1) + w2) + w3) + w2) + w3) + w3) + ((((((w4 + w5) + w6) + w7) + w6) + w7) + w7)) + ((((((w8 + w9) + w10) + w11) + w10) + w11) + w11))),
    (((((((w0 + w1) + w2) + w3) + w0) + w0) + w3) + ((((((((w0 + w1) + w2) + w3) + w0) + w0) + w3) + ((((((w4 + w5) + w6) + w7) + w4) + w4) + w7)) + ((((((w8 + w9) + w10) + w11) + w8) + w8) + w11))),
    (((((((w4 + w5) + w6) + w7) + w4) + w5) + w5) + ((((((((w0 + w1) + w2) + w3) + w0) + w1) + w1) + ((((((w4 + w5) + w6) + w7) + w4) + w5) + w5)) + ((((((w8 + w9) + w10) + w11) + w8) + w9) + w9))),
    (((((((w4 + w5) + w6) + w7) + w5) + w6) + w6) + ((((((((w0 + w1) + w2) + w3) + w1) + w2) + w2) + ((((((w4 + w5) + w6) + w7) + w5) + w6) + w6)) + ((((((w8 + w9) + w10) + w11) + w9) + w10) + w10))),
    (((((((w4 + w5) + w6) + w7) + w6) + w7) + w7) + ((((((((w0 + w1) + w2) + w3) + w2) + w3) + w3) + ((((((w4 + w5) + w6) + w7) + w6) + w7) + w7)) + ((((((w8 + w9) + w10) + w11) + w10) + w11) + w11))),
    (((((((w4 + w5) + w6) + w7) + w4) + w4) + w7) + ((((((((w0 + w1) + w2) + w3) + w0) + w0) + w3) + ((((((w4 + w5) + w6) + w7) + w4) + w4) + w7)) + ((((((w8 + w9) + w10) + w11) + w8) + w8) + w11))),
    (((((((w8 + w9) + w10) + w11) + w8) + w9) + w9) + ((((((((w0 + w1) + w2) + w3) + w0) + w1) + w1) + ((((((w4 + w5) + w6) + w7) + w4) + w5) + w5)) + ((((((w8 + w9) + w10) + w11) + w8) + w9) + w9))),
    (((((((w8 + w9) + w10) + w11) + w9) + w10) + w10) + ((((((((w0 + w1) + w2) + w3) + w1) + w2) + w2) + ((((((w4 + w5) + w6) + w7) + w5) + w6) + w6)) + ((((((w8 + w9) + w10) + w11) + w9) + w10) + w10))),
    (((((((w8 + w9) + w10) + w11) + w10) + w11) + w11) + ((((((((w0 + w1) + w2) + w3) + w2) + w3) + w3) + ((((((w4 + w5) + w6) + w7) + w6) + w7) + w7)) + ((((((w8 + w9) + w10) + w11) + w10) + w11) + w11))),
    (((((((w8 + w9) + w10) + w11) + w8) + w8) + w11) + ((((((((w0 + w1) + w2) + w3) + w0) + w0) + w3) + ((((((w4 + w5) + w6) + w7) + w4) + w4) + w7)) + ((((((w8 + w9) + w10) + w11) + w8) + w8) + w11)))]

/-- `internal_mix_base([w0..w11], diag=[w12..w23])`, lane outputs, extracted. -/
def internalMix (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 : ZMod p) : Fin 12 → ZMod p :=
  ![((w12 * w0) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w13 * w1) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w14 * w2) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w15 * w3) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w16 * w4) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w17 * w5) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w18 * w6) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w19 * w7) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w20 * w8) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w21 * w9) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w22 * w10) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11)),
    ((w23 * w11) + (((((((((((w0 + w1) + w2) + w3) + w4) + w5) + w6) + w7) + w8) + w9) + w10) + w11))]

end Plonky2Spec.Generated
