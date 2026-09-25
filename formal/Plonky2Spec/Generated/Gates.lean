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

/-- `arithmeticGate20` constraint #0, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c0 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w3 - (((w0 * w1) * c0) + (w2 * c1)))

/-- `arithmeticGate20` constraint #1, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c1 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w7 - (((w4 * w5) * c0) + (w6 * c1)))

/-- `arithmeticGate20` constraint #2, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c2 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w11 - (((w8 * w9) * c0) + (w10 * c1)))

/-- `arithmeticGate20` constraint #3, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c3 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w15 - (((w12 * w13) * c0) + (w14 * c1)))

/-- `arithmeticGate20` constraint #4, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c4 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w19 - (((w16 * w17) * c0) + (w18 * c1)))

/-- `arithmeticGate20` constraint #5, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c5 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w23 - (((w20 * w21) * c0) + (w22 * c1)))

/-- `arithmeticGate20` constraint #6, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c6 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w27 - (((w24 * w25) * c0) + (w26 * c1)))

/-- `arithmeticGate20` constraint #7, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c7 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w31 - (((w28 * w29) * c0) + (w30 * c1)))

/-- `arithmeticGate20` constraint #8, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c8 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w35 - (((w32 * w33) * c0) + (w34 * c1)))

/-- `arithmeticGate20` constraint #9, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c9 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w39 - (((w36 * w37) * c0) + (w38 * c1)))

/-- `arithmeticGate20` constraint #10, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c10 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w43 - (((w40 * w41) * c0) + (w42 * c1)))

/-- `arithmeticGate20` constraint #11, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c11 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w47 - (((w44 * w45) * c0) + (w46 * c1)))

/-- `arithmeticGate20` constraint #12, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c12 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w51 - (((w48 * w49) * c0) + (w50 * c1)))

/-- `arithmeticGate20` constraint #13, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c13 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w55 - (((w52 * w53) * c0) + (w54 * c1)))

/-- `arithmeticGate20` constraint #14, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c14 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w59 - (((w56 * w57) * c0) + (w58 * c1)))

/-- `arithmeticGate20` constraint #15, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c15 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w63 - (((w60 * w61) * c0) + (w62 * c1)))

/-- `arithmeticGate20` constraint #16, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c16 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w67 - (((w64 * w65) * c0) + (w66 * c1)))

/-- `arithmeticGate20` constraint #17, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c17 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w71 - (((w68 * w69) * c0) + (w70 * c1)))

/-- `arithmeticGate20` constraint #18, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c18 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w75 - (((w72 * w73) * c0) + (w74 * c1)))

/-- `arithmeticGate20` constraint #19, extracted verbatim from `arithmeticGate20::eval_unfiltered`. -/
def arithmeticGate20_c19 (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77 w78 w79 c0 c1 : ZMod p) : ZMod p :=
  (w79 - (((w76 * w77) * c0) + (w78 * c1)))

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
