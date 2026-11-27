/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PublicBatchWrapper89`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.Generated.PublicBatchWrapper89.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem publicBatchWrapper89_f9789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8453)) := by
  have c63240 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.1
  have c63241 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.1
  have c63242 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63243 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63244 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63245 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_16
  simp only [← c63240, ← c63241, ← c63242] at e_1020_16
  have e_1020_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_17
  simp only [← c63243, ← c63244, k2, ← c63245] at e_1020_17
  have hr := e_1020_17
  simp only [e_1020_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1020 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8454)) := by
  have c63246 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63247 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63248 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63249 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63250 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63251 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1020_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_18
  simp only [← c63246, ← c63247, ← c63248] at e_1020_18
  have e_1020_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1020 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1020_19
  simp only [← c63249, ← c63250, k2, ← c63251] at e_1020_19
  have hr := e_1020_19
  simp only [e_1020_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8455)) := by
  have c63252 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63253 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63254 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63255 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63256 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63257 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_0
  simp only [← c63252, ← c63253, ← c63254] at e_1021_0
  have e_1021_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_1
  simp only [← c63255, ← c63256, k2, ← c63257] at e_1021_1
  have hr := e_1021_1
  simp only [e_1021_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8456)) := by
  have c63258 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63259 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63260 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63261 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63262 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63263 := (publicBatchWrapper89_copies1976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_2
  simp only [← c63258, ← c63259, ← c63260] at e_1021_2
  have e_1021_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_3
  simp only [← c63261, ← c63262, k2, ← c63263] at e_1021_3
  have hr := e_1021_3
  simp only [e_1021_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8457)) := by
  have c63264 := (publicBatchWrapper89_copies1977 a h).1
  have c63265 := (publicBatchWrapper89_copies1977 a h).2.1
  have c63266 := (publicBatchWrapper89_copies1977 a h).2.2.1
  have c63267 := (publicBatchWrapper89_copies1977 a h).2.2.2.1
  have c63268 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.1
  have c63269 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_4
  simp only [← c63264, ← c63265, ← c63266] at e_1021_4
  have e_1021_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_5
  simp only [← c63267, ← c63268, k2, ← c63269] at e_1021_5
  have hr := e_1021_5
  simp only [e_1021_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8458)) := by
  have c63270 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.1
  have c63271 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.1
  have c63272 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.1
  have c63273 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.1
  have c63274 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63275 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_6
  simp only [← c63270, ← c63271, ← c63272] at e_1021_6
  have e_1021_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_7
  simp only [← c63273, ← c63274, k2, ← c63275] at e_1021_7
  have hr := e_1021_7
  simp only [e_1021_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8459)) := by
  have c63276 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63277 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63278 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63279 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63280 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63281 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_8
  simp only [← c63276, ← c63277, ← c63278] at e_1021_8
  have e_1021_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_9
  simp only [← c63279, ← c63280, k2, ← c63281] at e_1021_9
  have hr := e_1021_9
  simp only [e_1021_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8460)) := by
  have c63282 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63283 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63284 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63285 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63286 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63287 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_10
  simp only [← c63282, ← c63283, ← c63284] at e_1021_10
  have e_1021_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_11
  simp only [← c63285, ← c63286, k2, ← c63287] at e_1021_11
  have hr := e_1021_11
  simp only [e_1021_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8461)) := by
  have c63288 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63289 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63290 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63291 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63292 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63293 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_12
  simp only [← c63288, ← c63289, ← c63290] at e_1021_12
  have e_1021_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_13
  simp only [← c63291, ← c63292, k2, ← c63293] at e_1021_13
  have hr := e_1021_13
  simp only [e_1021_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8462)) := by
  have c63294 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63295 := (publicBatchWrapper89_copies1977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63296 := (publicBatchWrapper89_copies1978 a h).1
  have c63297 := (publicBatchWrapper89_copies1978 a h).2.1
  have c63298 := (publicBatchWrapper89_copies1978 a h).2.2.1
  have c63299 := (publicBatchWrapper89_copies1978 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_14
  simp only [← c63294, ← c63295, ← c63296] at e_1021_14
  have e_1021_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_15
  simp only [← c63297, ← c63298, k2, ← c63299] at e_1021_15
  have hr := e_1021_15
  simp only [e_1021_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8463)) := by
  have c63300 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.1
  have c63301 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.1
  have c63302 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.1
  have c63303 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.1
  have c63304 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.1
  have c63305 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_16
  simp only [← c63300, ← c63301, ← c63302] at e_1021_16
  have e_1021_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_17
  simp only [← c63303, ← c63304, k2, ← c63305] at e_1021_17
  have hr := e_1021_17
  simp only [e_1021_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1021 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8464)) := by
  have c63306 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63307 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63308 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63309 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63310 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63311 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1021_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_18
  simp only [← c63306, ← c63307, ← c63308] at e_1021_18
  have e_1021_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1021 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1021_19
  simp only [← c63309, ← c63310, k2, ← c63311] at e_1021_19
  have hr := e_1021_19
  simp only [e_1021_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8465)) := by
  have c63312 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63313 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63314 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63315 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63316 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63317 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_0
  simp only [← c63312, ← c63313, ← c63314] at e_1022_0
  have e_1022_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_1
  simp only [← c63315, ← c63316, k2, ← c63317] at e_1022_1
  have hr := e_1022_1
  simp only [e_1022_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8466)) := by
  have c63318 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63319 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63320 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63321 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63322 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63323 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_2
  simp only [← c63318, ← c63319, ← c63320] at e_1022_2
  have e_1022_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_3
  simp only [← c63321, ← c63322, k2, ← c63323] at e_1022_3
  have hr := e_1022_3
  simp only [e_1022_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8467)) := by
  have c63324 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63325 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63326 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63327 := (publicBatchWrapper89_copies1978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63328 := (publicBatchWrapper89_copies1979 a h).1
  have c63329 := (publicBatchWrapper89_copies1979 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_4
  simp only [← c63324, ← c63325, ← c63326] at e_1022_4
  have e_1022_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_5
  simp only [← c63327, ← c63328, k2, ← c63329] at e_1022_5
  have hr := e_1022_5
  simp only [e_1022_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8468)) := by
  have c63330 := (publicBatchWrapper89_copies1979 a h).2.2.1
  have c63331 := (publicBatchWrapper89_copies1979 a h).2.2.2.1
  have c63332 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.1
  have c63333 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.1
  have c63334 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.1
  have c63335 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_6
  simp only [← c63330, ← c63331, ← c63332] at e_1022_6
  have e_1022_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_7
  simp only [← c63333, ← c63334, k2, ← c63335] at e_1022_7
  have hr := e_1022_7
  simp only [e_1022_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8469)) := by
  have c63336 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.1
  have c63337 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.1
  have c63338 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63339 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63340 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63341 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_8
  simp only [← c63336, ← c63337, ← c63338] at e_1022_8
  have e_1022_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_9
  simp only [← c63339, ← c63340, k2, ← c63341] at e_1022_9
  have hr := e_1022_9
  simp only [e_1022_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8470)) := by
  have c63342 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63343 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63344 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63345 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63346 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63347 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_10
  simp only [← c63342, ← c63343, ← c63344] at e_1022_10
  have e_1022_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_11
  simp only [← c63345, ← c63346, k2, ← c63347] at e_1022_11
  have hr := e_1022_11
  simp only [e_1022_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8471)) := by
  have c63348 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63349 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63350 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63351 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63352 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63353 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_12
  simp only [← c63348, ← c63349, ← c63350] at e_1022_12
  have e_1022_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_13
  simp only [← c63351, ← c63352, k2, ← c63353] at e_1022_13
  have hr := e_1022_13
  simp only [e_1022_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8472)) := by
  have c63354 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63355 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63356 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63357 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63358 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63359 := (publicBatchWrapper89_copies1979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_14
  simp only [← c63354, ← c63355, ← c63356] at e_1022_14
  have e_1022_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_15
  simp only [← c63357, ← c63358, k2, ← c63359] at e_1022_15
  have hr := e_1022_15
  simp only [e_1022_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8473)) := by
  have c63360 := (publicBatchWrapper89_copies1980 a h).1
  have c63361 := (publicBatchWrapper89_copies1980 a h).2.1
  have c63362 := (publicBatchWrapper89_copies1980 a h).2.2.1
  have c63363 := (publicBatchWrapper89_copies1980 a h).2.2.2.1
  have c63364 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.1
  have c63365 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_16
  simp only [← c63360, ← c63361, ← c63362] at e_1022_16
  have e_1022_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_17
  simp only [← c63363, ← c63364, k2, ← c63365] at e_1022_17
  have hr := e_1022_17
  simp only [e_1022_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1022 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8474)) := by
  have c63366 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.1
  have c63367 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.1
  have c63368 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.1
  have c63369 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.1
  have c63370 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63371 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1022_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_18
  simp only [← c63366, ← c63367, ← c63368] at e_1022_18
  have e_1022_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1022 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1022_19
  simp only [← c63369, ← c63370, k2, ← c63371] at e_1022_19
  have hr := e_1022_19
  simp only [e_1022_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8475)) := by
  have c63372 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63373 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63374 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63375 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63376 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63377 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_0
  simp only [← c63372, ← c63373, ← c63374] at e_1023_0
  have e_1023_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_1
  simp only [← c63375, ← c63376, k2, ← c63377] at e_1023_1
  have hr := e_1023_1
  simp only [e_1023_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8476)) := by
  have c63378 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63379 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63380 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63381 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63382 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63383 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_2
  simp only [← c63378, ← c63379, ← c63380] at e_1023_2
  have e_1023_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_3
  simp only [← c63381, ← c63382, k2, ← c63383] at e_1023_3
  have hr := e_1023_3
  simp only [e_1023_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8477)) := by
  have c63384 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63385 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63386 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63387 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63388 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63389 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_4
  simp only [← c63384, ← c63385, ← c63386] at e_1023_4
  have e_1023_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_5
  simp only [← c63387, ← c63388, k2, ← c63389] at e_1023_5
  have hr := e_1023_5
  simp only [e_1023_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8478)) := by
  have c63390 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63391 := (publicBatchWrapper89_copies1980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63392 := (publicBatchWrapper89_copies1981 a h).1
  have c63393 := (publicBatchWrapper89_copies1981 a h).2.1
  have c63394 := (publicBatchWrapper89_copies1981 a h).2.2.1
  have c63395 := (publicBatchWrapper89_copies1981 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_6
  simp only [← c63390, ← c63391, ← c63392] at e_1023_6
  have e_1023_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_7
  simp only [← c63393, ← c63394, k2, ← c63395] at e_1023_7
  have hr := e_1023_7
  simp only [e_1023_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8479)) := by
  have c63396 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.1
  have c63397 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.1
  have c63398 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.1
  have c63399 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.1
  have c63400 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.1
  have c63401 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_8
  simp only [← c63396, ← c63397, ← c63398] at e_1023_8
  have e_1023_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_9
  simp only [← c63399, ← c63400, k2, ← c63401] at e_1023_9
  have hr := e_1023_9
  simp only [e_1023_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8480)) := by
  have c63402 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63403 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63404 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63405 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63406 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63407 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_10
  simp only [← c63402, ← c63403, ← c63404] at e_1023_10
  have e_1023_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_11
  simp only [← c63405, ← c63406, k2, ← c63407] at e_1023_11
  have hr := e_1023_11
  simp only [e_1023_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8481)) := by
  have c63408 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63409 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63410 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63411 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63412 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63413 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_12
  simp only [← c63408, ← c63409, ← c63410] at e_1023_12
  have e_1023_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_13
  simp only [← c63411, ← c63412, k2, ← c63413] at e_1023_13
  have hr := e_1023_13
  simp only [e_1023_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8482)) := by
  have c63414 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63415 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63416 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63417 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63418 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63419 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_14
  simp only [← c63414, ← c63415, ← c63416] at e_1023_14
  have e_1023_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_15
  simp only [← c63417, ← c63418, k2, ← c63419] at e_1023_15
  have hr := e_1023_15
  simp only [e_1023_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8483)) := by
  have c63420 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63421 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63422 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63423 := (publicBatchWrapper89_copies1981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63424 := (publicBatchWrapper89_copies1982 a h).1
  have c63425 := (publicBatchWrapper89_copies1982 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_16
  simp only [← c63420, ← c63421, ← c63422] at e_1023_16
  have e_1023_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_17
  simp only [← c63423, ← c63424, k2, ← c63425] at e_1023_17
  have hr := e_1023_17
  simp only [e_1023_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1023 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8484)) := by
  have c63426 := (publicBatchWrapper89_copies1982 a h).2.2.1
  have c63427 := (publicBatchWrapper89_copies1982 a h).2.2.2.1
  have c63428 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.1
  have c63429 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.1
  have c63430 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.1
  have c63431 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1023_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_18
  simp only [← c63426, ← c63427, ← c63428] at e_1023_18
  have e_1023_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1023 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1023_19
  simp only [← c63429, ← c63430, k2, ← c63431] at e_1023_19
  have hr := e_1023_19
  simp only [e_1023_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8485)) := by
  have c63432 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.1
  have c63433 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.1
  have c63434 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63435 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63436 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63437 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_0
  simp only [← c63432, ← c63433, ← c63434] at e_1024_0
  have e_1024_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_1
  simp only [← c63435, ← c63436, k2, ← c63437] at e_1024_1
  have hr := e_1024_1
  simp only [e_1024_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8486)) := by
  have c63438 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63439 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63440 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63441 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63442 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63443 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_2
  simp only [← c63438, ← c63439, ← c63440] at e_1024_2
  have e_1024_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_3
  simp only [← c63441, ← c63442, k2, ← c63443] at e_1024_3
  have hr := e_1024_3
  simp only [e_1024_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8487)) := by
  have c63444 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63445 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63446 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63447 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63448 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63449 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_4
  simp only [← c63444, ← c63445, ← c63446] at e_1024_4
  have e_1024_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_5
  simp only [← c63447, ← c63448, k2, ← c63449] at e_1024_5
  have hr := e_1024_5
  simp only [e_1024_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8488)) := by
  have c63450 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63451 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63452 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63453 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63454 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63455 := (publicBatchWrapper89_copies1982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_6
  simp only [← c63450, ← c63451, ← c63452] at e_1024_6
  have e_1024_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_7
  simp only [← c63453, ← c63454, k2, ← c63455] at e_1024_7
  have hr := e_1024_7
  simp only [e_1024_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8489)) := by
  have c63456 := (publicBatchWrapper89_copies1983 a h).1
  have c63457 := (publicBatchWrapper89_copies1983 a h).2.1
  have c63458 := (publicBatchWrapper89_copies1983 a h).2.2.1
  have c63459 := (publicBatchWrapper89_copies1983 a h).2.2.2.1
  have c63460 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.1
  have c63461 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_8
  simp only [← c63456, ← c63457, ← c63458] at e_1024_8
  have e_1024_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_9
  simp only [← c63459, ← c63460, k2, ← c63461] at e_1024_9
  have hr := e_1024_9
  simp only [e_1024_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8490)) := by
  have c63462 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.1
  have c63463 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.1
  have c63464 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.1
  have c63465 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.1
  have c63466 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63467 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_10
  simp only [← c63462, ← c63463, ← c63464] at e_1024_10
  have e_1024_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_11
  simp only [← c63465, ← c63466, k2, ← c63467] at e_1024_11
  have hr := e_1024_11
  simp only [e_1024_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8491)) := by
  have c63468 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63469 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63470 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63471 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63472 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63473 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_12
  simp only [← c63468, ← c63469, ← c63470] at e_1024_12
  have e_1024_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_13
  simp only [← c63471, ← c63472, k2, ← c63473] at e_1024_13
  have hr := e_1024_13
  simp only [e_1024_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8492)) := by
  have c63474 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63475 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63476 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63477 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63478 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63479 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_14
  simp only [← c63474, ← c63475, ← c63476] at e_1024_14
  have e_1024_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_15
  simp only [← c63477, ← c63478, k2, ← c63479] at e_1024_15
  have hr := e_1024_15
  simp only [e_1024_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8493)) := by
  have c63480 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63481 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63482 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63483 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63484 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63485 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_16
  simp only [← c63480, ← c63481, ← c63482] at e_1024_16
  have e_1024_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_17
  simp only [← c63483, ← c63484, k2, ← c63485] at e_1024_17
  have hr := e_1024_17
  simp only [e_1024_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1024 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8494)) := by
  have c63486 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63487 := (publicBatchWrapper89_copies1983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63488 := (publicBatchWrapper89_copies1984 a h).1
  have c63489 := (publicBatchWrapper89_copies1984 a h).2.1
  have c63490 := (publicBatchWrapper89_copies1984 a h).2.2.1
  have c63491 := (publicBatchWrapper89_copies1984 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1024_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_18
  simp only [← c63486, ← c63487, ← c63488] at e_1024_18
  have e_1024_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1024 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1024_19
  simp only [← c63489, ← c63490, k2, ← c63491] at e_1024_19
  have hr := e_1024_19
  simp only [e_1024_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8495)) := by
  have c63492 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.1
  have c63493 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.1
  have c63494 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.1
  have c63495 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.1
  have c63496 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.1
  have c63497 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_0
  simp only [← c63492, ← c63493, ← c63494] at e_1025_0
  have e_1025_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_1
  simp only [← c63495, ← c63496, k2, ← c63497] at e_1025_1
  have hr := e_1025_1
  simp only [e_1025_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8496)) := by
  have c63498 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63499 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63500 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63501 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63502 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63503 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_2
  simp only [← c63498, ← c63499, ← c63500] at e_1025_2
  have e_1025_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_3
  simp only [← c63501, ← c63502, k2, ← c63503] at e_1025_3
  have hr := e_1025_3
  simp only [e_1025_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8497)) := by
  have c63504 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63505 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63506 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63507 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63508 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63509 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_4
  simp only [← c63504, ← c63505, ← c63506] at e_1025_4
  have e_1025_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_5
  simp only [← c63507, ← c63508, k2, ← c63509] at e_1025_5
  have hr := e_1025_5
  simp only [e_1025_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8498)) := by
  have c63510 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63511 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63512 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63513 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63514 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63515 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_6
  simp only [← c63510, ← c63511, ← c63512] at e_1025_6
  have e_1025_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_7
  simp only [← c63513, ← c63514, k2, ← c63515] at e_1025_7
  have hr := e_1025_7
  simp only [e_1025_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8499)) := by
  have c63516 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63517 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63518 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63519 := (publicBatchWrapper89_copies1984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63520 := (publicBatchWrapper89_copies1985 a h).1
  have c63521 := (publicBatchWrapper89_copies1985 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_8
  simp only [← c63516, ← c63517, ← c63518] at e_1025_8
  have e_1025_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_9
  simp only [← c63519, ← c63520, k2, ← c63521] at e_1025_9
  have hr := e_1025_9
  simp only [e_1025_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8500)) := by
  have c63522 := (publicBatchWrapper89_copies1985 a h).2.2.1
  have c63523 := (publicBatchWrapper89_copies1985 a h).2.2.2.1
  have c63524 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.1
  have c63525 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.1
  have c63526 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.1
  have c63527 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_10
  simp only [← c63522, ← c63523, ← c63524] at e_1025_10
  have e_1025_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_11
  simp only [← c63525, ← c63526, k2, ← c63527] at e_1025_11
  have hr := e_1025_11
  simp only [e_1025_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8501)) := by
  have c63528 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.1
  have c63529 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.1
  have c63530 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63531 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63532 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63533 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_12
  simp only [← c63528, ← c63529, ← c63530] at e_1025_12
  have e_1025_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_13
  simp only [← c63531, ← c63532, k2, ← c63533] at e_1025_13
  have hr := e_1025_13
  simp only [e_1025_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8502)) := by
  have c63534 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63535 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63536 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63537 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63538 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63539 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_14
  simp only [← c63534, ← c63535, ← c63536] at e_1025_14
  have e_1025_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_15
  simp only [← c63537, ← c63538, k2, ← c63539] at e_1025_15
  have hr := e_1025_15
  simp only [e_1025_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8503)) := by
  have c63540 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63541 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63542 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63543 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63544 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63545 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_16
  simp only [← c63540, ← c63541, ← c63542] at e_1025_16
  have e_1025_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_17
  simp only [← c63543, ← c63544, k2, ← c63545] at e_1025_17
  have hr := e_1025_17
  simp only [e_1025_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1025 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8504)) := by
  have c63546 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63547 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63548 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63549 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63550 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63551 := (publicBatchWrapper89_copies1985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1025_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_18
  simp only [← c63546, ← c63547, ← c63548] at e_1025_18
  have e_1025_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1025 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1025_19
  simp only [← c63549, ← c63550, k2, ← c63551] at e_1025_19
  have hr := e_1025_19
  simp only [e_1025_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8505)) := by
  have c63552 := (publicBatchWrapper89_copies1986 a h).1
  have c63553 := (publicBatchWrapper89_copies1986 a h).2.1
  have c63554 := (publicBatchWrapper89_copies1986 a h).2.2.1
  have c63555 := (publicBatchWrapper89_copies1986 a h).2.2.2.1
  have c63556 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.1
  have c63557 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_0
  simp only [← c63552, ← c63553, ← c63554] at e_1026_0
  have e_1026_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_1
  simp only [← c63555, ← c63556, k2, ← c63557] at e_1026_1
  have hr := e_1026_1
  simp only [e_1026_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8506)) := by
  have c63558 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.1
  have c63559 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.1
  have c63560 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.1
  have c63561 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.1
  have c63562 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63563 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_2
  simp only [← c63558, ← c63559, ← c63560] at e_1026_2
  have e_1026_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_3
  simp only [← c63561, ← c63562, k2, ← c63563] at e_1026_3
  have hr := e_1026_3
  simp only [e_1026_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8507)) := by
  have c63564 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63565 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63566 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63567 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63568 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63569 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_4
  simp only [← c63564, ← c63565, ← c63566] at e_1026_4
  have e_1026_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_5
  simp only [← c63567, ← c63568, k2, ← c63569] at e_1026_5
  have hr := e_1026_5
  simp only [e_1026_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8508)) := by
  have c63570 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63571 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63572 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63573 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63574 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63575 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_6
  simp only [← c63570, ← c63571, ← c63572] at e_1026_6
  have e_1026_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_7
  simp only [← c63573, ← c63574, k2, ← c63575] at e_1026_7
  have hr := e_1026_7
  simp only [e_1026_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8509)) := by
  have c63576 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63577 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63578 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63579 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63580 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63581 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_8
  simp only [← c63576, ← c63577, ← c63578] at e_1026_8
  have e_1026_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_9
  simp only [← c63579, ← c63580, k2, ← c63581] at e_1026_9
  have hr := e_1026_9
  simp only [e_1026_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8510)) := by
  have c63582 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63583 := (publicBatchWrapper89_copies1986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63584 := (publicBatchWrapper89_copies1987 a h).1
  have c63585 := (publicBatchWrapper89_copies1987 a h).2.1
  have c63586 := (publicBatchWrapper89_copies1987 a h).2.2.1
  have c63587 := (publicBatchWrapper89_copies1987 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_10
  simp only [← c63582, ← c63583, ← c63584] at e_1026_10
  have e_1026_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_11
  simp only [← c63585, ← c63586, k2, ← c63587] at e_1026_11
  have hr := e_1026_11
  simp only [e_1026_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8511)) := by
  have c63588 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.1
  have c63589 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.1
  have c63590 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.1
  have c63591 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.1
  have c63592 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.1
  have c63593 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_12
  simp only [← c63588, ← c63589, ← c63590] at e_1026_12
  have e_1026_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_13
  simp only [← c63591, ← c63592, k2, ← c63593] at e_1026_13
  have hr := e_1026_13
  simp only [e_1026_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8512)) := by
  have c63594 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63595 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63596 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63597 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63598 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63599 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_14
  simp only [← c63594, ← c63595, ← c63596] at e_1026_14
  have e_1026_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_15
  simp only [← c63597, ← c63598, k2, ← c63599] at e_1026_15
  have hr := e_1026_15
  simp only [e_1026_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8513)) := by
  have c63600 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63601 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63602 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63603 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63604 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63605 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_16
  simp only [← c63600, ← c63601, ← c63602] at e_1026_16
  have e_1026_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_17
  simp only [← c63603, ← c63604, k2, ← c63605] at e_1026_17
  have hr := e_1026_17
  simp only [e_1026_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1026 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8514)) := by
  have c63606 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63607 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63608 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63609 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63610 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63611 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1026_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_18
  simp only [← c63606, ← c63607, ← c63608] at e_1026_18
  have e_1026_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1026 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1026_19
  simp only [← c63609, ← c63610, k2, ← c63611] at e_1026_19
  have hr := e_1026_19
  simp only [e_1026_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8515)) := by
  have c63612 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63613 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63614 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63615 := (publicBatchWrapper89_copies1987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63616 := (publicBatchWrapper89_copies1988 a h).1
  have c63617 := (publicBatchWrapper89_copies1988 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_0
  simp only [← c63612, ← c63613, ← c63614] at e_1027_0
  have e_1027_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_1
  simp only [← c63615, ← c63616, k2, ← c63617] at e_1027_1
  have hr := e_1027_1
  simp only [e_1027_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8516)) := by
  have c63618 := (publicBatchWrapper89_copies1988 a h).2.2.1
  have c63619 := (publicBatchWrapper89_copies1988 a h).2.2.2.1
  have c63620 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.1
  have c63621 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.1
  have c63622 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.1
  have c63623 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_2
  simp only [← c63618, ← c63619, ← c63620] at e_1027_2
  have e_1027_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_3
  simp only [← c63621, ← c63622, k2, ← c63623] at e_1027_3
  have hr := e_1027_3
  simp only [e_1027_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8517)) := by
  have c63624 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.1
  have c63625 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.1
  have c63626 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63627 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63628 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63629 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_4
  simp only [← c63624, ← c63625, ← c63626] at e_1027_4
  have e_1027_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_5
  simp only [← c63627, ← c63628, k2, ← c63629] at e_1027_5
  have hr := e_1027_5
  simp only [e_1027_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8518)) := by
  have c63630 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63631 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63632 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63633 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63634 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63635 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_6
  simp only [← c63630, ← c63631, ← c63632] at e_1027_6
  have e_1027_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_7
  simp only [← c63633, ← c63634, k2, ← c63635] at e_1027_7
  have hr := e_1027_7
  simp only [e_1027_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8519)) := by
  have c63636 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63637 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63638 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63639 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63640 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63641 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_8
  simp only [← c63636, ← c63637, ← c63638] at e_1027_8
  have e_1027_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_9
  simp only [← c63639, ← c63640, k2, ← c63641] at e_1027_9
  have hr := e_1027_9
  simp only [e_1027_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8520)) := by
  have c63642 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63643 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63644 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63645 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63646 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63647 := (publicBatchWrapper89_copies1988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_10
  simp only [← c63642, ← c63643, ← c63644] at e_1027_10
  have e_1027_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_11
  simp only [← c63645, ← c63646, k2, ← c63647] at e_1027_11
  have hr := e_1027_11
  simp only [e_1027_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8521)) := by
  have c63648 := (publicBatchWrapper89_copies1989 a h).1
  have c63649 := (publicBatchWrapper89_copies1989 a h).2.1
  have c63650 := (publicBatchWrapper89_copies1989 a h).2.2.1
  have c63651 := (publicBatchWrapper89_copies1989 a h).2.2.2.1
  have c63652 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.1
  have c63653 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_12
  simp only [← c63648, ← c63649, ← c63650] at e_1027_12
  have e_1027_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_13
  simp only [← c63651, ← c63652, k2, ← c63653] at e_1027_13
  have hr := e_1027_13
  simp only [e_1027_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8522)) := by
  have c63654 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.1
  have c63655 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.1
  have c63656 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.1
  have c63657 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.1
  have c63658 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63659 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_14
  simp only [← c63654, ← c63655, ← c63656] at e_1027_14
  have e_1027_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_15
  simp only [← c63657, ← c63658, k2, ← c63659] at e_1027_15
  have hr := e_1027_15
  simp only [e_1027_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8523)) := by
  have c63660 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63661 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63662 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63663 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63664 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63665 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_16
  simp only [← c63660, ← c63661, ← c63662] at e_1027_16
  have e_1027_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_17
  simp only [← c63663, ← c63664, k2, ← c63665] at e_1027_17
  have hr := e_1027_17
  simp only [e_1027_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1027 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8524)) := by
  have c63666 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63667 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63668 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63669 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63670 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63671 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1027_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_18
  simp only [← c63666, ← c63667, ← c63668] at e_1027_18
  have e_1027_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1027 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1027_19
  simp only [← c63669, ← c63670, k2, ← c63671] at e_1027_19
  have hr := e_1027_19
  simp only [e_1027_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8525)) := by
  have c63672 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63673 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63674 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63675 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63676 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63677 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_0
  simp only [← c63672, ← c63673, ← c63674] at e_1028_0
  have e_1028_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_1
  simp only [← c63675, ← c63676, k2, ← c63677] at e_1028_1
  have hr := e_1028_1
  simp only [e_1028_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8526)) := by
  have c63678 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63679 := (publicBatchWrapper89_copies1989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63680 := (publicBatchWrapper89_copies1990 a h).1
  have c63681 := (publicBatchWrapper89_copies1990 a h).2.1
  have c63682 := (publicBatchWrapper89_copies1990 a h).2.2.1
  have c63683 := (publicBatchWrapper89_copies1990 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_2
  simp only [← c63678, ← c63679, ← c63680] at e_1028_2
  have e_1028_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_3
  simp only [← c63681, ← c63682, k2, ← c63683] at e_1028_3
  have hr := e_1028_3
  simp only [e_1028_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8527)) := by
  have c63684 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.1
  have c63685 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.1
  have c63686 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.1
  have c63687 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.1
  have c63688 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.1
  have c63689 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_4
  simp only [← c63684, ← c63685, ← c63686] at e_1028_4
  have e_1028_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_5
  simp only [← c63687, ← c63688, k2, ← c63689] at e_1028_5
  have hr := e_1028_5
  simp only [e_1028_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8528)) := by
  have c63690 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63691 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63692 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63693 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63694 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63695 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_6
  simp only [← c63690, ← c63691, ← c63692] at e_1028_6
  have e_1028_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_7
  simp only [← c63693, ← c63694, k2, ← c63695] at e_1028_7
  have hr := e_1028_7
  simp only [e_1028_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8529)) := by
  have c63696 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63697 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63698 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63699 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63700 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63701 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_8
  simp only [← c63696, ← c63697, ← c63698] at e_1028_8
  have e_1028_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_9
  simp only [← c63699, ← c63700, k2, ← c63701] at e_1028_9
  have hr := e_1028_9
  simp only [e_1028_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8530)) := by
  have c63702 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63703 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63704 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63705 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63706 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63707 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_10
  simp only [← c63702, ← c63703, ← c63704] at e_1028_10
  have e_1028_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_11
  simp only [← c63705, ← c63706, k2, ← c63707] at e_1028_11
  have hr := e_1028_11
  simp only [e_1028_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8531)) := by
  have c63708 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63709 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63710 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63711 := (publicBatchWrapper89_copies1990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63712 := (publicBatchWrapper89_copies1991 a h).1
  have c63713 := (publicBatchWrapper89_copies1991 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_12
  simp only [← c63708, ← c63709, ← c63710] at e_1028_12
  have e_1028_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_13
  simp only [← c63711, ← c63712, k2, ← c63713] at e_1028_13
  have hr := e_1028_13
  simp only [e_1028_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8532)) := by
  have c63714 := (publicBatchWrapper89_copies1991 a h).2.2.1
  have c63715 := (publicBatchWrapper89_copies1991 a h).2.2.2.1
  have c63716 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.1
  have c63717 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.1
  have c63718 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.1
  have c63719 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_14
  simp only [← c63714, ← c63715, ← c63716] at e_1028_14
  have e_1028_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_15
  simp only [← c63717, ← c63718, k2, ← c63719] at e_1028_15
  have hr := e_1028_15
  simp only [e_1028_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8533)) := by
  have c63720 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.1
  have c63721 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.1
  have c63722 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63723 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63724 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63725 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_16
  simp only [← c63720, ← c63721, ← c63722] at e_1028_16
  have e_1028_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_17
  simp only [← c63723, ← c63724, k2, ← c63725] at e_1028_17
  have hr := e_1028_17
  simp only [e_1028_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1028 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8534)) := by
  have c63726 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63727 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63728 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63729 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63730 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63731 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1028_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_18
  simp only [← c63726, ← c63727, ← c63728] at e_1028_18
  have e_1028_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1028 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1028_19
  simp only [← c63729, ← c63730, k2, ← c63731] at e_1028_19
  have hr := e_1028_19
  simp only [e_1028_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8535)) := by
  have c63732 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63733 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63734 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63735 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63736 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63737 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_0
  simp only [← c63732, ← c63733, ← c63734] at e_1029_0
  have e_1029_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_1
  simp only [← c63735, ← c63736, k2, ← c63737] at e_1029_1
  have hr := e_1029_1
  simp only [e_1029_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8536)) := by
  have c63738 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63739 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63740 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63741 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63742 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63743 := (publicBatchWrapper89_copies1991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_2
  simp only [← c63738, ← c63739, ← c63740] at e_1029_2
  have e_1029_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_3
  simp only [← c63741, ← c63742, k2, ← c63743] at e_1029_3
  have hr := e_1029_3
  simp only [e_1029_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8537)) := by
  have c63744 := (publicBatchWrapper89_copies1992 a h).1
  have c63745 := (publicBatchWrapper89_copies1992 a h).2.1
  have c63746 := (publicBatchWrapper89_copies1992 a h).2.2.1
  have c63747 := (publicBatchWrapper89_copies1992 a h).2.2.2.1
  have c63748 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.1
  have c63749 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_4
  simp only [← c63744, ← c63745, ← c63746] at e_1029_4
  have e_1029_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_5
  simp only [← c63747, ← c63748, k2, ← c63749] at e_1029_5
  have hr := e_1029_5
  simp only [e_1029_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8538)) := by
  have c63750 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.1
  have c63751 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.1
  have c63752 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.1
  have c63753 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.1
  have c63754 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63755 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_6
  simp only [← c63750, ← c63751, ← c63752] at e_1029_6
  have e_1029_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_7
  simp only [← c63753, ← c63754, k2, ← c63755] at e_1029_7
  have hr := e_1029_7
  simp only [e_1029_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8539)) := by
  have c63756 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63757 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63758 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63759 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63760 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63761 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_8
  simp only [← c63756, ← c63757, ← c63758] at e_1029_8
  have e_1029_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_9
  simp only [← c63759, ← c63760, k2, ← c63761] at e_1029_9
  have hr := e_1029_9
  simp only [e_1029_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8540)) := by
  have c63762 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63763 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63764 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63765 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63766 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63767 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_10
  simp only [← c63762, ← c63763, ← c63764] at e_1029_10
  have e_1029_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_11
  simp only [← c63765, ← c63766, k2, ← c63767] at e_1029_11
  have hr := e_1029_11
  simp only [e_1029_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8541)) := by
  have c63768 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63769 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63770 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63771 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63772 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63773 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_12
  simp only [← c63768, ← c63769, ← c63770] at e_1029_12
  have e_1029_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_13
  simp only [← c63771, ← c63772, k2, ← c63773] at e_1029_13
  have hr := e_1029_13
  simp only [e_1029_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8542)) := by
  have c63774 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63775 := (publicBatchWrapper89_copies1992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63776 := (publicBatchWrapper89_copies1993 a h).1
  have c63777 := (publicBatchWrapper89_copies1993 a h).2.1
  have c63778 := (publicBatchWrapper89_copies1993 a h).2.2.1
  have c63779 := (publicBatchWrapper89_copies1993 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_14
  simp only [← c63774, ← c63775, ← c63776] at e_1029_14
  have e_1029_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_15
  simp only [← c63777, ← c63778, k2, ← c63779] at e_1029_15
  have hr := e_1029_15
  simp only [e_1029_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8543)) := by
  have c63780 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.1
  have c63781 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.1
  have c63782 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.1
  have c63783 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.1
  have c63784 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.1
  have c63785 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_16
  simp only [← c63780, ← c63781, ← c63782] at e_1029_16
  have e_1029_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_17
  simp only [← c63783, ← c63784, k2, ← c63785] at e_1029_17
  have hr := e_1029_17
  simp only [e_1029_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1029 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8544)) := by
  have c63786 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63787 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63788 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63789 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63790 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63791 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1029_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_18
  simp only [← c63786, ← c63787, ← c63788] at e_1029_18
  have e_1029_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1029 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1029_19
  simp only [← c63789, ← c63790, k2, ← c63791] at e_1029_19
  have hr := e_1029_19
  simp only [e_1029_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8545)) := by
  have c63792 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63793 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63794 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63795 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63796 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63797 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_0
  simp only [← c63792, ← c63793, ← c63794] at e_1030_0
  have e_1030_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_1
  simp only [← c63795, ← c63796, k2, ← c63797] at e_1030_1
  have hr := e_1030_1
  simp only [e_1030_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8546)) := by
  have c63798 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63799 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63800 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63801 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63802 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63803 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_2
  simp only [← c63798, ← c63799, ← c63800] at e_1030_2
  have e_1030_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_3
  simp only [← c63801, ← c63802, k2, ← c63803] at e_1030_3
  have hr := e_1030_3
  simp only [e_1030_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8547)) := by
  have c63804 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63805 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63806 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63807 := (publicBatchWrapper89_copies1993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63808 := (publicBatchWrapper89_copies1994 a h).1
  have c63809 := (publicBatchWrapper89_copies1994 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_4
  simp only [← c63804, ← c63805, ← c63806] at e_1030_4
  have e_1030_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_5
  simp only [← c63807, ← c63808, k2, ← c63809] at e_1030_5
  have hr := e_1030_5
  simp only [e_1030_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8548)) := by
  have c63810 := (publicBatchWrapper89_copies1994 a h).2.2.1
  have c63811 := (publicBatchWrapper89_copies1994 a h).2.2.2.1
  have c63812 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.1
  have c63813 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.1
  have c63814 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.1
  have c63815 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_6
  simp only [← c63810, ← c63811, ← c63812] at e_1030_6
  have e_1030_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_7
  simp only [← c63813, ← c63814, k2, ← c63815] at e_1030_7
  have hr := e_1030_7
  simp only [e_1030_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8549)) := by
  have c63816 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.1
  have c63817 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.1
  have c63818 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63819 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63820 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63821 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_8
  simp only [← c63816, ← c63817, ← c63818] at e_1030_8
  have e_1030_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_9
  simp only [← c63819, ← c63820, k2, ← c63821] at e_1030_9
  have hr := e_1030_9
  simp only [e_1030_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8550)) := by
  have c63822 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63823 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63824 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63825 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63826 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63827 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_10
  simp only [← c63822, ← c63823, ← c63824] at e_1030_10
  have e_1030_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_11
  simp only [← c63825, ← c63826, k2, ← c63827] at e_1030_11
  have hr := e_1030_11
  simp only [e_1030_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8551)) := by
  have c63828 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63829 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63830 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63831 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63832 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63833 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_12
  simp only [← c63828, ← c63829, ← c63830] at e_1030_12
  have e_1030_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_13
  simp only [← c63831, ← c63832, k2, ← c63833] at e_1030_13
  have hr := e_1030_13
  simp only [e_1030_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8552)) := by
  have c63834 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63835 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63836 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63837 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63838 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63839 := (publicBatchWrapper89_copies1994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_14
  simp only [← c63834, ← c63835, ← c63836] at e_1030_14
  have e_1030_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_15
  simp only [← c63837, ← c63838, k2, ← c63839] at e_1030_15
  have hr := e_1030_15
  simp only [e_1030_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8553)) := by
  have c63840 := (publicBatchWrapper89_copies1995 a h).1
  have c63841 := (publicBatchWrapper89_copies1995 a h).2.1
  have c63842 := (publicBatchWrapper89_copies1995 a h).2.2.1
  have c63843 := (publicBatchWrapper89_copies1995 a h).2.2.2.1
  have c63844 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.1
  have c63845 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_16
  simp only [← c63840, ← c63841, ← c63842] at e_1030_16
  have e_1030_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_17
  simp only [← c63843, ← c63844, k2, ← c63845] at e_1030_17
  have hr := e_1030_17
  simp only [e_1030_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1030 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8554)) := by
  have c63846 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.1
  have c63847 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.1
  have c63848 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.1
  have c63849 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.1
  have c63850 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63851 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1030_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_18
  simp only [← c63846, ← c63847, ← c63848] at e_1030_18
  have e_1030_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1030 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1030_19
  simp only [← c63849, ← c63850, k2, ← c63851] at e_1030_19
  have hr := e_1030_19
  simp only [e_1030_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8555)) := by
  have c63852 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63853 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63854 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63855 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63856 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63857 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_0
  simp only [← c63852, ← c63853, ← c63854] at e_1031_0
  have e_1031_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_1
  simp only [← c63855, ← c63856, k2, ← c63857] at e_1031_1
  have hr := e_1031_1
  simp only [e_1031_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8556)) := by
  have c63858 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63859 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63860 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63861 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63862 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63863 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_2
  simp only [← c63858, ← c63859, ← c63860] at e_1031_2
  have e_1031_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_3
  simp only [← c63861, ← c63862, k2, ← c63863] at e_1031_3
  have hr := e_1031_3
  simp only [e_1031_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8557)) := by
  have c63864 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63865 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63866 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63867 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63868 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63869 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_4
  simp only [← c63864, ← c63865, ← c63866] at e_1031_4
  have e_1031_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_5
  simp only [← c63867, ← c63868, k2, ← c63869] at e_1031_5
  have hr := e_1031_5
  simp only [e_1031_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8558)) := by
  have c63870 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63871 := (publicBatchWrapper89_copies1995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63872 := (publicBatchWrapper89_copies1996 a h).1
  have c63873 := (publicBatchWrapper89_copies1996 a h).2.1
  have c63874 := (publicBatchWrapper89_copies1996 a h).2.2.1
  have c63875 := (publicBatchWrapper89_copies1996 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_6
  simp only [← c63870, ← c63871, ← c63872] at e_1031_6
  have e_1031_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_7
  simp only [← c63873, ← c63874, k2, ← c63875] at e_1031_7
  have hr := e_1031_7
  simp only [e_1031_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8559)) := by
  have c63876 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.1
  have c63877 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.1
  have c63878 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.1
  have c63879 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.1
  have c63880 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.1
  have c63881 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_8
  simp only [← c63876, ← c63877, ← c63878] at e_1031_8
  have e_1031_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_9
  simp only [← c63879, ← c63880, k2, ← c63881] at e_1031_9
  have hr := e_1031_9
  simp only [e_1031_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8560)) := by
  have c63882 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63883 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63884 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63885 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63886 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63887 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_10
  simp only [← c63882, ← c63883, ← c63884] at e_1031_10
  have e_1031_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_11
  simp only [← c63885, ← c63886, k2, ← c63887] at e_1031_11
  have hr := e_1031_11
  simp only [e_1031_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8561)) := by
  have c63888 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63889 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63890 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63891 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63892 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63893 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_12
  simp only [← c63888, ← c63889, ← c63890] at e_1031_12
  have e_1031_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_13
  simp only [← c63891, ← c63892, k2, ← c63893] at e_1031_13
  have hr := e_1031_13
  simp only [e_1031_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8562)) := by
  have c63894 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63895 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63896 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63897 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63898 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63899 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_14
  simp only [← c63894, ← c63895, ← c63896] at e_1031_14
  have e_1031_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_15
  simp only [← c63897, ← c63898, k2, ← c63899] at e_1031_15
  have hr := e_1031_15
  simp only [e_1031_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8563)) := by
  have c63900 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63901 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63902 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63903 := (publicBatchWrapper89_copies1996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63904 := (publicBatchWrapper89_copies1997 a h).1
  have c63905 := (publicBatchWrapper89_copies1997 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_16
  simp only [← c63900, ← c63901, ← c63902] at e_1031_16
  have e_1031_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_17
  simp only [← c63903, ← c63904, k2, ← c63905] at e_1031_17
  have hr := e_1031_17
  simp only [e_1031_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1031 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8564)) := by
  have c63906 := (publicBatchWrapper89_copies1997 a h).2.2.1
  have c63907 := (publicBatchWrapper89_copies1997 a h).2.2.2.1
  have c63908 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.1
  have c63909 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.1
  have c63910 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.1
  have c63911 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1031_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_18
  simp only [← c63906, ← c63907, ← c63908] at e_1031_18
  have e_1031_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1031 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1031_19
  simp only [← c63909, ← c63910, k2, ← c63911] at e_1031_19
  have hr := e_1031_19
  simp only [e_1031_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8565)) := by
  have c63912 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.1
  have c63913 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.1
  have c63914 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63915 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63916 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63917 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_0
  simp only [← c63912, ← c63913, ← c63914] at e_1032_0
  have e_1032_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_1
  simp only [← c63915, ← c63916, k2, ← c63917] at e_1032_1
  have hr := e_1032_1
  simp only [e_1032_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8566)) := by
  have c63918 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63919 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63920 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63921 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63922 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63923 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_2
  simp only [← c63918, ← c63919, ← c63920] at e_1032_2
  have e_1032_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_3
  simp only [← c63921, ← c63922, k2, ← c63923] at e_1032_3
  have hr := e_1032_3
  simp only [e_1032_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8567)) := by
  have c63924 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63925 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63926 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63927 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63928 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63929 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_4
  simp only [← c63924, ← c63925, ← c63926] at e_1032_4
  have e_1032_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_5
  simp only [← c63927, ← c63928, k2, ← c63929] at e_1032_5
  have hr := e_1032_5
  simp only [e_1032_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8568)) := by
  have c63930 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63931 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63932 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63933 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63934 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63935 := (publicBatchWrapper89_copies1997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_6
  simp only [← c63930, ← c63931, ← c63932] at e_1032_6
  have e_1032_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_7
  simp only [← c63933, ← c63934, k2, ← c63935] at e_1032_7
  have hr := e_1032_7
  simp only [e_1032_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8569)) := by
  have c63936 := (publicBatchWrapper89_copies1998 a h).1
  have c63937 := (publicBatchWrapper89_copies1998 a h).2.1
  have c63938 := (publicBatchWrapper89_copies1998 a h).2.2.1
  have c63939 := (publicBatchWrapper89_copies1998 a h).2.2.2.1
  have c63940 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.1
  have c63941 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_8
  simp only [← c63936, ← c63937, ← c63938] at e_1032_8
  have e_1032_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_9
  simp only [← c63939, ← c63940, k2, ← c63941] at e_1032_9
  have hr := e_1032_9
  simp only [e_1032_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8610)) := by
  have c63942 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.1
  have c63943 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.1
  have c63944 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.1
  have c63945 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.1
  have c63946 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63947 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_10
  simp only [← c63942, ← c63943, ← c63944] at e_1032_10
  have e_1032_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_11
  simp only [← c63945, ← c63946, k2, ← c63947] at e_1032_11
  have hr := e_1032_11
  simp only [e_1032_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8611)) := by
  have c63948 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63949 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63950 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63951 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63952 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63953 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_12
  simp only [← c63948, ← c63949, ← c63950] at e_1032_12
  have e_1032_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_13
  simp only [← c63951, ← c63952, k2, ← c63953] at e_1032_13
  have hr := e_1032_13
  simp only [e_1032_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8612)) := by
  have c63954 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63955 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63956 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63957 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63958 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63959 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_14
  simp only [← c63954, ← c63955, ← c63956] at e_1032_14
  have e_1032_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_15
  simp only [← c63957, ← c63958, k2, ← c63959] at e_1032_15
  have hr := e_1032_15
  simp only [e_1032_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8613)) := by
  have c63960 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63961 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63962 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63963 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63964 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63965 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_16
  simp only [← c63960, ← c63961, ← c63962] at e_1032_16
  have e_1032_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_17
  simp only [← c63963, ← c63964, k2, ← c63965] at e_1032_17
  have hr := e_1032_17
  simp only [e_1032_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1032 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8614)) := by
  have c63966 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63967 := (publicBatchWrapper89_copies1998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c63968 := (publicBatchWrapper89_copies1999 a h).1
  have c63969 := (publicBatchWrapper89_copies1999 a h).2.1
  have c63970 := (publicBatchWrapper89_copies1999 a h).2.2.1
  have c63971 := (publicBatchWrapper89_copies1999 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1032_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_18
  simp only [← c63966, ← c63967, ← c63968] at e_1032_18
  have e_1032_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1032 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1032_19
  simp only [← c63969, ← c63970, k2, ← c63971] at e_1032_19
  have hr := e_1032_19
  simp only [e_1032_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8615)) := by
  have c63972 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.1
  have c63973 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.1
  have c63974 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.1
  have c63975 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.1
  have c63976 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.1
  have c63977 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_0
  simp only [← c63972, ← c63973, ← c63974] at e_1033_0
  have e_1033_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_1
  simp only [← c63975, ← c63976, k2, ← c63977] at e_1033_1
  have hr := e_1033_1
  simp only [e_1033_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8616)) := by
  have c63978 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.1
  have c63979 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c63980 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63981 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63982 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63983 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_2
  simp only [← c63978, ← c63979, ← c63980] at e_1033_2
  have e_1033_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_3
  simp only [← c63981, ← c63982, k2, ← c63983] at e_1033_3
  have hr := e_1033_3
  simp only [e_1033_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8617)) := by
  have c63984 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63985 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63986 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63987 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63988 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63989 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_4
  simp only [← c63984, ← c63985, ← c63986] at e_1033_4
  have e_1033_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_5
  simp only [← c63987, ← c63988, k2, ← c63989] at e_1033_5
  have hr := e_1033_5
  simp only [e_1033_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8618)) := by
  have c63990 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63991 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63992 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63993 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63994 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63995 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_6
  simp only [← c63990, ← c63991, ← c63992] at e_1033_6
  have e_1033_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_7
  simp only [← c63993, ← c63994, k2, ← c63995] at e_1033_7
  have hr := e_1033_7
  simp only [e_1033_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8619)) := by
  have c63996 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63997 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63998 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c63999 := (publicBatchWrapper89_copies1999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64000 := (publicBatchWrapper89_copies2000 a h).1
  have c64001 := (publicBatchWrapper89_copies2000 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_8
  simp only [← c63996, ← c63997, ← c63998] at e_1033_8
  have e_1033_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_9
  simp only [← c63999, ← c64000, k2, ← c64001] at e_1033_9
  have hr := e_1033_9
  simp only [e_1033_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8620)) := by
  have c64002 := (publicBatchWrapper89_copies2000 a h).2.2.1
  have c64003 := (publicBatchWrapper89_copies2000 a h).2.2.2.1
  have c64004 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.1
  have c64005 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.1
  have c64006 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.1
  have c64007 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_10
  simp only [← c64002, ← c64003, ← c64004] at e_1033_10
  have e_1033_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_11
  simp only [← c64005, ← c64006, k2, ← c64007] at e_1033_11
  have hr := e_1033_11
  simp only [e_1033_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8621)) := by
  have c64008 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.1
  have c64009 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.1
  have c64010 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64011 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64012 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64013 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_12
  simp only [← c64008, ← c64009, ← c64010] at e_1033_12
  have e_1033_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_13
  simp only [← c64011, ← c64012, k2, ← c64013] at e_1033_13
  have hr := e_1033_13
  simp only [e_1033_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8622)) := by
  have c64014 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64015 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64016 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64017 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64018 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64019 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_14
  simp only [← c64014, ← c64015, ← c64016] at e_1033_14
  have e_1033_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_15
  simp only [← c64017, ← c64018, k2, ← c64019] at e_1033_15
  have hr := e_1033_15
  simp only [e_1033_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8623)) := by
  have c64020 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64021 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64022 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64023 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64024 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64025 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_16
  simp only [← c64020, ← c64021, ← c64022] at e_1033_16
  have e_1033_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_17
  simp only [← c64023, ← c64024, k2, ← c64025] at e_1033_17
  have hr := e_1033_17
  simp only [e_1033_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1033 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8624)) := by
  have c64026 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64027 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64028 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64029 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64030 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64031 := (publicBatchWrapper89_copies2000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1033_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_18
  simp only [← c64026, ← c64027, ← c64028] at e_1033_18
  have e_1033_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1033 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1033_19
  simp only [← c64029, ← c64030, k2, ← c64031] at e_1033_19
  have hr := e_1033_19
  simp only [e_1033_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8625)) := by
  have c64032 := (publicBatchWrapper89_copies2001 a h).1
  have c64033 := (publicBatchWrapper89_copies2001 a h).2.1
  have c64034 := (publicBatchWrapper89_copies2001 a h).2.2.1
  have c64035 := (publicBatchWrapper89_copies2001 a h).2.2.2.1
  have c64036 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.1
  have c64037 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_0
  simp only [← c64032, ← c64033, ← c64034] at e_1034_0
  have e_1034_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_1
  simp only [← c64035, ← c64036, k2, ← c64037] at e_1034_1
  have hr := e_1034_1
  simp only [e_1034_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8626)) := by
  have c64038 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.1
  have c64039 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.1
  have c64040 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.1
  have c64041 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.1
  have c64042 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64043 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_2
  simp only [← c64038, ← c64039, ← c64040] at e_1034_2
  have e_1034_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_3
  simp only [← c64041, ← c64042, k2, ← c64043] at e_1034_3
  have hr := e_1034_3
  simp only [e_1034_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8627)) := by
  have c64044 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64045 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64046 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64047 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64048 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64049 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_4
  simp only [← c64044, ← c64045, ← c64046] at e_1034_4
  have e_1034_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_5
  simp only [← c64047, ← c64048, k2, ← c64049] at e_1034_5
  have hr := e_1034_5
  simp only [e_1034_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8628)) := by
  have c64050 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64051 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64052 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64053 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64054 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64055 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_6
  simp only [← c64050, ← c64051, ← c64052] at e_1034_6
  have e_1034_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_7
  simp only [← c64053, ← c64054, k2, ← c64055] at e_1034_7
  have hr := e_1034_7
  simp only [e_1034_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8629)) := by
  have c64056 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64057 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64058 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64059 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64060 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64061 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_8
  simp only [← c64056, ← c64057, ← c64058] at e_1034_8
  have e_1034_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_9
  simp only [← c64059, ← c64060, k2, ← c64061] at e_1034_9
  have hr := e_1034_9
  simp only [e_1034_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8630)) := by
  have c64062 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64063 := (publicBatchWrapper89_copies2001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64064 := (publicBatchWrapper89_copies2002 a h).1
  have c64065 := (publicBatchWrapper89_copies2002 a h).2.1
  have c64066 := (publicBatchWrapper89_copies2002 a h).2.2.1
  have c64067 := (publicBatchWrapper89_copies2002 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_10
  simp only [← c64062, ← c64063, ← c64064] at e_1034_10
  have e_1034_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_11
  simp only [← c64065, ← c64066, k2, ← c64067] at e_1034_11
  have hr := e_1034_11
  simp only [e_1034_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8631)) := by
  have c64068 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.1
  have c64069 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.1
  have c64070 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.1
  have c64071 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.1
  have c64072 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.1
  have c64073 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_12
  simp only [← c64068, ← c64069, ← c64070] at e_1034_12
  have e_1034_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_13
  simp only [← c64071, ← c64072, k2, ← c64073] at e_1034_13
  have hr := e_1034_13
  simp only [e_1034_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8632)) := by
  have c64074 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64075 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64076 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64077 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64078 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64079 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_14
  simp only [← c64074, ← c64075, ← c64076] at e_1034_14
  have e_1034_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_15
  simp only [← c64077, ← c64078, k2, ← c64079] at e_1034_15
  have hr := e_1034_15
  simp only [e_1034_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8633)) := by
  have c64080 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64081 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64082 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64083 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64084 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64085 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_16
  simp only [← c64080, ← c64081, ← c64082] at e_1034_16
  have e_1034_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_17
  simp only [← c64083, ← c64084, k2, ← c64085] at e_1034_17
  have hr := e_1034_17
  simp only [e_1034_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1034 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8634)) := by
  have c64086 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64087 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64088 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64089 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64090 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64091 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1034_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_18
  simp only [← c64086, ← c64087, ← c64088] at e_1034_18
  have e_1034_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1034 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1034_19
  simp only [← c64089, ← c64090, k2, ← c64091] at e_1034_19
  have hr := e_1034_19
  simp only [e_1034_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8635)) := by
  have c64092 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64093 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64094 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64095 := (publicBatchWrapper89_copies2002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64096 := (publicBatchWrapper89_copies2003 a h).1
  have c64097 := (publicBatchWrapper89_copies2003 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_0
  simp only [← c64092, ← c64093, ← c64094] at e_1035_0
  have e_1035_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_1
  simp only [← c64095, ← c64096, k2, ← c64097] at e_1035_1
  have hr := e_1035_1
  simp only [e_1035_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8636)) := by
  have c64098 := (publicBatchWrapper89_copies2003 a h).2.2.1
  have c64099 := (publicBatchWrapper89_copies2003 a h).2.2.2.1
  have c64100 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.1
  have c64101 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.1
  have c64102 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.1
  have c64103 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_2
  simp only [← c64098, ← c64099, ← c64100] at e_1035_2
  have e_1035_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_3
  simp only [← c64101, ← c64102, k2, ← c64103] at e_1035_3
  have hr := e_1035_3
  simp only [e_1035_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8637)) := by
  have c64104 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.1
  have c64105 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.1
  have c64106 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64107 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64108 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64109 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_4
  simp only [← c64104, ← c64105, ← c64106] at e_1035_4
  have e_1035_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_5
  simp only [← c64107, ← c64108, k2, ← c64109] at e_1035_5
  have hr := e_1035_5
  simp only [e_1035_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8638)) := by
  have c64110 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64111 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64112 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64113 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64114 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64115 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_6
  simp only [← c64110, ← c64111, ← c64112] at e_1035_6
  have e_1035_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_7
  simp only [← c64113, ← c64114, k2, ← c64115] at e_1035_7
  have hr := e_1035_7
  simp only [e_1035_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8639)) := by
  have c64116 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64117 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64118 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64119 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64120 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64121 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_8
  simp only [← c64116, ← c64117, ← c64118] at e_1035_8
  have e_1035_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_9
  simp only [← c64119, ← c64120, k2, ← c64121] at e_1035_9
  have hr := e_1035_9
  simp only [e_1035_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8640)) := by
  have c64122 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64123 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64124 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64125 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64126 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64127 := (publicBatchWrapper89_copies2003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_10
  simp only [← c64122, ← c64123, ← c64124] at e_1035_10
  have e_1035_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_11
  simp only [← c64125, ← c64126, k2, ← c64127] at e_1035_11
  have hr := e_1035_11
  simp only [e_1035_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8641)) := by
  have c64128 := (publicBatchWrapper89_copies2004 a h).1
  have c64129 := (publicBatchWrapper89_copies2004 a h).2.1
  have c64130 := (publicBatchWrapper89_copies2004 a h).2.2.1
  have c64131 := (publicBatchWrapper89_copies2004 a h).2.2.2.1
  have c64132 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.1
  have c64133 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_12
  simp only [← c64128, ← c64129, ← c64130] at e_1035_12
  have e_1035_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_13
  simp only [← c64131, ← c64132, k2, ← c64133] at e_1035_13
  have hr := e_1035_13
  simp only [e_1035_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8642)) := by
  have c64134 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.1
  have c64135 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.1
  have c64136 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.1
  have c64137 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.1
  have c64138 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64139 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_14
  simp only [← c64134, ← c64135, ← c64136] at e_1035_14
  have e_1035_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_15
  simp only [← c64137, ← c64138, k2, ← c64139] at e_1035_15
  have hr := e_1035_15
  simp only [e_1035_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8643)) := by
  have c64140 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64141 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64142 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64143 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64144 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64145 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_16
  simp only [← c64140, ← c64141, ← c64142] at e_1035_16
  have e_1035_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_17
  simp only [← c64143, ← c64144, k2, ← c64145] at e_1035_17
  have hr := e_1035_17
  simp only [e_1035_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1035 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8644)) := by
  have c64146 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64147 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64148 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64149 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64150 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64151 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1035_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_18
  simp only [← c64146, ← c64147, ← c64148] at e_1035_18
  have e_1035_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1035 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1035_19
  simp only [← c64149, ← c64150, k2, ← c64151] at e_1035_19
  have hr := e_1035_19
  simp only [e_1035_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8645)) := by
  have c64152 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64153 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64154 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64155 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64156 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64157 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_0
  simp only [← c64152, ← c64153, ← c64154] at e_1036_0
  have e_1036_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_1
  simp only [← c64155, ← c64156, k2, ← c64157] at e_1036_1
  have hr := e_1036_1
  simp only [e_1036_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8646)) := by
  have c64158 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64159 := (publicBatchWrapper89_copies2004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64160 := (publicBatchWrapper89_copies2005 a h).1
  have c64161 := (publicBatchWrapper89_copies2005 a h).2.1
  have c64162 := (publicBatchWrapper89_copies2005 a h).2.2.1
  have c64163 := (publicBatchWrapper89_copies2005 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_2
  simp only [← c64158, ← c64159, ← c64160] at e_1036_2
  have e_1036_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_3
  simp only [← c64161, ← c64162, k2, ← c64163] at e_1036_3
  have hr := e_1036_3
  simp only [e_1036_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8647)) := by
  have c64164 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.1
  have c64165 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.1
  have c64166 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.1
  have c64167 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.1
  have c64168 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.1
  have c64169 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_4
  simp only [← c64164, ← c64165, ← c64166] at e_1036_4
  have e_1036_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_5
  simp only [← c64167, ← c64168, k2, ← c64169] at e_1036_5
  have hr := e_1036_5
  simp only [e_1036_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8648)) := by
  have c64170 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64171 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64172 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64173 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64174 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64175 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_6
  simp only [← c64170, ← c64171, ← c64172] at e_1036_6
  have e_1036_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_7
  simp only [← c64173, ← c64174, k2, ← c64175] at e_1036_7
  have hr := e_1036_7
  simp only [e_1036_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8649)) := by
  have c64176 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64177 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64178 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64179 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64180 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64181 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_8
  simp only [← c64176, ← c64177, ← c64178] at e_1036_8
  have e_1036_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_9
  simp only [← c64179, ← c64180, k2, ← c64181] at e_1036_9
  have hr := e_1036_9
  simp only [e_1036_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8650)) := by
  have c64182 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64183 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64184 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64185 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64186 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64187 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_10
  simp only [← c64182, ← c64183, ← c64184] at e_1036_10
  have e_1036_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_11
  simp only [← c64185, ← c64186, k2, ← c64187] at e_1036_11
  have hr := e_1036_11
  simp only [e_1036_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8651)) := by
  have c64188 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64189 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64190 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64191 := (publicBatchWrapper89_copies2005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64192 := (publicBatchWrapper89_copies2006 a h).1
  have c64193 := (publicBatchWrapper89_copies2006 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_12
  simp only [← c64188, ← c64189, ← c64190] at e_1036_12
  have e_1036_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_13
  simp only [← c64191, ← c64192, k2, ← c64193] at e_1036_13
  have hr := e_1036_13
  simp only [e_1036_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8652)) := by
  have c64194 := (publicBatchWrapper89_copies2006 a h).2.2.1
  have c64195 := (publicBatchWrapper89_copies2006 a h).2.2.2.1
  have c64196 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.1
  have c64197 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.1
  have c64198 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.1
  have c64199 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_14
  simp only [← c64194, ← c64195, ← c64196] at e_1036_14
  have e_1036_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_15
  simp only [← c64197, ← c64198, k2, ← c64199] at e_1036_15
  have hr := e_1036_15
  simp only [e_1036_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8653)) := by
  have c64200 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.1
  have c64201 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.1
  have c64202 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64203 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64204 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64205 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_16
  simp only [← c64200, ← c64201, ← c64202] at e_1036_16
  have e_1036_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_17
  simp only [← c64203, ← c64204, k2, ← c64205] at e_1036_17
  have hr := e_1036_17
  simp only [e_1036_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1036 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8654)) := by
  have c64206 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64207 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64208 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64209 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64210 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64211 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1036_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_18
  simp only [← c64206, ← c64207, ← c64208] at e_1036_18
  have e_1036_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1036 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1036_19
  simp only [← c64209, ← c64210, k2, ← c64211] at e_1036_19
  have hr := e_1036_19
  simp only [e_1036_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8655)) := by
  have c64212 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64213 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64214 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64215 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64216 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64217 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_0
  simp only [← c64212, ← c64213, ← c64214] at e_1037_0
  have e_1037_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_1
  simp only [← c64215, ← c64216, k2, ← c64217] at e_1037_1
  have hr := e_1037_1
  simp only [e_1037_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8656)) := by
  have c64218 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64219 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64220 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64221 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64222 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64223 := (publicBatchWrapper89_copies2006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_2
  simp only [← c64218, ← c64219, ← c64220] at e_1037_2
  have e_1037_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_3
  simp only [← c64221, ← c64222, k2, ← c64223] at e_1037_3
  have hr := e_1037_3
  simp only [e_1037_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8657)) := by
  have c64224 := (publicBatchWrapper89_copies2007 a h).1
  have c64225 := (publicBatchWrapper89_copies2007 a h).2.1
  have c64226 := (publicBatchWrapper89_copies2007 a h).2.2.1
  have c64227 := (publicBatchWrapper89_copies2007 a h).2.2.2.1
  have c64228 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.1
  have c64229 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_4
  simp only [← c64224, ← c64225, ← c64226] at e_1037_4
  have e_1037_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_5
  simp only [← c64227, ← c64228, k2, ← c64229] at e_1037_5
  have hr := e_1037_5
  simp only [e_1037_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8658)) := by
  have c64230 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.1
  have c64231 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.1
  have c64232 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.1
  have c64233 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.1
  have c64234 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64235 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_6
  simp only [← c64230, ← c64231, ← c64232] at e_1037_6
  have e_1037_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_7
  simp only [← c64233, ← c64234, k2, ← c64235] at e_1037_7
  have hr := e_1037_7
  simp only [e_1037_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8659)) := by
  have c64236 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64237 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64238 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64239 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64240 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64241 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_8
  simp only [← c64236, ← c64237, ← c64238] at e_1037_8
  have e_1037_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_9
  simp only [← c64239, ← c64240, k2, ← c64241] at e_1037_9
  have hr := e_1037_9
  simp only [e_1037_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8660)) := by
  have c64242 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64243 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64244 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64245 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64246 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64247 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_10
  simp only [← c64242, ← c64243, ← c64244] at e_1037_10
  have e_1037_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_11
  simp only [← c64245, ← c64246, k2, ← c64247] at e_1037_11
  have hr := e_1037_11
  simp only [e_1037_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8661)) := by
  have c64248 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64249 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64250 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64251 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64252 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64253 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_12
  simp only [← c64248, ← c64249, ← c64250] at e_1037_12
  have e_1037_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_13
  simp only [← c64251, ← c64252, k2, ← c64253] at e_1037_13
  have hr := e_1037_13
  simp only [e_1037_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8662)) := by
  have c64254 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64255 := (publicBatchWrapper89_copies2007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64256 := (publicBatchWrapper89_copies2008 a h).1
  have c64257 := (publicBatchWrapper89_copies2008 a h).2.1
  have c64258 := (publicBatchWrapper89_copies2008 a h).2.2.1
  have c64259 := (publicBatchWrapper89_copies2008 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_14
  simp only [← c64254, ← c64255, ← c64256] at e_1037_14
  have e_1037_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_15
  simp only [← c64257, ← c64258, k2, ← c64259] at e_1037_15
  have hr := e_1037_15
  simp only [e_1037_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8663)) := by
  have c64260 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.1
  have c64261 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.1
  have c64262 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.1
  have c64263 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.1
  have c64264 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.1
  have c64265 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_16
  simp only [← c64260, ← c64261, ← c64262] at e_1037_16
  have e_1037_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_17
  simp only [← c64263, ← c64264, k2, ← c64265] at e_1037_17
  have hr := e_1037_17
  simp only [e_1037_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1037 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8664)) := by
  have c64266 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64267 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64268 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64269 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64270 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64271 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1037_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_18
  simp only [← c64266, ← c64267, ← c64268] at e_1037_18
  have e_1037_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1037 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1037_19
  simp only [← c64269, ← c64270, k2, ← c64271] at e_1037_19
  have hr := e_1037_19
  simp only [e_1037_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8665)) := by
  have c64272 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64273 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64274 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64275 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64276 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64277 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_0
  simp only [← c64272, ← c64273, ← c64274] at e_1038_0
  have e_1038_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_1
  simp only [← c64275, ← c64276, k2, ← c64277] at e_1038_1
  have hr := e_1038_1
  simp only [e_1038_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8666)) := by
  have c64278 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64279 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64280 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64281 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64282 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64283 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_2
  simp only [← c64278, ← c64279, ← c64280] at e_1038_2
  have e_1038_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_3
  simp only [← c64281, ← c64282, k2, ← c64283] at e_1038_3
  have hr := e_1038_3
  simp only [e_1038_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8667)) := by
  have c64284 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64285 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64286 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64287 := (publicBatchWrapper89_copies2008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64288 := (publicBatchWrapper89_copies2009 a h).1
  have c64289 := (publicBatchWrapper89_copies2009 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_4
  simp only [← c64284, ← c64285, ← c64286] at e_1038_4
  have e_1038_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_5
  simp only [← c64287, ← c64288, k2, ← c64289] at e_1038_5
  have hr := e_1038_5
  simp only [e_1038_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8668)) := by
  have c64290 := (publicBatchWrapper89_copies2009 a h).2.2.1
  have c64291 := (publicBatchWrapper89_copies2009 a h).2.2.2.1
  have c64292 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.1
  have c64293 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.1
  have c64294 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.1
  have c64295 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_6
  simp only [← c64290, ← c64291, ← c64292] at e_1038_6
  have e_1038_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_7
  simp only [← c64293, ← c64294, k2, ← c64295] at e_1038_7
  have hr := e_1038_7
  simp only [e_1038_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8669)) := by
  have c64296 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.1
  have c64297 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.1
  have c64298 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64299 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64300 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64301 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_8
  simp only [← c64296, ← c64297, ← c64298] at e_1038_8
  have e_1038_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_9
  simp only [← c64299, ← c64300, k2, ← c64301] at e_1038_9
  have hr := e_1038_9
  simp only [e_1038_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8670)) := by
  have c64302 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64303 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64304 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64305 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64306 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64307 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_10
  simp only [← c64302, ← c64303, ← c64304] at e_1038_10
  have e_1038_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_11
  simp only [← c64305, ← c64306, k2, ← c64307] at e_1038_11
  have hr := e_1038_11
  simp only [e_1038_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8671)) := by
  have c64308 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64309 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64310 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64311 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64312 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64313 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_12
  simp only [← c64308, ← c64309, ← c64310] at e_1038_12
  have e_1038_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_13
  simp only [← c64311, ← c64312, k2, ← c64313] at e_1038_13
  have hr := e_1038_13
  simp only [e_1038_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8672)) := by
  have c64314 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64315 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64316 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64317 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64318 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64319 := (publicBatchWrapper89_copies2009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_14
  simp only [← c64314, ← c64315, ← c64316] at e_1038_14
  have e_1038_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_15
  simp only [← c64317, ← c64318, k2, ← c64319] at e_1038_15
  have hr := e_1038_15
  simp only [e_1038_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8673)) := by
  have c64320 := (publicBatchWrapper89_copies2010 a h).1
  have c64321 := (publicBatchWrapper89_copies2010 a h).2.1
  have c64322 := (publicBatchWrapper89_copies2010 a h).2.2.1
  have c64323 := (publicBatchWrapper89_copies2010 a h).2.2.2.1
  have c64324 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.1
  have c64325 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_16
  simp only [← c64320, ← c64321, ← c64322] at e_1038_16
  have e_1038_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_17
  simp only [← c64323, ← c64324, k2, ← c64325] at e_1038_17
  have hr := e_1038_17
  simp only [e_1038_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1038 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8674)) := by
  have c64326 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.1
  have c64327 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.1
  have c64328 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.1
  have c64329 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.1
  have c64330 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64331 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1038_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_18
  simp only [← c64326, ← c64327, ← c64328] at e_1038_18
  have e_1038_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1038 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1038_19
  simp only [← c64329, ← c64330, k2, ← c64331] at e_1038_19
  have hr := e_1038_19
  simp only [e_1038_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8675)) := by
  have c64332 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64333 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64334 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64335 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64336 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64337 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_0
  simp only [← c64332, ← c64333, ← c64334] at e_1039_0
  have e_1039_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_1
  simp only [← c64335, ← c64336, k2, ← c64337] at e_1039_1
  have hr := e_1039_1
  simp only [e_1039_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8676)) := by
  have c64338 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64339 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64340 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64341 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64342 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64343 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_2
  simp only [← c64338, ← c64339, ← c64340] at e_1039_2
  have e_1039_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_3
  simp only [← c64341, ← c64342, k2, ← c64343] at e_1039_3
  have hr := e_1039_3
  simp only [e_1039_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8677)) := by
  have c64344 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64345 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64346 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64347 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64348 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64349 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_4
  simp only [← c64344, ← c64345, ← c64346] at e_1039_4
  have e_1039_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_5
  simp only [← c64347, ← c64348, k2, ← c64349] at e_1039_5
  have hr := e_1039_5
  simp only [e_1039_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8678)) := by
  have c64350 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64351 := (publicBatchWrapper89_copies2010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64352 := (publicBatchWrapper89_copies2011 a h).1
  have c64353 := (publicBatchWrapper89_copies2011 a h).2.1
  have c64354 := (publicBatchWrapper89_copies2011 a h).2.2.1
  have c64355 := (publicBatchWrapper89_copies2011 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_6
  simp only [← c64350, ← c64351, ← c64352] at e_1039_6
  have e_1039_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_7
  simp only [← c64353, ← c64354, k2, ← c64355] at e_1039_7
  have hr := e_1039_7
  simp only [e_1039_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8679)) := by
  have c64356 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.1
  have c64357 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.1
  have c64358 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.1
  have c64359 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.1
  have c64360 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.1
  have c64361 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_8
  simp only [← c64356, ← c64357, ← c64358] at e_1039_8
  have e_1039_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_9
  simp only [← c64359, ← c64360, k2, ← c64361] at e_1039_9
  have hr := e_1039_9
  simp only [e_1039_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8680)) := by
  have c64362 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64363 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64364 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64365 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64366 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64367 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_10
  simp only [← c64362, ← c64363, ← c64364] at e_1039_10
  have e_1039_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_11
  simp only [← c64365, ← c64366, k2, ← c64367] at e_1039_11
  have hr := e_1039_11
  simp only [e_1039_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8681)) := by
  have c64368 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64369 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64370 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64371 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64372 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64373 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_12
  simp only [← c64368, ← c64369, ← c64370] at e_1039_12
  have e_1039_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_13
  simp only [← c64371, ← c64372, k2, ← c64373] at e_1039_13
  have hr := e_1039_13
  simp only [e_1039_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8682)) := by
  have c64374 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64375 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64376 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64377 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64378 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64379 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_14
  simp only [← c64374, ← c64375, ← c64376] at e_1039_14
  have e_1039_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_15
  simp only [← c64377, ← c64378, k2, ← c64379] at e_1039_15
  have hr := e_1039_15
  simp only [e_1039_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8683)) := by
  have c64380 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64381 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64382 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64383 := (publicBatchWrapper89_copies2011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64384 := (publicBatchWrapper89_copies2012 a h).1
  have c64385 := (publicBatchWrapper89_copies2012 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_16
  simp only [← c64380, ← c64381, ← c64382] at e_1039_16
  have e_1039_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_17
  simp only [← c64383, ← c64384, k2, ← c64385] at e_1039_17
  have hr := e_1039_17
  simp only [e_1039_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1039 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8684)) := by
  have c64386 := (publicBatchWrapper89_copies2012 a h).2.2.1
  have c64387 := (publicBatchWrapper89_copies2012 a h).2.2.2.1
  have c64388 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.1
  have c64389 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.1
  have c64390 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.1
  have c64391 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1039_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_18
  simp only [← c64386, ← c64387, ← c64388] at e_1039_18
  have e_1039_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1039 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1039_19
  simp only [← c64389, ← c64390, k2, ← c64391] at e_1039_19
  have hr := e_1039_19
  simp only [e_1039_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8685)) := by
  have c64392 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.1
  have c64393 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.1
  have c64394 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64395 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64396 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64397 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_0
  simp only [← c64392, ← c64393, ← c64394] at e_1040_0
  have e_1040_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_1
  simp only [← c64395, ← c64396, k2, ← c64397] at e_1040_1
  have hr := e_1040_1
  simp only [e_1040_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8686)) := by
  have c64398 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64399 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64400 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64401 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64402 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64403 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_2
  simp only [← c64398, ← c64399, ← c64400] at e_1040_2
  have e_1040_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_3
  simp only [← c64401, ← c64402, k2, ← c64403] at e_1040_3
  have hr := e_1040_3
  simp only [e_1040_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8687)) := by
  have c64404 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64405 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64406 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64407 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64408 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64409 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_4
  simp only [← c64404, ← c64405, ← c64406] at e_1040_4
  have e_1040_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_5
  simp only [← c64407, ← c64408, k2, ← c64409] at e_1040_5
  have hr := e_1040_5
  simp only [e_1040_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8688)) := by
  have c64410 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64411 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64412 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64413 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64414 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64415 := (publicBatchWrapper89_copies2012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_6
  simp only [← c64410, ← c64411, ← c64412] at e_1040_6
  have e_1040_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_7
  simp only [← c64413, ← c64414, k2, ← c64415] at e_1040_7
  have hr := e_1040_7
  simp only [e_1040_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8689)) := by
  have c64416 := (publicBatchWrapper89_copies2013 a h).1
  have c64417 := (publicBatchWrapper89_copies2013 a h).2.1
  have c64418 := (publicBatchWrapper89_copies2013 a h).2.2.1
  have c64419 := (publicBatchWrapper89_copies2013 a h).2.2.2.1
  have c64420 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.1
  have c64421 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_8
  simp only [← c64416, ← c64417, ← c64418] at e_1040_8
  have e_1040_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_9
  simp only [← c64419, ← c64420, k2, ← c64421] at e_1040_9
  have hr := e_1040_9
  simp only [e_1040_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8690)) := by
  have c64422 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.1
  have c64423 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.1
  have c64424 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.1
  have c64425 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.1
  have c64426 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64427 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_10
  simp only [← c64422, ← c64423, ← c64424] at e_1040_10
  have e_1040_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_11
  simp only [← c64425, ← c64426, k2, ← c64427] at e_1040_11
  have hr := e_1040_11
  simp only [e_1040_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8691)) := by
  have c64428 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64429 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64430 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64431 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64432 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64433 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_12
  simp only [← c64428, ← c64429, ← c64430] at e_1040_12
  have e_1040_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_13
  simp only [← c64431, ← c64432, k2, ← c64433] at e_1040_13
  have hr := e_1040_13
  simp only [e_1040_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8692)) := by
  have c64434 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64435 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64436 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64437 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64438 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64439 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_14
  simp only [← c64434, ← c64435, ← c64436] at e_1040_14
  have e_1040_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_15
  simp only [← c64437, ← c64438, k2, ← c64439] at e_1040_15
  have hr := e_1040_15
  simp only [e_1040_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8693)) := by
  have c64440 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64441 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64442 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64443 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64444 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64445 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_16
  simp only [← c64440, ← c64441, ← c64442] at e_1040_16
  have e_1040_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_17
  simp only [← c64443, ← c64444, k2, ← c64445] at e_1040_17
  have hr := e_1040_17
  simp only [e_1040_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1040 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8694)) := by
  have c64446 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64447 := (publicBatchWrapper89_copies2013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64448 := (publicBatchWrapper89_copies2014 a h).1
  have c64449 := (publicBatchWrapper89_copies2014 a h).2.1
  have c64450 := (publicBatchWrapper89_copies2014 a h).2.2.1
  have c64451 := (publicBatchWrapper89_copies2014 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1040_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_18
  simp only [← c64446, ← c64447, ← c64448] at e_1040_18
  have e_1040_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1040 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1040_19
  simp only [← c64449, ← c64450, k2, ← c64451] at e_1040_19
  have hr := e_1040_19
  simp only [e_1040_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8695)) := by
  have c64452 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.1
  have c64453 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.1
  have c64454 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.1
  have c64455 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.1
  have c64456 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.1
  have c64457 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_0
  simp only [← c64452, ← c64453, ← c64454] at e_1041_0
  have e_1041_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_1
  simp only [← c64455, ← c64456, k2, ← c64457] at e_1041_1
  have hr := e_1041_1
  simp only [e_1041_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8696)) := by
  have c64458 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64459 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64460 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64461 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64462 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64463 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_2
  simp only [← c64458, ← c64459, ← c64460] at e_1041_2
  have e_1041_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_3
  simp only [← c64461, ← c64462, k2, ← c64463] at e_1041_3
  have hr := e_1041_3
  simp only [e_1041_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8697)) := by
  have c64464 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64465 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64466 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64467 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64468 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64469 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_4
  simp only [← c64464, ← c64465, ← c64466] at e_1041_4
  have e_1041_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_5
  simp only [← c64467, ← c64468, k2, ← c64469] at e_1041_5
  have hr := e_1041_5
  simp only [e_1041_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8698)) := by
  have c64470 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64471 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64472 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64473 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64474 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64475 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_6
  simp only [← c64470, ← c64471, ← c64472] at e_1041_6
  have e_1041_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_7
  simp only [← c64473, ← c64474, k2, ← c64475] at e_1041_7
  have hr := e_1041_7
  simp only [e_1041_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8699)) := by
  have c64476 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64477 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64478 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64479 := (publicBatchWrapper89_copies2014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64480 := (publicBatchWrapper89_copies2015 a h).1
  have c64481 := (publicBatchWrapper89_copies2015 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_8
  simp only [← c64476, ← c64477, ← c64478] at e_1041_8
  have e_1041_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_9
  simp only [← c64479, ← c64480, k2, ← c64481] at e_1041_9
  have hr := e_1041_9
  simp only [e_1041_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8700)) := by
  have c64482 := (publicBatchWrapper89_copies2015 a h).2.2.1
  have c64483 := (publicBatchWrapper89_copies2015 a h).2.2.2.1
  have c64484 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.1
  have c64485 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.1
  have c64486 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.1
  have c64487 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_10
  simp only [← c64482, ← c64483, ← c64484] at e_1041_10
  have e_1041_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_11
  simp only [← c64485, ← c64486, k2, ← c64487] at e_1041_11
  have hr := e_1041_11
  simp only [e_1041_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8701)) := by
  have c64488 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.1
  have c64489 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.1
  have c64490 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64491 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64492 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64493 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_12
  simp only [← c64488, ← c64489, ← c64490] at e_1041_12
  have e_1041_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_13
  simp only [← c64491, ← c64492, k2, ← c64493] at e_1041_13
  have hr := e_1041_13
  simp only [e_1041_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8702)) := by
  have c64494 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64495 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64496 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64497 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64498 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64499 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_14
  simp only [← c64494, ← c64495, ← c64496] at e_1041_14
  have e_1041_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_15
  simp only [← c64497, ← c64498, k2, ← c64499] at e_1041_15
  have hr := e_1041_15
  simp only [e_1041_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f9999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8703)) := by
  have c64500 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64501 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64502 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64503 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64504 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64505 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_16
  simp only [← c64500, ← c64501, ← c64502] at e_1041_16
  have e_1041_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_17
  simp only [← c64503, ← c64504, k2, ← c64505] at e_1041_17
  have hr := e_1041_17
  simp only [e_1041_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1041 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8704)) := by
  have c64506 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64507 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64508 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64509 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64510 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64511 := (publicBatchWrapper89_copies2015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1041_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_18
  simp only [← c64506, ← c64507, ← c64508] at e_1041_18
  have e_1041_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1041 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1041_19
  simp only [← c64509, ← c64510, k2, ← c64511] at e_1041_19
  have hr := e_1041_19
  simp only [e_1041_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8705)) := by
  have c64512 := (publicBatchWrapper89_copies2016 a h).1
  have c64513 := (publicBatchWrapper89_copies2016 a h).2.1
  have c64514 := (publicBatchWrapper89_copies2016 a h).2.2.1
  have c64515 := (publicBatchWrapper89_copies2016 a h).2.2.2.1
  have c64516 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.1
  have c64517 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_0
  simp only [← c64512, ← c64513, ← c64514] at e_1042_0
  have e_1042_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_1
  simp only [← c64515, ← c64516, k2, ← c64517] at e_1042_1
  have hr := e_1042_1
  simp only [e_1042_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8706)) := by
  have c64518 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.1
  have c64519 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.1
  have c64520 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.1
  have c64521 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.1
  have c64522 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64523 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_2
  simp only [← c64518, ← c64519, ← c64520] at e_1042_2
  have e_1042_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_3
  simp only [← c64521, ← c64522, k2, ← c64523] at e_1042_3
  have hr := e_1042_3
  simp only [e_1042_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8707)) := by
  have c64524 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64525 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64526 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64527 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64528 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64529 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_4
  simp only [← c64524, ← c64525, ← c64526] at e_1042_4
  have e_1042_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_5
  simp only [← c64527, ← c64528, k2, ← c64529] at e_1042_5
  have hr := e_1042_5
  simp only [e_1042_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8708)) := by
  have c64530 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64531 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64532 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64533 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64534 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64535 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_6
  simp only [← c64530, ← c64531, ← c64532] at e_1042_6
  have e_1042_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_7
  simp only [← c64533, ← c64534, k2, ← c64535] at e_1042_7
  have hr := e_1042_7
  simp only [e_1042_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8709)) := by
  have c64536 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64537 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64538 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64539 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64540 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64541 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_8
  simp only [← c64536, ← c64537, ← c64538] at e_1042_8
  have e_1042_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_9
  simp only [← c64539, ← c64540, k2, ← c64541] at e_1042_9
  have hr := e_1042_9
  simp only [e_1042_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8710)) := by
  have c64542 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64543 := (publicBatchWrapper89_copies2016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64544 := (publicBatchWrapper89_copies2017 a h).1
  have c64545 := (publicBatchWrapper89_copies2017 a h).2.1
  have c64546 := (publicBatchWrapper89_copies2017 a h).2.2.1
  have c64547 := (publicBatchWrapper89_copies2017 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_10
  simp only [← c64542, ← c64543, ← c64544] at e_1042_10
  have e_1042_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_11
  simp only [← c64545, ← c64546, k2, ← c64547] at e_1042_11
  have hr := e_1042_11
  simp only [e_1042_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8711)) := by
  have c64548 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.1
  have c64549 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.1
  have c64550 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.1
  have c64551 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.1
  have c64552 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.1
  have c64553 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_12
  simp only [← c64548, ← c64549, ← c64550] at e_1042_12
  have e_1042_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_13
  simp only [← c64551, ← c64552, k2, ← c64553] at e_1042_13
  have hr := e_1042_13
  simp only [e_1042_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8712)) := by
  have c64554 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64555 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64556 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64557 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64558 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64559 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_14
  simp only [← c64554, ← c64555, ← c64556] at e_1042_14
  have e_1042_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_15
  simp only [← c64557, ← c64558, k2, ← c64559] at e_1042_15
  have hr := e_1042_15
  simp only [e_1042_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8713)) := by
  have c64560 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64561 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64562 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64563 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64564 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64565 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_16
  simp only [← c64560, ← c64561, ← c64562] at e_1042_16
  have e_1042_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_17
  simp only [← c64563, ← c64564, k2, ← c64565] at e_1042_17
  have hr := e_1042_17
  simp only [e_1042_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1042 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8714)) := by
  have c64566 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64567 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64568 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64569 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64570 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64571 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1042_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_18
  simp only [← c64566, ← c64567, ← c64568] at e_1042_18
  have e_1042_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1042 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1042_19
  simp only [← c64569, ← c64570, k2, ← c64571] at e_1042_19
  have hr := e_1042_19
  simp only [e_1042_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8715)) := by
  have c64572 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64573 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64574 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64575 := (publicBatchWrapper89_copies2017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64576 := (publicBatchWrapper89_copies2018 a h).1
  have c64577 := (publicBatchWrapper89_copies2018 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_0
  simp only [← c64572, ← c64573, ← c64574] at e_1043_0
  have e_1043_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_1
  simp only [← c64575, ← c64576, k2, ← c64577] at e_1043_1
  have hr := e_1043_1
  simp only [e_1043_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8716)) := by
  have c64578 := (publicBatchWrapper89_copies2018 a h).2.2.1
  have c64579 := (publicBatchWrapper89_copies2018 a h).2.2.2.1
  have c64580 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.1
  have c64581 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.1
  have c64582 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.1
  have c64583 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_2
  simp only [← c64578, ← c64579, ← c64580] at e_1043_2
  have e_1043_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_3
  simp only [← c64581, ← c64582, k2, ← c64583] at e_1043_3
  have hr := e_1043_3
  simp only [e_1043_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8717)) := by
  have c64584 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.1
  have c64585 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.1
  have c64586 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64587 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64588 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64589 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_4
  simp only [← c64584, ← c64585, ← c64586] at e_1043_4
  have e_1043_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_5
  simp only [← c64587, ← c64588, k2, ← c64589] at e_1043_5
  have hr := e_1043_5
  simp only [e_1043_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8718)) := by
  have c64590 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64591 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64592 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64593 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64594 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64595 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_6
  simp only [← c64590, ← c64591, ← c64592] at e_1043_6
  have e_1043_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_7
  simp only [← c64593, ← c64594, k2, ← c64595] at e_1043_7
  have hr := e_1043_7
  simp only [e_1043_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8719)) := by
  have c64596 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64597 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64598 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64599 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64600 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64601 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_8
  simp only [← c64596, ← c64597, ← c64598] at e_1043_8
  have e_1043_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_9
  simp only [← c64599, ← c64600, k2, ← c64601] at e_1043_9
  have hr := e_1043_9
  simp only [e_1043_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8720)) := by
  have c64602 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64603 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64604 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64605 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64606 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64607 := (publicBatchWrapper89_copies2018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_10
  simp only [← c64602, ← c64603, ← c64604] at e_1043_10
  have e_1043_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_11
  simp only [← c64605, ← c64606, k2, ← c64607] at e_1043_11
  have hr := e_1043_11
  simp only [e_1043_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8721)) := by
  have c64608 := (publicBatchWrapper89_copies2019 a h).1
  have c64609 := (publicBatchWrapper89_copies2019 a h).2.1
  have c64610 := (publicBatchWrapper89_copies2019 a h).2.2.1
  have c64611 := (publicBatchWrapper89_copies2019 a h).2.2.2.1
  have c64612 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.1
  have c64613 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_12
  simp only [← c64608, ← c64609, ← c64610] at e_1043_12
  have e_1043_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_13
  simp only [← c64611, ← c64612, k2, ← c64613] at e_1043_13
  have hr := e_1043_13
  simp only [e_1043_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8722)) := by
  have c64614 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.1
  have c64615 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.1
  have c64616 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.1
  have c64617 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.1
  have c64618 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64619 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_14
  simp only [← c64614, ← c64615, ← c64616] at e_1043_14
  have e_1043_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_15
  simp only [← c64617, ← c64618, k2, ← c64619] at e_1043_15
  have hr := e_1043_15
  simp only [e_1043_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8723)) := by
  have c64620 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64621 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64622 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64623 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64624 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64625 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_16
  simp only [← c64620, ← c64621, ← c64622] at e_1043_16
  have e_1043_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_17
  simp only [← c64623, ← c64624, k2, ← c64625] at e_1043_17
  have hr := e_1043_17
  simp only [e_1043_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1043 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8724)) := by
  have c64626 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64627 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64628 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64629 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64630 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64631 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1043_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_18
  simp only [← c64626, ← c64627, ← c64628] at e_1043_18
  have e_1043_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1043 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1043_19
  simp only [← c64629, ← c64630, k2, ← c64631] at e_1043_19
  have hr := e_1043_19
  simp only [e_1043_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8725)) := by
  have c64632 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64633 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64634 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64635 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64636 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64637 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_0
  simp only [← c64632, ← c64633, ← c64634] at e_1044_0
  have e_1044_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_1
  simp only [← c64635, ← c64636, k2, ← c64637] at e_1044_1
  have hr := e_1044_1
  simp only [e_1044_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8726)) := by
  have c64638 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64639 := (publicBatchWrapper89_copies2019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64640 := (publicBatchWrapper89_copies2020 a h).1
  have c64641 := (publicBatchWrapper89_copies2020 a h).2.1
  have c64642 := (publicBatchWrapper89_copies2020 a h).2.2.1
  have c64643 := (publicBatchWrapper89_copies2020 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_2
  simp only [← c64638, ← c64639, ← c64640] at e_1044_2
  have e_1044_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_3
  simp only [← c64641, ← c64642, k2, ← c64643] at e_1044_3
  have hr := e_1044_3
  simp only [e_1044_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8727)) := by
  have c64644 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.1
  have c64645 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.1
  have c64646 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.1
  have c64647 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.1
  have c64648 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.1
  have c64649 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_4
  simp only [← c64644, ← c64645, ← c64646] at e_1044_4
  have e_1044_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_5
  simp only [← c64647, ← c64648, k2, ← c64649] at e_1044_5
  have hr := e_1044_5
  simp only [e_1044_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8728)) := by
  have c64650 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64651 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64652 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64653 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64654 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64655 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_6
  simp only [← c64650, ← c64651, ← c64652] at e_1044_6
  have e_1044_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_7
  simp only [← c64653, ← c64654, k2, ← c64655] at e_1044_7
  have hr := e_1044_7
  simp only [e_1044_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8729)) := by
  have c64656 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64657 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64658 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64659 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64660 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64661 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_8
  simp only [← c64656, ← c64657, ← c64658] at e_1044_8
  have e_1044_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_9
  simp only [← c64659, ← c64660, k2, ← c64661] at e_1044_9
  have hr := e_1044_9
  simp only [e_1044_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8730)) := by
  have c64662 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64663 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64664 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64665 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64666 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64667 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_10
  simp only [← c64662, ← c64663, ← c64664] at e_1044_10
  have e_1044_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_11
  simp only [← c64665, ← c64666, k2, ← c64667] at e_1044_11
  have hr := e_1044_11
  simp only [e_1044_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8731)) := by
  have c64668 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64669 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64670 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64671 := (publicBatchWrapper89_copies2020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64672 := (publicBatchWrapper89_copies2021 a h).1
  have c64673 := (publicBatchWrapper89_copies2021 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_12
  simp only [← c64668, ← c64669, ← c64670] at e_1044_12
  have e_1044_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_13
  simp only [← c64671, ← c64672, k2, ← c64673] at e_1044_13
  have hr := e_1044_13
  simp only [e_1044_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8732)) := by
  have c64674 := (publicBatchWrapper89_copies2021 a h).2.2.1
  have c64675 := (publicBatchWrapper89_copies2021 a h).2.2.2.1
  have c64676 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.1
  have c64677 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.1
  have c64678 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.1
  have c64679 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_14
  simp only [← c64674, ← c64675, ← c64676] at e_1044_14
  have e_1044_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_15
  simp only [← c64677, ← c64678, k2, ← c64679] at e_1044_15
  have hr := e_1044_15
  simp only [e_1044_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8733)) := by
  have c64680 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.1
  have c64681 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.1
  have c64682 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64683 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64684 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64685 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_16
  simp only [← c64680, ← c64681, ← c64682] at e_1044_16
  have e_1044_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_17
  simp only [← c64683, ← c64684, k2, ← c64685] at e_1044_17
  have hr := e_1044_17
  simp only [e_1044_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1044 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8734)) := by
  have c64686 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64687 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64688 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64689 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64690 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64691 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1044_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_18
  simp only [← c64686, ← c64687, ← c64688] at e_1044_18
  have e_1044_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1044 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1044_19
  simp only [← c64689, ← c64690, k2, ← c64691] at e_1044_19
  have hr := e_1044_19
  simp only [e_1044_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8735)) := by
  have c64692 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64693 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64694 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64695 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64696 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64697 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_0
  simp only [← c64692, ← c64693, ← c64694] at e_1045_0
  have e_1045_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_1
  simp only [← c64695, ← c64696, k2, ← c64697] at e_1045_1
  have hr := e_1045_1
  simp only [e_1045_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8736)) := by
  have c64698 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64699 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64700 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64701 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64702 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64703 := (publicBatchWrapper89_copies2021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_2
  simp only [← c64698, ← c64699, ← c64700] at e_1045_2
  have e_1045_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_3
  simp only [← c64701, ← c64702, k2, ← c64703] at e_1045_3
  have hr := e_1045_3
  simp only [e_1045_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8737)) := by
  have c64704 := (publicBatchWrapper89_copies2022 a h).1
  have c64705 := (publicBatchWrapper89_copies2022 a h).2.1
  have c64706 := (publicBatchWrapper89_copies2022 a h).2.2.1
  have c64707 := (publicBatchWrapper89_copies2022 a h).2.2.2.1
  have c64708 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.1
  have c64709 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_4
  simp only [← c64704, ← c64705, ← c64706] at e_1045_4
  have e_1045_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_5
  simp only [← c64707, ← c64708, k2, ← c64709] at e_1045_5
  have hr := e_1045_5
  simp only [e_1045_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8738)) := by
  have c64710 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.1
  have c64711 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.1
  have c64712 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.1
  have c64713 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.1
  have c64714 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64715 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_6
  simp only [← c64710, ← c64711, ← c64712] at e_1045_6
  have e_1045_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_7
  simp only [← c64713, ← c64714, k2, ← c64715] at e_1045_7
  have hr := e_1045_7
  simp only [e_1045_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8739)) := by
  have c64716 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64717 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64718 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64719 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64720 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64721 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_8
  simp only [← c64716, ← c64717, ← c64718] at e_1045_8
  have e_1045_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_9
  simp only [← c64719, ← c64720, k2, ← c64721] at e_1045_9
  have hr := e_1045_9
  simp only [e_1045_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8740)) := by
  have c64722 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64723 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64724 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64725 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64726 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64727 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_10
  simp only [← c64722, ← c64723, ← c64724] at e_1045_10
  have e_1045_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_11
  simp only [← c64725, ← c64726, k2, ← c64727] at e_1045_11
  have hr := e_1045_11
  simp only [e_1045_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8741)) := by
  have c64728 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64729 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64730 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64731 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64732 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64733 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_12
  simp only [← c64728, ← c64729, ← c64730] at e_1045_12
  have e_1045_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_13
  simp only [← c64731, ← c64732, k2, ← c64733] at e_1045_13
  have hr := e_1045_13
  simp only [e_1045_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8742)) := by
  have c64734 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64735 := (publicBatchWrapper89_copies2022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64736 := (publicBatchWrapper89_copies2023 a h).1
  have c64737 := (publicBatchWrapper89_copies2023 a h).2.1
  have c64738 := (publicBatchWrapper89_copies2023 a h).2.2.1
  have c64739 := (publicBatchWrapper89_copies2023 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_14
  simp only [← c64734, ← c64735, ← c64736] at e_1045_14
  have e_1045_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_15
  simp only [← c64737, ← c64738, k2, ← c64739] at e_1045_15
  have hr := e_1045_15
  simp only [e_1045_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8743)) := by
  have c64740 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.1
  have c64741 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.1
  have c64742 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.1
  have c64743 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.1
  have c64744 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.1
  have c64745 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_16
  simp only [← c64740, ← c64741, ← c64742] at e_1045_16
  have e_1045_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_17
  simp only [← c64743, ← c64744, k2, ← c64745] at e_1045_17
  have hr := e_1045_17
  simp only [e_1045_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1045 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8744)) := by
  have c64746 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64747 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64748 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64749 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64750 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64751 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1045_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_18
  simp only [← c64746, ← c64747, ← c64748] at e_1045_18
  have e_1045_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1045 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1045_19
  simp only [← c64749, ← c64750, k2, ← c64751] at e_1045_19
  have hr := e_1045_19
  simp only [e_1045_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8745)) := by
  have c64752 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64753 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64754 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64755 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64756 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64757 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_0
  simp only [← c64752, ← c64753, ← c64754] at e_1046_0
  have e_1046_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_1
  simp only [← c64755, ← c64756, k2, ← c64757] at e_1046_1
  have hr := e_1046_1
  simp only [e_1046_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8746)) := by
  have c64758 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64759 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64760 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64761 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64762 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64763 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_2
  simp only [← c64758, ← c64759, ← c64760] at e_1046_2
  have e_1046_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_3
  simp only [← c64761, ← c64762, k2, ← c64763] at e_1046_3
  have hr := e_1046_3
  simp only [e_1046_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8747)) := by
  have c64764 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64765 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64766 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64767 := (publicBatchWrapper89_copies2023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64768 := (publicBatchWrapper89_copies2024 a h).1
  have c64769 := (publicBatchWrapper89_copies2024 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_4
  simp only [← c64764, ← c64765, ← c64766] at e_1046_4
  have e_1046_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_5
  simp only [← c64767, ← c64768, k2, ← c64769] at e_1046_5
  have hr := e_1046_5
  simp only [e_1046_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8748)) := by
  have c64770 := (publicBatchWrapper89_copies2024 a h).2.2.1
  have c64771 := (publicBatchWrapper89_copies2024 a h).2.2.2.1
  have c64772 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.1
  have c64773 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.1
  have c64774 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.1
  have c64775 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_6
  simp only [← c64770, ← c64771, ← c64772] at e_1046_6
  have e_1046_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_7
  simp only [← c64773, ← c64774, k2, ← c64775] at e_1046_7
  have hr := e_1046_7
  simp only [e_1046_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8749)) := by
  have c64776 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.1
  have c64777 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.1
  have c64778 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64779 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64780 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64781 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_8
  simp only [← c64776, ← c64777, ← c64778] at e_1046_8
  have e_1046_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_9
  simp only [← c64779, ← c64780, k2, ← c64781] at e_1046_9
  have hr := e_1046_9
  simp only [e_1046_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8750)) := by
  have c64782 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64783 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64784 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64785 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64786 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64787 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_10
  simp only [← c64782, ← c64783, ← c64784] at e_1046_10
  have e_1046_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_11
  simp only [← c64785, ← c64786, k2, ← c64787] at e_1046_11
  have hr := e_1046_11
  simp only [e_1046_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8751)) := by
  have c64788 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64789 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64790 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64791 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64792 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64793 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_12
  simp only [← c64788, ← c64789, ← c64790] at e_1046_12
  have e_1046_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_13
  simp only [← c64791, ← c64792, k2, ← c64793] at e_1046_13
  have hr := e_1046_13
  simp only [e_1046_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8752)) := by
  have c64794 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64795 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64796 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64797 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64798 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64799 := (publicBatchWrapper89_copies2024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_14
  simp only [← c64794, ← c64795, ← c64796] at e_1046_14
  have e_1046_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_15
  simp only [← c64797, ← c64798, k2, ← c64799] at e_1046_15
  have hr := e_1046_15
  simp only [e_1046_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8753)) := by
  have c64800 := (publicBatchWrapper89_copies2025 a h).1
  have c64801 := (publicBatchWrapper89_copies2025 a h).2.1
  have c64802 := (publicBatchWrapper89_copies2025 a h).2.2.1
  have c64803 := (publicBatchWrapper89_copies2025 a h).2.2.2.1
  have c64804 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.1
  have c64805 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_16
  simp only [← c64800, ← c64801, ← c64802] at e_1046_16
  have e_1046_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_17
  simp only [← c64803, ← c64804, k2, ← c64805] at e_1046_17
  have hr := e_1046_17
  simp only [e_1046_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1046 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8754)) := by
  have c64806 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.1
  have c64807 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.1
  have c64808 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.1
  have c64809 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.1
  have c64810 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64811 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1046_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_18
  simp only [← c64806, ← c64807, ← c64808] at e_1046_18
  have e_1046_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1046 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1046_19
  simp only [← c64809, ← c64810, k2, ← c64811] at e_1046_19
  have hr := e_1046_19
  simp only [e_1046_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8755)) := by
  have c64812 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64813 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64814 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64815 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64816 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64817 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_0
  simp only [← c64812, ← c64813, ← c64814] at e_1047_0
  have e_1047_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_1
  simp only [← c64815, ← c64816, k2, ← c64817] at e_1047_1
  have hr := e_1047_1
  simp only [e_1047_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8756)) := by
  have c64818 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64819 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64820 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64821 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64822 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64823 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_2
  simp only [← c64818, ← c64819, ← c64820] at e_1047_2
  have e_1047_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_3
  simp only [← c64821, ← c64822, k2, ← c64823] at e_1047_3
  have hr := e_1047_3
  simp only [e_1047_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8757)) := by
  have c64824 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64825 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64826 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64827 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64828 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64829 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_4
  simp only [← c64824, ← c64825, ← c64826] at e_1047_4
  have e_1047_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_5
  simp only [← c64827, ← c64828, k2, ← c64829] at e_1047_5
  have hr := e_1047_5
  simp only [e_1047_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8758)) := by
  have c64830 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64831 := (publicBatchWrapper89_copies2025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64832 := (publicBatchWrapper89_copies2026 a h).1
  have c64833 := (publicBatchWrapper89_copies2026 a h).2.1
  have c64834 := (publicBatchWrapper89_copies2026 a h).2.2.1
  have c64835 := (publicBatchWrapper89_copies2026 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_6
  simp only [← c64830, ← c64831, ← c64832] at e_1047_6
  have e_1047_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_7
  simp only [← c64833, ← c64834, k2, ← c64835] at e_1047_7
  have hr := e_1047_7
  simp only [e_1047_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8759)) := by
  have c64836 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.1
  have c64837 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.1
  have c64838 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.1
  have c64839 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.1
  have c64840 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.1
  have c64841 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_8
  simp only [← c64836, ← c64837, ← c64838] at e_1047_8
  have e_1047_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_9
  simp only [← c64839, ← c64840, k2, ← c64841] at e_1047_9
  have hr := e_1047_9
  simp only [e_1047_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8760)) := by
  have c64842 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64843 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64844 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64845 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64846 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64847 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_10
  simp only [← c64842, ← c64843, ← c64844] at e_1047_10
  have e_1047_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_11
  simp only [← c64845, ← c64846, k2, ← c64847] at e_1047_11
  have hr := e_1047_11
  simp only [e_1047_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8761)) := by
  have c64848 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64849 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64850 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64851 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64852 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64853 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_12
  simp only [← c64848, ← c64849, ← c64850] at e_1047_12
  have e_1047_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_13
  simp only [← c64851, ← c64852, k2, ← c64853] at e_1047_13
  have hr := e_1047_13
  simp only [e_1047_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8762)) := by
  have c64854 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64855 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64856 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64857 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64858 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64859 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_14
  simp only [← c64854, ← c64855, ← c64856] at e_1047_14
  have e_1047_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_15
  simp only [← c64857, ← c64858, k2, ← c64859] at e_1047_15
  have hr := e_1047_15
  simp only [e_1047_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8763)) := by
  have c64860 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64861 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64862 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64863 := (publicBatchWrapper89_copies2026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64864 := (publicBatchWrapper89_copies2027 a h).1
  have c64865 := (publicBatchWrapper89_copies2027 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_16
  simp only [← c64860, ← c64861, ← c64862] at e_1047_16
  have e_1047_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_17
  simp only [← c64863, ← c64864, k2, ← c64865] at e_1047_17
  have hr := e_1047_17
  simp only [e_1047_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1047 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8764)) := by
  have c64866 := (publicBatchWrapper89_copies2027 a h).2.2.1
  have c64867 := (publicBatchWrapper89_copies2027 a h).2.2.2.1
  have c64868 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.1
  have c64869 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.1
  have c64870 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.1
  have c64871 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1047_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_18
  simp only [← c64866, ← c64867, ← c64868] at e_1047_18
  have e_1047_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1047 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1047_19
  simp only [← c64869, ← c64870, k2, ← c64871] at e_1047_19
  have hr := e_1047_19
  simp only [e_1047_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8765)) := by
  have c64872 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.1
  have c64873 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.1
  have c64874 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64875 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64876 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64877 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_0
  simp only [← c64872, ← c64873, ← c64874] at e_1048_0
  have e_1048_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_1
  simp only [← c64875, ← c64876, k2, ← c64877] at e_1048_1
  have hr := e_1048_1
  simp only [e_1048_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8766)) := by
  have c64878 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64879 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64880 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64881 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64882 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64883 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_2
  simp only [← c64878, ← c64879, ← c64880] at e_1048_2
  have e_1048_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_3
  simp only [← c64881, ← c64882, k2, ← c64883] at e_1048_3
  have hr := e_1048_3
  simp only [e_1048_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8767)) := by
  have c64884 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64885 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64886 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64887 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64888 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64889 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_4
  simp only [← c64884, ← c64885, ← c64886] at e_1048_4
  have e_1048_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_5
  simp only [← c64887, ← c64888, k2, ← c64889] at e_1048_5
  have hr := e_1048_5
  simp only [e_1048_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8768)) := by
  have c64890 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64891 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64892 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64893 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64894 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64895 := (publicBatchWrapper89_copies2027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_6
  simp only [← c64890, ← c64891, ← c64892] at e_1048_6
  have e_1048_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_7
  simp only [← c64893, ← c64894, k2, ← c64895] at e_1048_7
  have hr := e_1048_7
  simp only [e_1048_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8769)) := by
  have c64896 := (publicBatchWrapper89_copies2028 a h).1
  have c64897 := (publicBatchWrapper89_copies2028 a h).2.1
  have c64898 := (publicBatchWrapper89_copies2028 a h).2.2.1
  have c64899 := (publicBatchWrapper89_copies2028 a h).2.2.2.1
  have c64900 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.1
  have c64901 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_8
  simp only [← c64896, ← c64897, ← c64898] at e_1048_8
  have e_1048_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_9
  simp only [← c64899, ← c64900, k2, ← c64901] at e_1048_9
  have hr := e_1048_9
  simp only [e_1048_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8810)) := by
  have c64902 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.1
  have c64903 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.1
  have c64904 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.1
  have c64905 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.1
  have c64906 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64907 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_10
  simp only [← c64902, ← c64903, ← c64904] at e_1048_10
  have e_1048_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_11
  simp only [← c64905, ← c64906, k2, ← c64907] at e_1048_11
  have hr := e_1048_11
  simp only [e_1048_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8811)) := by
  have c64908 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64909 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64910 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64911 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64912 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64913 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_12
  simp only [← c64908, ← c64909, ← c64910] at e_1048_12
  have e_1048_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_13
  simp only [← c64911, ← c64912, k2, ← c64913] at e_1048_13
  have hr := e_1048_13
  simp only [e_1048_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8812)) := by
  have c64914 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64915 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64916 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64917 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64918 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64919 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_14
  simp only [← c64914, ← c64915, ← c64916] at e_1048_14
  have e_1048_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_15
  simp only [← c64917, ← c64918, k2, ← c64919] at e_1048_15
  have hr := e_1048_15
  simp only [e_1048_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8813)) := by
  have c64920 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64921 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64922 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64923 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64924 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64925 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_16
  simp only [← c64920, ← c64921, ← c64922] at e_1048_16
  have e_1048_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_17
  simp only [← c64923, ← c64924, k2, ← c64925] at e_1048_17
  have hr := e_1048_17
  simp only [e_1048_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1048 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8814)) := by
  have c64926 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64927 := (publicBatchWrapper89_copies2028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64928 := (publicBatchWrapper89_copies2029 a h).1
  have c64929 := (publicBatchWrapper89_copies2029 a h).2.1
  have c64930 := (publicBatchWrapper89_copies2029 a h).2.2.1
  have c64931 := (publicBatchWrapper89_copies2029 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1048_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_18
  simp only [← c64926, ← c64927, ← c64928] at e_1048_18
  have e_1048_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1048 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1048_19
  simp only [← c64929, ← c64930, k2, ← c64931] at e_1048_19
  have hr := e_1048_19
  simp only [e_1048_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8815)) := by
  have c64932 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.1
  have c64933 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.1
  have c64934 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.1
  have c64935 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.1
  have c64936 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.1
  have c64937 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_0
  simp only [← c64932, ← c64933, ← c64934] at e_1049_0
  have e_1049_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_1
  simp only [← c64935, ← c64936, k2, ← c64937] at e_1049_1
  have hr := e_1049_1
  simp only [e_1049_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8816)) := by
  have c64938 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64939 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64940 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64941 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64942 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64943 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_2
  simp only [← c64938, ← c64939, ← c64940] at e_1049_2
  have e_1049_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_3
  simp only [← c64941, ← c64942, k2, ← c64943] at e_1049_3
  have hr := e_1049_3
  simp only [e_1049_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8817)) := by
  have c64944 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64945 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64946 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64947 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64948 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64949 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_4
  simp only [← c64944, ← c64945, ← c64946] at e_1049_4
  have e_1049_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_5
  simp only [← c64947, ← c64948, k2, ← c64949] at e_1049_5
  have hr := e_1049_5
  simp only [e_1049_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8818)) := by
  have c64950 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64951 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64952 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64953 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64954 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64955 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_6
  simp only [← c64950, ← c64951, ← c64952] at e_1049_6
  have e_1049_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_7
  simp only [← c64953, ← c64954, k2, ← c64955] at e_1049_7
  have hr := e_1049_7
  simp only [e_1049_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8819)) := by
  have c64956 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64957 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64958 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64959 := (publicBatchWrapper89_copies2029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c64960 := (publicBatchWrapper89_copies2030 a h).1
  have c64961 := (publicBatchWrapper89_copies2030 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_8
  simp only [← c64956, ← c64957, ← c64958] at e_1049_8
  have e_1049_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_9
  simp only [← c64959, ← c64960, k2, ← c64961] at e_1049_9
  have hr := e_1049_9
  simp only [e_1049_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8820)) := by
  have c64962 := (publicBatchWrapper89_copies2030 a h).2.2.1
  have c64963 := (publicBatchWrapper89_copies2030 a h).2.2.2.1
  have c64964 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.1
  have c64965 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.1
  have c64966 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.1
  have c64967 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_10
  simp only [← c64962, ← c64963, ← c64964] at e_1049_10
  have e_1049_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_11
  simp only [← c64965, ← c64966, k2, ← c64967] at e_1049_11
  have hr := e_1049_11
  simp only [e_1049_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8821)) := by
  have c64968 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.1
  have c64969 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.1
  have c64970 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.1
  have c64971 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c64972 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64973 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_12
  simp only [← c64968, ← c64969, ← c64970] at e_1049_12
  have e_1049_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_13
  simp only [← c64971, ← c64972, k2, ← c64973] at e_1049_13
  have hr := e_1049_13
  simp only [e_1049_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8822)) := by
  have c64974 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64975 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64976 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64977 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64978 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64979 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_14
  simp only [← c64974, ← c64975, ← c64976] at e_1049_14
  have e_1049_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_15
  simp only [← c64977, ← c64978, k2, ← c64979] at e_1049_15
  have hr := e_1049_15
  simp only [e_1049_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8823)) := by
  have c64980 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64981 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64982 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64983 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64984 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64985 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_16
  simp only [← c64980, ← c64981, ← c64982] at e_1049_16
  have e_1049_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_17
  simp only [← c64983, ← c64984, k2, ← c64985] at e_1049_17
  have hr := e_1049_17
  simp only [e_1049_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1049 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8824)) := by
  have c64986 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64987 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64988 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64989 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64990 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c64991 := (publicBatchWrapper89_copies2030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1049_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_18
  simp only [← c64986, ← c64987, ← c64988] at e_1049_18
  have e_1049_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1049 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1049_19
  simp only [← c64989, ← c64990, k2, ← c64991] at e_1049_19
  have hr := e_1049_19
  simp only [e_1049_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8825)) := by
  have c64992 := (publicBatchWrapper89_copies2031 a h).1
  have c64993 := (publicBatchWrapper89_copies2031 a h).2.1
  have c64994 := (publicBatchWrapper89_copies2031 a h).2.2.1
  have c64995 := (publicBatchWrapper89_copies2031 a h).2.2.2.1
  have c64996 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.1
  have c64997 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_0
  simp only [← c64992, ← c64993, ← c64994] at e_1050_0
  have e_1050_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_1
  simp only [← c64995, ← c64996, k2, ← c64997] at e_1050_1
  have hr := e_1050_1
  simp only [e_1050_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8826)) := by
  have c64998 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.1
  have c64999 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.1
  have c65000 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.1
  have c65001 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.1
  have c65002 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65003 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_2
  simp only [← c64998, ← c64999, ← c65000] at e_1050_2
  have e_1050_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_3
  simp only [← c65001, ← c65002, k2, ← c65003] at e_1050_3
  have hr := e_1050_3
  simp only [e_1050_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8827)) := by
  have c65004 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65005 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65006 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65007 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65008 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65009 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_4
  simp only [← c65004, ← c65005, ← c65006] at e_1050_4
  have e_1050_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_5
  simp only [← c65007, ← c65008, k2, ← c65009] at e_1050_5
  have hr := e_1050_5
  simp only [e_1050_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8828)) := by
  have c65010 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65011 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65012 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65013 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65014 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65015 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_6
  simp only [← c65010, ← c65011, ← c65012] at e_1050_6
  have e_1050_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_7
  simp only [← c65013, ← c65014, k2, ← c65015] at e_1050_7
  have hr := e_1050_7
  simp only [e_1050_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8829)) := by
  have c65016 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65017 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65018 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65019 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65020 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65021 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_8
  simp only [← c65016, ← c65017, ← c65018] at e_1050_8
  have e_1050_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_9
  simp only [← c65019, ← c65020, k2, ← c65021] at e_1050_9
  have hr := e_1050_9
  simp only [e_1050_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8830)) := by
  have c65022 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65023 := (publicBatchWrapper89_copies2031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65024 := (publicBatchWrapper89_copies2032 a h).1
  have c65025 := (publicBatchWrapper89_copies2032 a h).2.1
  have c65026 := (publicBatchWrapper89_copies2032 a h).2.2.1
  have c65027 := (publicBatchWrapper89_copies2032 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_10
  simp only [← c65022, ← c65023, ← c65024] at e_1050_10
  have e_1050_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_11
  simp only [← c65025, ← c65026, k2, ← c65027] at e_1050_11
  have hr := e_1050_11
  simp only [e_1050_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8831)) := by
  have c65028 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.1
  have c65029 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.1
  have c65030 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.1
  have c65031 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.1
  have c65032 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.1
  have c65033 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_12
  simp only [← c65028, ← c65029, ← c65030] at e_1050_12
  have e_1050_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_13
  simp only [← c65031, ← c65032, k2, ← c65033] at e_1050_13
  have hr := e_1050_13
  simp only [e_1050_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8832)) := by
  have c65034 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65035 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65036 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65037 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65038 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65039 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_14
  simp only [← c65034, ← c65035, ← c65036] at e_1050_14
  have e_1050_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_15
  simp only [← c65037, ← c65038, k2, ← c65039] at e_1050_15
  have hr := e_1050_15
  simp only [e_1050_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8833)) := by
  have c65040 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65041 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65042 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65043 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65044 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65045 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_16
  simp only [← c65040, ← c65041, ← c65042] at e_1050_16
  have e_1050_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_17
  simp only [← c65043, ← c65044, k2, ← c65045] at e_1050_17
  have hr := e_1050_17
  simp only [e_1050_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1050 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8834)) := by
  have c65046 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65047 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65048 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65049 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65050 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65051 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1050_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_18
  simp only [← c65046, ← c65047, ← c65048] at e_1050_18
  have e_1050_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1050 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1050_19
  simp only [← c65049, ← c65050, k2, ← c65051] at e_1050_19
  have hr := e_1050_19
  simp only [e_1050_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8835)) := by
  have c65052 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65053 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65054 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65055 := (publicBatchWrapper89_copies2032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65056 := (publicBatchWrapper89_copies2033 a h).1
  have c65057 := (publicBatchWrapper89_copies2033 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_0
  simp only [← c65052, ← c65053, ← c65054] at e_1051_0
  have e_1051_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_1
  simp only [← c65055, ← c65056, k2, ← c65057] at e_1051_1
  have hr := e_1051_1
  simp only [e_1051_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8836)) := by
  have c65058 := (publicBatchWrapper89_copies2033 a h).2.2.1
  have c65059 := (publicBatchWrapper89_copies2033 a h).2.2.2.1
  have c65060 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.1
  have c65061 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.1
  have c65062 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.1
  have c65063 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_2
  simp only [← c65058, ← c65059, ← c65060] at e_1051_2
  have e_1051_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_3
  simp only [← c65061, ← c65062, k2, ← c65063] at e_1051_3
  have hr := e_1051_3
  simp only [e_1051_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8837)) := by
  have c65064 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.1
  have c65065 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.1
  have c65066 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65067 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65068 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65069 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_4
  simp only [← c65064, ← c65065, ← c65066] at e_1051_4
  have e_1051_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_5
  simp only [← c65067, ← c65068, k2, ← c65069] at e_1051_5
  have hr := e_1051_5
  simp only [e_1051_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8838)) := by
  have c65070 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65071 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65072 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65073 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65074 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65075 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_6
  simp only [← c65070, ← c65071, ← c65072] at e_1051_6
  have e_1051_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_7
  simp only [← c65073, ← c65074, k2, ← c65075] at e_1051_7
  have hr := e_1051_7
  simp only [e_1051_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8839)) := by
  have c65076 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65077 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65078 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65079 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65080 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65081 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_8
  simp only [← c65076, ← c65077, ← c65078] at e_1051_8
  have e_1051_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_9
  simp only [← c65079, ← c65080, k2, ← c65081] at e_1051_9
  have hr := e_1051_9
  simp only [e_1051_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8840)) := by
  have c65082 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65083 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65084 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65085 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65086 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65087 := (publicBatchWrapper89_copies2033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_10
  simp only [← c65082, ← c65083, ← c65084] at e_1051_10
  have e_1051_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_11
  simp only [← c65085, ← c65086, k2, ← c65087] at e_1051_11
  have hr := e_1051_11
  simp only [e_1051_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8841)) := by
  have c65088 := (publicBatchWrapper89_copies2034 a h).1
  have c65089 := (publicBatchWrapper89_copies2034 a h).2.1
  have c65090 := (publicBatchWrapper89_copies2034 a h).2.2.1
  have c65091 := (publicBatchWrapper89_copies2034 a h).2.2.2.1
  have c65092 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.1
  have c65093 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_12
  simp only [← c65088, ← c65089, ← c65090] at e_1051_12
  have e_1051_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_13
  simp only [← c65091, ← c65092, k2, ← c65093] at e_1051_13
  have hr := e_1051_13
  simp only [e_1051_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8842)) := by
  have c65094 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.1
  have c65095 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.1
  have c65096 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.1
  have c65097 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.1
  have c65098 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65099 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_14
  simp only [← c65094, ← c65095, ← c65096] at e_1051_14
  have e_1051_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_15
  simp only [← c65097, ← c65098, k2, ← c65099] at e_1051_15
  have hr := e_1051_15
  simp only [e_1051_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8843)) := by
  have c65100 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65101 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65102 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65103 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65104 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65105 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_16
  simp only [← c65100, ← c65101, ← c65102] at e_1051_16
  have e_1051_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_17
  simp only [← c65103, ← c65104, k2, ← c65105] at e_1051_17
  have hr := e_1051_17
  simp only [e_1051_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1051 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8844)) := by
  have c65106 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65107 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65108 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65109 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65110 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65111 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1051_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_18
  simp only [← c65106, ← c65107, ← c65108] at e_1051_18
  have e_1051_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1051 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1051_19
  simp only [← c65109, ← c65110, k2, ← c65111] at e_1051_19
  have hr := e_1051_19
  simp only [e_1051_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8845)) := by
  have c65112 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65113 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65114 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65115 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65116 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65117 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_0
  simp only [← c65112, ← c65113, ← c65114] at e_1052_0
  have e_1052_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_1
  simp only [← c65115, ← c65116, k2, ← c65117] at e_1052_1
  have hr := e_1052_1
  simp only [e_1052_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8846)) := by
  have c65118 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65119 := (publicBatchWrapper89_copies2034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65120 := (publicBatchWrapper89_copies2035 a h).1
  have c65121 := (publicBatchWrapper89_copies2035 a h).2.1
  have c65122 := (publicBatchWrapper89_copies2035 a h).2.2.1
  have c65123 := (publicBatchWrapper89_copies2035 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_2
  simp only [← c65118, ← c65119, ← c65120] at e_1052_2
  have e_1052_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_3
  simp only [← c65121, ← c65122, k2, ← c65123] at e_1052_3
  have hr := e_1052_3
  simp only [e_1052_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8847)) := by
  have c65124 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.1
  have c65125 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.1
  have c65126 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.1
  have c65127 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.1
  have c65128 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.1
  have c65129 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_4
  simp only [← c65124, ← c65125, ← c65126] at e_1052_4
  have e_1052_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_5
  simp only [← c65127, ← c65128, k2, ← c65129] at e_1052_5
  have hr := e_1052_5
  simp only [e_1052_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8848)) := by
  have c65130 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65131 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65132 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65133 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65134 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65135 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_6
  simp only [← c65130, ← c65131, ← c65132] at e_1052_6
  have e_1052_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_7
  simp only [← c65133, ← c65134, k2, ← c65135] at e_1052_7
  have hr := e_1052_7
  simp only [e_1052_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8849)) := by
  have c65136 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65137 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65138 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65139 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65140 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65141 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_8
  simp only [← c65136, ← c65137, ← c65138] at e_1052_8
  have e_1052_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_9
  simp only [← c65139, ← c65140, k2, ← c65141] at e_1052_9
  have hr := e_1052_9
  simp only [e_1052_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8850)) := by
  have c65142 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65143 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65144 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65145 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65146 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65147 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_10
  simp only [← c65142, ← c65143, ← c65144] at e_1052_10
  have e_1052_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_11
  simp only [← c65145, ← c65146, k2, ← c65147] at e_1052_11
  have hr := e_1052_11
  simp only [e_1052_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8851)) := by
  have c65148 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65149 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65150 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65151 := (publicBatchWrapper89_copies2035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65152 := (publicBatchWrapper89_copies2036 a h).1
  have c65153 := (publicBatchWrapper89_copies2036 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_12
  simp only [← c65148, ← c65149, ← c65150] at e_1052_12
  have e_1052_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_13
  simp only [← c65151, ← c65152, k2, ← c65153] at e_1052_13
  have hr := e_1052_13
  simp only [e_1052_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8852)) := by
  have c65154 := (publicBatchWrapper89_copies2036 a h).2.2.1
  have c65155 := (publicBatchWrapper89_copies2036 a h).2.2.2.1
  have c65156 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.1
  have c65157 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.1
  have c65158 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.1
  have c65159 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_14
  simp only [← c65154, ← c65155, ← c65156] at e_1052_14
  have e_1052_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_15
  simp only [← c65157, ← c65158, k2, ← c65159] at e_1052_15
  have hr := e_1052_15
  simp only [e_1052_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8853)) := by
  have c65160 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.1
  have c65161 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.1
  have c65162 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65163 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65164 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65165 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_16
  simp only [← c65160, ← c65161, ← c65162] at e_1052_16
  have e_1052_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_17
  simp only [← c65163, ← c65164, k2, ← c65165] at e_1052_17
  have hr := e_1052_17
  simp only [e_1052_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1052 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8854)) := by
  have c65166 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65167 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65168 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65169 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65170 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65171 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1052_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_18
  simp only [← c65166, ← c65167, ← c65168] at e_1052_18
  have e_1052_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1052 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1052_19
  simp only [← c65169, ← c65170, k2, ← c65171] at e_1052_19
  have hr := e_1052_19
  simp only [e_1052_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8855)) := by
  have c65172 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65173 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65174 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65175 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65176 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65177 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_0
  simp only [← c65172, ← c65173, ← c65174] at e_1053_0
  have e_1053_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_1
  simp only [← c65175, ← c65176, k2, ← c65177] at e_1053_1
  have hr := e_1053_1
  simp only [e_1053_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8856)) := by
  have c65178 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65179 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65180 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65181 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65182 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65183 := (publicBatchWrapper89_copies2036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_2
  simp only [← c65178, ← c65179, ← c65180] at e_1053_2
  have e_1053_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_3
  simp only [← c65181, ← c65182, k2, ← c65183] at e_1053_3
  have hr := e_1053_3
  simp only [e_1053_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8857)) := by
  have c65184 := (publicBatchWrapper89_copies2037 a h).1
  have c65185 := (publicBatchWrapper89_copies2037 a h).2.1
  have c65186 := (publicBatchWrapper89_copies2037 a h).2.2.1
  have c65187 := (publicBatchWrapper89_copies2037 a h).2.2.2.1
  have c65188 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.1
  have c65189 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_4
  simp only [← c65184, ← c65185, ← c65186] at e_1053_4
  have e_1053_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_5
  simp only [← c65187, ← c65188, k2, ← c65189] at e_1053_5
  have hr := e_1053_5
  simp only [e_1053_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8858)) := by
  have c65190 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.1
  have c65191 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.1
  have c65192 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.1
  have c65193 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.1
  have c65194 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65195 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_6
  simp only [← c65190, ← c65191, ← c65192] at e_1053_6
  have e_1053_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_7
  simp only [← c65193, ← c65194, k2, ← c65195] at e_1053_7
  have hr := e_1053_7
  simp only [e_1053_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8859)) := by
  have c65196 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65197 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65198 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65199 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65200 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65201 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_8
  simp only [← c65196, ← c65197, ← c65198] at e_1053_8
  have e_1053_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_9
  simp only [← c65199, ← c65200, k2, ← c65201] at e_1053_9
  have hr := e_1053_9
  simp only [e_1053_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8860)) := by
  have c65202 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65203 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65204 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65205 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65206 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65207 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_10
  simp only [← c65202, ← c65203, ← c65204] at e_1053_10
  have e_1053_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_11
  simp only [← c65205, ← c65206, k2, ← c65207] at e_1053_11
  have hr := e_1053_11
  simp only [e_1053_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8861)) := by
  have c65208 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65209 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65210 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65211 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65212 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65213 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_12
  simp only [← c65208, ← c65209, ← c65210] at e_1053_12
  have e_1053_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_13
  simp only [← c65211, ← c65212, k2, ← c65213] at e_1053_13
  have hr := e_1053_13
  simp only [e_1053_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8862)) := by
  have c65214 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65215 := (publicBatchWrapper89_copies2037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65216 := (publicBatchWrapper89_copies2038 a h).1
  have c65217 := (publicBatchWrapper89_copies2038 a h).2.1
  have c65218 := (publicBatchWrapper89_copies2038 a h).2.2.1
  have c65219 := (publicBatchWrapper89_copies2038 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_14
  simp only [← c65214, ← c65215, ← c65216] at e_1053_14
  have e_1053_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_15
  simp only [← c65217, ← c65218, k2, ← c65219] at e_1053_15
  have hr := e_1053_15
  simp only [e_1053_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8863)) := by
  have c65220 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.1
  have c65221 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.1
  have c65222 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.1
  have c65223 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.1
  have c65224 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.1
  have c65225 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_16
  simp only [← c65220, ← c65221, ← c65222] at e_1053_16
  have e_1053_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_17
  simp only [← c65223, ← c65224, k2, ← c65225] at e_1053_17
  have hr := e_1053_17
  simp only [e_1053_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1053 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8864)) := by
  have c65226 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65227 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65228 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65229 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65230 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65231 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1053_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_18
  simp only [← c65226, ← c65227, ← c65228] at e_1053_18
  have e_1053_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1053 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1053_19
  simp only [← c65229, ← c65230, k2, ← c65231] at e_1053_19
  have hr := e_1053_19
  simp only [e_1053_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8865)) := by
  have c65232 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65233 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65234 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65235 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65236 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65237 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_0
  simp only [← c65232, ← c65233, ← c65234] at e_1054_0
  have e_1054_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_1
  simp only [← c65235, ← c65236, k2, ← c65237] at e_1054_1
  have hr := e_1054_1
  simp only [e_1054_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8866)) := by
  have c65238 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65239 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65240 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65241 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65242 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65243 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_2
  simp only [← c65238, ← c65239, ← c65240] at e_1054_2
  have e_1054_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_3
  simp only [← c65241, ← c65242, k2, ← c65243] at e_1054_3
  have hr := e_1054_3
  simp only [e_1054_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8867)) := by
  have c65244 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65245 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65246 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65247 := (publicBatchWrapper89_copies2038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65248 := (publicBatchWrapper89_copies2039 a h).1
  have c65249 := (publicBatchWrapper89_copies2039 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_4
  simp only [← c65244, ← c65245, ← c65246] at e_1054_4
  have e_1054_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_5
  simp only [← c65247, ← c65248, k2, ← c65249] at e_1054_5
  have hr := e_1054_5
  simp only [e_1054_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8868)) := by
  have c65250 := (publicBatchWrapper89_copies2039 a h).2.2.1
  have c65251 := (publicBatchWrapper89_copies2039 a h).2.2.2.1
  have c65252 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.1
  have c65253 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.1
  have c65254 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.1
  have c65255 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_6
  simp only [← c65250, ← c65251, ← c65252] at e_1054_6
  have e_1054_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_7
  simp only [← c65253, ← c65254, k2, ← c65255] at e_1054_7
  have hr := e_1054_7
  simp only [e_1054_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8869)) := by
  have c65256 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.1
  have c65257 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.1
  have c65258 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65259 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65260 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65261 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_8
  simp only [← c65256, ← c65257, ← c65258] at e_1054_8
  have e_1054_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_9
  simp only [← c65259, ← c65260, k2, ← c65261] at e_1054_9
  have hr := e_1054_9
  simp only [e_1054_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8870)) := by
  have c65262 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65263 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65264 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65265 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65266 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65267 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_10
  simp only [← c65262, ← c65263, ← c65264] at e_1054_10
  have e_1054_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_11
  simp only [← c65265, ← c65266, k2, ← c65267] at e_1054_11
  have hr := e_1054_11
  simp only [e_1054_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8871)) := by
  have c65268 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65269 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65270 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65271 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65272 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65273 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_12
  simp only [← c65268, ← c65269, ← c65270] at e_1054_12
  have e_1054_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_13
  simp only [← c65271, ← c65272, k2, ← c65273] at e_1054_13
  have hr := e_1054_13
  simp only [e_1054_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8872)) := by
  have c65274 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65275 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65276 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65277 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65278 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65279 := (publicBatchWrapper89_copies2039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_14
  simp only [← c65274, ← c65275, ← c65276] at e_1054_14
  have e_1054_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_15
  simp only [← c65277, ← c65278, k2, ← c65279] at e_1054_15
  have hr := e_1054_15
  simp only [e_1054_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8873)) := by
  have c65280 := (publicBatchWrapper89_copies2040 a h).1
  have c65281 := (publicBatchWrapper89_copies2040 a h).2.1
  have c65282 := (publicBatchWrapper89_copies2040 a h).2.2.1
  have c65283 := (publicBatchWrapper89_copies2040 a h).2.2.2.1
  have c65284 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.1
  have c65285 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_16
  simp only [← c65280, ← c65281, ← c65282] at e_1054_16
  have e_1054_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_17
  simp only [← c65283, ← c65284, k2, ← c65285] at e_1054_17
  have hr := e_1054_17
  simp only [e_1054_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1054 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8874)) := by
  have c65286 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.1
  have c65287 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.1
  have c65288 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.1
  have c65289 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.1
  have c65290 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65291 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1054_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_18
  simp only [← c65286, ← c65287, ← c65288] at e_1054_18
  have e_1054_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1054 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1054_19
  simp only [← c65289, ← c65290, k2, ← c65291] at e_1054_19
  have hr := e_1054_19
  simp only [e_1054_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8875)) := by
  have c65292 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65293 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65294 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65295 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65296 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65297 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_0
  simp only [← c65292, ← c65293, ← c65294] at e_1055_0
  have e_1055_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_1
  simp only [← c65295, ← c65296, k2, ← c65297] at e_1055_1
  have hr := e_1055_1
  simp only [e_1055_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8876)) := by
  have c65298 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65299 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65300 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65301 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65302 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65303 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_2
  simp only [← c65298, ← c65299, ← c65300] at e_1055_2
  have e_1055_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_3
  simp only [← c65301, ← c65302, k2, ← c65303] at e_1055_3
  have hr := e_1055_3
  simp only [e_1055_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8877)) := by
  have c65304 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65305 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65306 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65307 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65308 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65309 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_4
  simp only [← c65304, ← c65305, ← c65306] at e_1055_4
  have e_1055_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_5
  simp only [← c65307, ← c65308, k2, ← c65309] at e_1055_5
  have hr := e_1055_5
  simp only [e_1055_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8878)) := by
  have c65310 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65311 := (publicBatchWrapper89_copies2040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65312 := (publicBatchWrapper89_copies2041 a h).1
  have c65313 := (publicBatchWrapper89_copies2041 a h).2.1
  have c65314 := (publicBatchWrapper89_copies2041 a h).2.2.1
  have c65315 := (publicBatchWrapper89_copies2041 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_6
  simp only [← c65310, ← c65311, ← c65312] at e_1055_6
  have e_1055_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_7
  simp only [← c65313, ← c65314, k2, ← c65315] at e_1055_7
  have hr := e_1055_7
  simp only [e_1055_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8879)) := by
  have c65316 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.1
  have c65317 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.1
  have c65318 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.1
  have c65319 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.1
  have c65320 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.1
  have c65321 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_8
  simp only [← c65316, ← c65317, ← c65318] at e_1055_8
  have e_1055_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_9
  simp only [← c65319, ← c65320, k2, ← c65321] at e_1055_9
  have hr := e_1055_9
  simp only [e_1055_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8880)) := by
  have c65322 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65323 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65324 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65325 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65326 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65327 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_10
  simp only [← c65322, ← c65323, ← c65324] at e_1055_10
  have e_1055_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_11
  simp only [← c65325, ← c65326, k2, ← c65327] at e_1055_11
  have hr := e_1055_11
  simp only [e_1055_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8881)) := by
  have c65328 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65329 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65330 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65331 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65332 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65333 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_12
  simp only [← c65328, ← c65329, ← c65330] at e_1055_12
  have e_1055_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_13
  simp only [← c65331, ← c65332, k2, ← c65333] at e_1055_13
  have hr := e_1055_13
  simp only [e_1055_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8882)) := by
  have c65334 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65335 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65336 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65337 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65338 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65339 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_14
  simp only [← c65334, ← c65335, ← c65336] at e_1055_14
  have e_1055_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_15
  simp only [← c65337, ← c65338, k2, ← c65339] at e_1055_15
  have hr := e_1055_15
  simp only [e_1055_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8883)) := by
  have c65340 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65341 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65342 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65343 := (publicBatchWrapper89_copies2041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65344 := (publicBatchWrapper89_copies2042 a h).1
  have c65345 := (publicBatchWrapper89_copies2042 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_16
  simp only [← c65340, ← c65341, ← c65342] at e_1055_16
  have e_1055_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_17
  simp only [← c65343, ← c65344, k2, ← c65345] at e_1055_17
  have hr := e_1055_17
  simp only [e_1055_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1055 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8884)) := by
  have c65346 := (publicBatchWrapper89_copies2042 a h).2.2.1
  have c65347 := (publicBatchWrapper89_copies2042 a h).2.2.2.1
  have c65348 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.1
  have c65349 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.1
  have c65350 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.1
  have c65351 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1055_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_18
  simp only [← c65346, ← c65347, ← c65348] at e_1055_18
  have e_1055_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1055 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1055_19
  simp only [← c65349, ← c65350, k2, ← c65351] at e_1055_19
  have hr := e_1055_19
  simp only [e_1055_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8885)) := by
  have c65352 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.1
  have c65353 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.1
  have c65354 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65355 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65356 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65357 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_0
  simp only [← c65352, ← c65353, ← c65354] at e_1056_0
  have e_1056_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_1
  simp only [← c65355, ← c65356, k2, ← c65357] at e_1056_1
  have hr := e_1056_1
  simp only [e_1056_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8886)) := by
  have c65358 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65359 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65360 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65361 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65362 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65363 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_2
  simp only [← c65358, ← c65359, ← c65360] at e_1056_2
  have e_1056_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_3
  simp only [← c65361, ← c65362, k2, ← c65363] at e_1056_3
  have hr := e_1056_3
  simp only [e_1056_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8887)) := by
  have c65364 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65365 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65366 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65367 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65368 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65369 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_4
  simp only [← c65364, ← c65365, ← c65366] at e_1056_4
  have e_1056_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_5
  simp only [← c65367, ← c65368, k2, ← c65369] at e_1056_5
  have hr := e_1056_5
  simp only [e_1056_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8888)) := by
  have c65370 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65371 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65372 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65373 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65374 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65375 := (publicBatchWrapper89_copies2042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_6
  simp only [← c65370, ← c65371, ← c65372] at e_1056_6
  have e_1056_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_7
  simp only [← c65373, ← c65374, k2, ← c65375] at e_1056_7
  have hr := e_1056_7
  simp only [e_1056_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8889)) := by
  have c65376 := (publicBatchWrapper89_copies2043 a h).1
  have c65377 := (publicBatchWrapper89_copies2043 a h).2.1
  have c65378 := (publicBatchWrapper89_copies2043 a h).2.2.1
  have c65379 := (publicBatchWrapper89_copies2043 a h).2.2.2.1
  have c65380 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.1
  have c65381 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_8
  simp only [← c65376, ← c65377, ← c65378] at e_1056_8
  have e_1056_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_9
  simp only [← c65379, ← c65380, k2, ← c65381] at e_1056_9
  have hr := e_1056_9
  simp only [e_1056_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8890)) := by
  have c65382 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.1
  have c65383 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.1
  have c65384 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.1
  have c65385 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.1
  have c65386 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65387 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_10
  simp only [← c65382, ← c65383, ← c65384] at e_1056_10
  have e_1056_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_11
  simp only [← c65385, ← c65386, k2, ← c65387] at e_1056_11
  have hr := e_1056_11
  simp only [e_1056_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8891)) := by
  have c65388 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65389 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65390 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65391 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65392 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65393 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_12
  simp only [← c65388, ← c65389, ← c65390] at e_1056_12
  have e_1056_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_13
  simp only [← c65391, ← c65392, k2, ← c65393] at e_1056_13
  have hr := e_1056_13
  simp only [e_1056_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8892)) := by
  have c65394 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65395 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65396 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65397 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65398 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65399 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_14
  simp only [← c65394, ← c65395, ← c65396] at e_1056_14
  have e_1056_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_15
  simp only [← c65397, ← c65398, k2, ← c65399] at e_1056_15
  have hr := e_1056_15
  simp only [e_1056_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8893)) := by
  have c65400 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65401 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65402 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65403 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65404 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65405 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_16
  simp only [← c65400, ← c65401, ← c65402] at e_1056_16
  have e_1056_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_17
  simp only [← c65403, ← c65404, k2, ← c65405] at e_1056_17
  have hr := e_1056_17
  simp only [e_1056_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1056 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8894)) := by
  have c65406 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65407 := (publicBatchWrapper89_copies2043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65408 := (publicBatchWrapper89_copies2044 a h).1
  have c65409 := (publicBatchWrapper89_copies2044 a h).2.1
  have c65410 := (publicBatchWrapper89_copies2044 a h).2.2.1
  have c65411 := (publicBatchWrapper89_copies2044 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1056_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_18
  simp only [← c65406, ← c65407, ← c65408] at e_1056_18
  have e_1056_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1056 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1056_19
  simp only [← c65409, ← c65410, k2, ← c65411] at e_1056_19
  have hr := e_1056_19
  simp only [e_1056_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8895)) := by
  have c65412 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.1
  have c65413 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.1
  have c65414 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.1
  have c65415 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.1
  have c65416 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.1
  have c65417 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_0
  simp only [← c65412, ← c65413, ← c65414] at e_1057_0
  have e_1057_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_1
  simp only [← c65415, ← c65416, k2, ← c65417] at e_1057_1
  have hr := e_1057_1
  simp only [e_1057_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8896)) := by
  have c65418 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65419 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65420 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65421 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65422 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65423 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_2
  simp only [← c65418, ← c65419, ← c65420] at e_1057_2
  have e_1057_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_3
  simp only [← c65421, ← c65422, k2, ← c65423] at e_1057_3
  have hr := e_1057_3
  simp only [e_1057_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8897)) := by
  have c65424 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65425 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65426 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65427 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65428 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65429 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_4
  simp only [← c65424, ← c65425, ← c65426] at e_1057_4
  have e_1057_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_5
  simp only [← c65427, ← c65428, k2, ← c65429] at e_1057_5
  have hr := e_1057_5
  simp only [e_1057_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8898)) := by
  have c65430 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65431 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65432 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65433 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65434 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65435 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_6
  simp only [← c65430, ← c65431, ← c65432] at e_1057_6
  have e_1057_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_7
  simp only [← c65433, ← c65434, k2, ← c65435] at e_1057_7
  have hr := e_1057_7
  simp only [e_1057_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8899)) := by
  have c65436 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65437 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65438 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65439 := (publicBatchWrapper89_copies2044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65440 := (publicBatchWrapper89_copies2045 a h).1
  have c65441 := (publicBatchWrapper89_copies2045 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_8
  simp only [← c65436, ← c65437, ← c65438] at e_1057_8
  have e_1057_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_9
  simp only [← c65439, ← c65440, k2, ← c65441] at e_1057_9
  have hr := e_1057_9
  simp only [e_1057_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8900)) := by
  have c65442 := (publicBatchWrapper89_copies2045 a h).2.2.1
  have c65443 := (publicBatchWrapper89_copies2045 a h).2.2.2.1
  have c65444 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.1
  have c65445 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.1
  have c65446 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.1
  have c65447 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_10
  simp only [← c65442, ← c65443, ← c65444] at e_1057_10
  have e_1057_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_11
  simp only [← c65445, ← c65446, k2, ← c65447] at e_1057_11
  have hr := e_1057_11
  simp only [e_1057_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8901)) := by
  have c65448 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.1
  have c65449 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.1
  have c65450 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65451 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65452 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65453 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_12
  simp only [← c65448, ← c65449, ← c65450] at e_1057_12
  have e_1057_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_13
  simp only [← c65451, ← c65452, k2, ← c65453] at e_1057_13
  have hr := e_1057_13
  simp only [e_1057_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8902)) := by
  have c65454 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65455 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65456 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65457 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65458 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65459 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_14
  simp only [← c65454, ← c65455, ← c65456] at e_1057_14
  have e_1057_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_15
  simp only [← c65457, ← c65458, k2, ← c65459] at e_1057_15
  have hr := e_1057_15
  simp only [e_1057_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8903)) := by
  have c65460 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65461 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65462 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65463 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65464 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65465 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_16
  simp only [← c65460, ← c65461, ← c65462] at e_1057_16
  have e_1057_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_17
  simp only [← c65463, ← c65464, k2, ← c65465] at e_1057_17
  have hr := e_1057_17
  simp only [e_1057_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1057 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8904)) := by
  have c65466 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65467 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65468 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65469 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65470 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65471 := (publicBatchWrapper89_copies2045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1057_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_18
  simp only [← c65466, ← c65467, ← c65468] at e_1057_18
  have e_1057_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1057 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1057_19
  simp only [← c65469, ← c65470, k2, ← c65471] at e_1057_19
  have hr := e_1057_19
  simp only [e_1057_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8905)) := by
  have c65472 := (publicBatchWrapper89_copies2046 a h).1
  have c65473 := (publicBatchWrapper89_copies2046 a h).2.1
  have c65474 := (publicBatchWrapper89_copies2046 a h).2.2.1
  have c65475 := (publicBatchWrapper89_copies2046 a h).2.2.2.1
  have c65476 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.1
  have c65477 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_0
  simp only [← c65472, ← c65473, ← c65474] at e_1058_0
  have e_1058_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_1
  simp only [← c65475, ← c65476, k2, ← c65477] at e_1058_1
  have hr := e_1058_1
  simp only [e_1058_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8906)) := by
  have c65478 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.1
  have c65479 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.1
  have c65480 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.1
  have c65481 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.1
  have c65482 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65483 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_2
  simp only [← c65478, ← c65479, ← c65480] at e_1058_2
  have e_1058_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_3
  simp only [← c65481, ← c65482, k2, ← c65483] at e_1058_3
  have hr := e_1058_3
  simp only [e_1058_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8907)) := by
  have c65484 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65485 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65486 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65487 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65488 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65489 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_4
  simp only [← c65484, ← c65485, ← c65486] at e_1058_4
  have e_1058_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_5
  simp only [← c65487, ← c65488, k2, ← c65489] at e_1058_5
  have hr := e_1058_5
  simp only [e_1058_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8908)) := by
  have c65490 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65491 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65492 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65493 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65494 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65495 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_6
  simp only [← c65490, ← c65491, ← c65492] at e_1058_6
  have e_1058_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_7
  simp only [← c65493, ← c65494, k2, ← c65495] at e_1058_7
  have hr := e_1058_7
  simp only [e_1058_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8909)) := by
  have c65496 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65497 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65498 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65499 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65500 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65501 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_8
  simp only [← c65496, ← c65497, ← c65498] at e_1058_8
  have e_1058_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_9
  simp only [← c65499, ← c65500, k2, ← c65501] at e_1058_9
  have hr := e_1058_9
  simp only [e_1058_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8910)) := by
  have c65502 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65503 := (publicBatchWrapper89_copies2046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65504 := (publicBatchWrapper89_copies2047 a h).1
  have c65505 := (publicBatchWrapper89_copies2047 a h).2.1
  have c65506 := (publicBatchWrapper89_copies2047 a h).2.2.1
  have c65507 := (publicBatchWrapper89_copies2047 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_10
  simp only [← c65502, ← c65503, ← c65504] at e_1058_10
  have e_1058_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_11
  simp only [← c65505, ← c65506, k2, ← c65507] at e_1058_11
  have hr := e_1058_11
  simp only [e_1058_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8911)) := by
  have c65508 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.1
  have c65509 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.1
  have c65510 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.1
  have c65511 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.1
  have c65512 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.1
  have c65513 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_12
  simp only [← c65508, ← c65509, ← c65510] at e_1058_12
  have e_1058_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_13
  simp only [← c65511, ← c65512, k2, ← c65513] at e_1058_13
  have hr := e_1058_13
  simp only [e_1058_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8912)) := by
  have c65514 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65515 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65516 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65517 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65518 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65519 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_14
  simp only [← c65514, ← c65515, ← c65516] at e_1058_14
  have e_1058_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_15
  simp only [← c65517, ← c65518, k2, ← c65519] at e_1058_15
  have hr := e_1058_15
  simp only [e_1058_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8913)) := by
  have c65520 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65521 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65522 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65523 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65524 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65525 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_16
  simp only [← c65520, ← c65521, ← c65522] at e_1058_16
  have e_1058_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_17
  simp only [← c65523, ← c65524, k2, ← c65525] at e_1058_17
  have hr := e_1058_17
  simp only [e_1058_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1058 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8914)) := by
  have c65526 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65527 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65528 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65529 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65530 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65531 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1058_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_18
  simp only [← c65526, ← c65527, ← c65528] at e_1058_18
  have e_1058_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1058 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1058_19
  simp only [← c65529, ← c65530, k2, ← c65531] at e_1058_19
  have hr := e_1058_19
  simp only [e_1058_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8915)) := by
  have c65532 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65533 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65534 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65535 := (publicBatchWrapper89_copies2047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65536 := (publicBatchWrapper89_copies2048 a h).1
  have c65537 := (publicBatchWrapper89_copies2048 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_0
  simp only [← c65532, ← c65533, ← c65534] at e_1059_0
  have e_1059_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_1
  simp only [← c65535, ← c65536, k2, ← c65537] at e_1059_1
  have hr := e_1059_1
  simp only [e_1059_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8916)) := by
  have c65538 := (publicBatchWrapper89_copies2048 a h).2.2.1
  have c65539 := (publicBatchWrapper89_copies2048 a h).2.2.2.1
  have c65540 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.1
  have c65541 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.1
  have c65542 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.1
  have c65543 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_2
  simp only [← c65538, ← c65539, ← c65540] at e_1059_2
  have e_1059_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_3
  simp only [← c65541, ← c65542, k2, ← c65543] at e_1059_3
  have hr := e_1059_3
  simp only [e_1059_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8917)) := by
  have c65544 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.1
  have c65545 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.1
  have c65546 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65547 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65548 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65549 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_4
  simp only [← c65544, ← c65545, ← c65546] at e_1059_4
  have e_1059_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_5
  simp only [← c65547, ← c65548, k2, ← c65549] at e_1059_5
  have hr := e_1059_5
  simp only [e_1059_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8918)) := by
  have c65550 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65551 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65552 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65553 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65554 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65555 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_6
  simp only [← c65550, ← c65551, ← c65552] at e_1059_6
  have e_1059_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_7
  simp only [← c65553, ← c65554, k2, ← c65555] at e_1059_7
  have hr := e_1059_7
  simp only [e_1059_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8919)) := by
  have c65556 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65557 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65558 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65559 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65560 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65561 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_8
  simp only [← c65556, ← c65557, ← c65558] at e_1059_8
  have e_1059_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_9
  simp only [← c65559, ← c65560, k2, ← c65561] at e_1059_9
  have hr := e_1059_9
  simp only [e_1059_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8920)) := by
  have c65562 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65563 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65564 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65565 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65566 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65567 := (publicBatchWrapper89_copies2048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_10
  simp only [← c65562, ← c65563, ← c65564] at e_1059_10
  have e_1059_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_11
  simp only [← c65565, ← c65566, k2, ← c65567] at e_1059_11
  have hr := e_1059_11
  simp only [e_1059_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8921)) := by
  have c65568 := (publicBatchWrapper89_copies2049 a h).1
  have c65569 := (publicBatchWrapper89_copies2049 a h).2.1
  have c65570 := (publicBatchWrapper89_copies2049 a h).2.2.1
  have c65571 := (publicBatchWrapper89_copies2049 a h).2.2.2.1
  have c65572 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.1
  have c65573 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_12
  simp only [← c65568, ← c65569, ← c65570] at e_1059_12
  have e_1059_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_13
  simp only [← c65571, ← c65572, k2, ← c65573] at e_1059_13
  have hr := e_1059_13
  simp only [e_1059_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8922)) := by
  have c65574 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.1
  have c65575 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.1
  have c65576 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.1
  have c65577 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.1
  have c65578 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65579 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_14
  simp only [← c65574, ← c65575, ← c65576] at e_1059_14
  have e_1059_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_15
  simp only [← c65577, ← c65578, k2, ← c65579] at e_1059_15
  have hr := e_1059_15
  simp only [e_1059_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8923)) := by
  have c65580 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65581 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65582 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65583 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65584 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65585 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_16
  simp only [← c65580, ← c65581, ← c65582] at e_1059_16
  have e_1059_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_17
  simp only [← c65583, ← c65584, k2, ← c65585] at e_1059_17
  have hr := e_1059_17
  simp only [e_1059_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1059 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8924)) := by
  have c65586 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65587 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65588 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65589 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65590 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65591 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1059_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_18
  simp only [← c65586, ← c65587, ← c65588] at e_1059_18
  have e_1059_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1059 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1059_19
  simp only [← c65589, ← c65590, k2, ← c65591] at e_1059_19
  have hr := e_1059_19
  simp only [e_1059_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8925)) := by
  have c65592 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65593 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65594 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65595 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65596 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65597 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_0
  simp only [← c65592, ← c65593, ← c65594] at e_1060_0
  have e_1060_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_1
  simp only [← c65595, ← c65596, k2, ← c65597] at e_1060_1
  have hr := e_1060_1
  simp only [e_1060_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8926)) := by
  have c65598 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65599 := (publicBatchWrapper89_copies2049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65600 := (publicBatchWrapper89_copies2050 a h).1
  have c65601 := (publicBatchWrapper89_copies2050 a h).2.1
  have c65602 := (publicBatchWrapper89_copies2050 a h).2.2.1
  have c65603 := (publicBatchWrapper89_copies2050 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_2
  simp only [← c65598, ← c65599, ← c65600] at e_1060_2
  have e_1060_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_3
  simp only [← c65601, ← c65602, k2, ← c65603] at e_1060_3
  have hr := e_1060_3
  simp only [e_1060_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8927)) := by
  have c65604 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.1
  have c65605 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.1
  have c65606 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.1
  have c65607 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.1
  have c65608 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.1
  have c65609 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_4
  simp only [← c65604, ← c65605, ← c65606] at e_1060_4
  have e_1060_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_5
  simp only [← c65607, ← c65608, k2, ← c65609] at e_1060_5
  have hr := e_1060_5
  simp only [e_1060_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8928)) := by
  have c65610 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65611 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65612 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65613 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65614 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65615 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_6
  simp only [← c65610, ← c65611, ← c65612] at e_1060_6
  have e_1060_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_7
  simp only [← c65613, ← c65614, k2, ← c65615] at e_1060_7
  have hr := e_1060_7
  simp only [e_1060_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8929)) := by
  have c65616 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65617 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65618 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65619 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65620 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65621 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_8
  simp only [← c65616, ← c65617, ← c65618] at e_1060_8
  have e_1060_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_9
  simp only [← c65619, ← c65620, k2, ← c65621] at e_1060_9
  have hr := e_1060_9
  simp only [e_1060_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8930)) := by
  have c65622 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65623 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65624 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65625 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65626 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65627 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_10
  simp only [← c65622, ← c65623, ← c65624] at e_1060_10
  have e_1060_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_11
  simp only [← c65625, ← c65626, k2, ← c65627] at e_1060_11
  have hr := e_1060_11
  simp only [e_1060_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8931)) := by
  have c65628 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65629 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65630 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65631 := (publicBatchWrapper89_copies2050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65632 := (publicBatchWrapper89_copies2051 a h).1
  have c65633 := (publicBatchWrapper89_copies2051 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_12
  simp only [← c65628, ← c65629, ← c65630] at e_1060_12
  have e_1060_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_13
  simp only [← c65631, ← c65632, k2, ← c65633] at e_1060_13
  have hr := e_1060_13
  simp only [e_1060_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8932)) := by
  have c65634 := (publicBatchWrapper89_copies2051 a h).2.2.1
  have c65635 := (publicBatchWrapper89_copies2051 a h).2.2.2.1
  have c65636 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.1
  have c65637 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.1
  have c65638 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.1
  have c65639 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_14
  simp only [← c65634, ← c65635, ← c65636] at e_1060_14
  have e_1060_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_15
  simp only [← c65637, ← c65638, k2, ← c65639] at e_1060_15
  have hr := e_1060_15
  simp only [e_1060_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8933)) := by
  have c65640 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.1
  have c65641 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.1
  have c65642 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65643 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65644 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65645 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_16
  simp only [← c65640, ← c65641, ← c65642] at e_1060_16
  have e_1060_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_17
  simp only [← c65643, ← c65644, k2, ← c65645] at e_1060_17
  have hr := e_1060_17
  simp only [e_1060_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1060 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8934)) := by
  have c65646 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65647 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65648 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65649 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65650 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65651 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1060_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_18
  simp only [← c65646, ← c65647, ← c65648] at e_1060_18
  have e_1060_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1060 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1060_19
  simp only [← c65649, ← c65650, k2, ← c65651] at e_1060_19
  have hr := e_1060_19
  simp only [e_1060_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8935)) := by
  have c65652 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65653 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65654 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65655 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65656 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65657 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_0
  simp only [← c65652, ← c65653, ← c65654] at e_1061_0
  have e_1061_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_1
  simp only [← c65655, ← c65656, k2, ← c65657] at e_1061_1
  have hr := e_1061_1
  simp only [e_1061_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8936)) := by
  have c65658 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65659 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65660 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65661 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65662 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65663 := (publicBatchWrapper89_copies2051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_2
  simp only [← c65658, ← c65659, ← c65660] at e_1061_2
  have e_1061_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_3
  simp only [← c65661, ← c65662, k2, ← c65663] at e_1061_3
  have hr := e_1061_3
  simp only [e_1061_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8937)) := by
  have c65664 := (publicBatchWrapper89_copies2052 a h).1
  have c65665 := (publicBatchWrapper89_copies2052 a h).2.1
  have c65666 := (publicBatchWrapper89_copies2052 a h).2.2.1
  have c65667 := (publicBatchWrapper89_copies2052 a h).2.2.2.1
  have c65668 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.1
  have c65669 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_4
  simp only [← c65664, ← c65665, ← c65666] at e_1061_4
  have e_1061_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_5
  simp only [← c65667, ← c65668, k2, ← c65669] at e_1061_5
  have hr := e_1061_5
  simp only [e_1061_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8938)) := by
  have c65670 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.1
  have c65671 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.1
  have c65672 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.1
  have c65673 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.1
  have c65674 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65675 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_6
  simp only [← c65670, ← c65671, ← c65672] at e_1061_6
  have e_1061_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_7
  simp only [← c65673, ← c65674, k2, ← c65675] at e_1061_7
  have hr := e_1061_7
  simp only [e_1061_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8939)) := by
  have c65676 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65677 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65678 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65679 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65680 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65681 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_8
  simp only [← c65676, ← c65677, ← c65678] at e_1061_8
  have e_1061_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_9
  simp only [← c65679, ← c65680, k2, ← c65681] at e_1061_9
  have hr := e_1061_9
  simp only [e_1061_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8940)) := by
  have c65682 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65683 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65684 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65685 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65686 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65687 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_10
  simp only [← c65682, ← c65683, ← c65684] at e_1061_10
  have e_1061_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_11
  simp only [← c65685, ← c65686, k2, ← c65687] at e_1061_11
  have hr := e_1061_11
  simp only [e_1061_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8941)) := by
  have c65688 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65689 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65690 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65691 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65692 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65693 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_12
  simp only [← c65688, ← c65689, ← c65690] at e_1061_12
  have e_1061_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_13
  simp only [← c65691, ← c65692, k2, ← c65693] at e_1061_13
  have hr := e_1061_13
  simp only [e_1061_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8942)) := by
  have c65694 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65695 := (publicBatchWrapper89_copies2052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65696 := (publicBatchWrapper89_copies2053 a h).1
  have c65697 := (publicBatchWrapper89_copies2053 a h).2.1
  have c65698 := (publicBatchWrapper89_copies2053 a h).2.2.1
  have c65699 := (publicBatchWrapper89_copies2053 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_14
  simp only [← c65694, ← c65695, ← c65696] at e_1061_14
  have e_1061_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_15
  simp only [← c65697, ← c65698, k2, ← c65699] at e_1061_15
  have hr := e_1061_15
  simp only [e_1061_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8943)) := by
  have c65700 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.1
  have c65701 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.1
  have c65702 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.1
  have c65703 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.1
  have c65704 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.1
  have c65705 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_16
  simp only [← c65700, ← c65701, ← c65702] at e_1061_16
  have e_1061_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_17
  simp only [← c65703, ← c65704, k2, ← c65705] at e_1061_17
  have hr := e_1061_17
  simp only [e_1061_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1061 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8944)) := by
  have c65706 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65707 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65708 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65709 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65710 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65711 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1061_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_18
  simp only [← c65706, ← c65707, ← c65708] at e_1061_18
  have e_1061_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1061 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1061_19
  simp only [← c65709, ← c65710, k2, ← c65711] at e_1061_19
  have hr := e_1061_19
  simp only [e_1061_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8945)) := by
  have c65712 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65713 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65714 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65715 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65716 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65717 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_0
  simp only [← c65712, ← c65713, ← c65714] at e_1062_0
  have e_1062_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_1
  simp only [← c65715, ← c65716, k2, ← c65717] at e_1062_1
  have hr := e_1062_1
  simp only [e_1062_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8946)) := by
  have c65718 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65719 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65720 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65721 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65722 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65723 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_2
  simp only [← c65718, ← c65719, ← c65720] at e_1062_2
  have e_1062_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_3
  simp only [← c65721, ← c65722, k2, ← c65723] at e_1062_3
  have hr := e_1062_3
  simp only [e_1062_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8947)) := by
  have c65724 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65725 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65726 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65727 := (publicBatchWrapper89_copies2053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65728 := (publicBatchWrapper89_copies2054 a h).1
  have c65729 := (publicBatchWrapper89_copies2054 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_4
  simp only [← c65724, ← c65725, ← c65726] at e_1062_4
  have e_1062_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_5
  simp only [← c65727, ← c65728, k2, ← c65729] at e_1062_5
  have hr := e_1062_5
  simp only [e_1062_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8948)) := by
  have c65730 := (publicBatchWrapper89_copies2054 a h).2.2.1
  have c65731 := (publicBatchWrapper89_copies2054 a h).2.2.2.1
  have c65732 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.1
  have c65733 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.1
  have c65734 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.1
  have c65735 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_6
  simp only [← c65730, ← c65731, ← c65732] at e_1062_6
  have e_1062_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_7
  simp only [← c65733, ← c65734, k2, ← c65735] at e_1062_7
  have hr := e_1062_7
  simp only [e_1062_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8949)) := by
  have c65736 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.1
  have c65737 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.1
  have c65738 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65739 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65740 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65741 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_8
  simp only [← c65736, ← c65737, ← c65738] at e_1062_8
  have e_1062_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_9
  simp only [← c65739, ← c65740, k2, ← c65741] at e_1062_9
  have hr := e_1062_9
  simp only [e_1062_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8950)) := by
  have c65742 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65743 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65744 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65745 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65746 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65747 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_10
  simp only [← c65742, ← c65743, ← c65744] at e_1062_10
  have e_1062_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_11
  simp only [← c65745, ← c65746, k2, ← c65747] at e_1062_11
  have hr := e_1062_11
  simp only [e_1062_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8951)) := by
  have c65748 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65749 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65750 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65751 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65752 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65753 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_12
  simp only [← c65748, ← c65749, ← c65750] at e_1062_12
  have e_1062_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_13
  simp only [← c65751, ← c65752, k2, ← c65753] at e_1062_13
  have hr := e_1062_13
  simp only [e_1062_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8952)) := by
  have c65754 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65755 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65756 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65757 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65758 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65759 := (publicBatchWrapper89_copies2054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_14
  simp only [← c65754, ← c65755, ← c65756] at e_1062_14
  have e_1062_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_15
  simp only [← c65757, ← c65758, k2, ← c65759] at e_1062_15
  have hr := e_1062_15
  simp only [e_1062_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8953)) := by
  have c65760 := (publicBatchWrapper89_copies2055 a h).1
  have c65761 := (publicBatchWrapper89_copies2055 a h).2.1
  have c65762 := (publicBatchWrapper89_copies2055 a h).2.2.1
  have c65763 := (publicBatchWrapper89_copies2055 a h).2.2.2.1
  have c65764 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.1
  have c65765 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_16
  simp only [← c65760, ← c65761, ← c65762] at e_1062_16
  have e_1062_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_17
  simp only [← c65763, ← c65764, k2, ← c65765] at e_1062_17
  have hr := e_1062_17
  simp only [e_1062_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1062 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8954)) := by
  have c65766 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.1
  have c65767 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.1
  have c65768 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.1
  have c65769 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.1
  have c65770 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65771 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1062_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_18
  simp only [← c65766, ← c65767, ← c65768] at e_1062_18
  have e_1062_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1062 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1062_19
  simp only [← c65769, ← c65770, k2, ← c65771] at e_1062_19
  have hr := e_1062_19
  simp only [e_1062_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8955)) := by
  have c65772 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65773 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65774 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65775 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65776 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65777 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_0
  simp only [← c65772, ← c65773, ← c65774] at e_1063_0
  have e_1063_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_1
  simp only [← c65775, ← c65776, k2, ← c65777] at e_1063_1
  have hr := e_1063_1
  simp only [e_1063_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8956)) := by
  have c65778 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65779 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65780 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65781 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65782 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65783 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_2
  simp only [← c65778, ← c65779, ← c65780] at e_1063_2
  have e_1063_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_3
  simp only [← c65781, ← c65782, k2, ← c65783] at e_1063_3
  have hr := e_1063_3
  simp only [e_1063_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8957)) := by
  have c65784 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65785 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65786 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65787 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65788 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65789 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_4
  simp only [← c65784, ← c65785, ← c65786] at e_1063_4
  have e_1063_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_5
  simp only [← c65787, ← c65788, k2, ← c65789] at e_1063_5
  have hr := e_1063_5
  simp only [e_1063_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8958)) := by
  have c65790 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65791 := (publicBatchWrapper89_copies2055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65792 := (publicBatchWrapper89_copies2056 a h).1
  have c65793 := (publicBatchWrapper89_copies2056 a h).2.1
  have c65794 := (publicBatchWrapper89_copies2056 a h).2.2.1
  have c65795 := (publicBatchWrapper89_copies2056 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_6
  simp only [← c65790, ← c65791, ← c65792] at e_1063_6
  have e_1063_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_7
  simp only [← c65793, ← c65794, k2, ← c65795] at e_1063_7
  have hr := e_1063_7
  simp only [e_1063_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8959)) := by
  have c65796 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.1
  have c65797 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.1
  have c65798 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.1
  have c65799 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.1
  have c65800 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.1
  have c65801 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_8
  simp only [← c65796, ← c65797, ← c65798] at e_1063_8
  have e_1063_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_9
  simp only [← c65799, ← c65800, k2, ← c65801] at e_1063_9
  have hr := e_1063_9
  simp only [e_1063_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8960)) := by
  have c65802 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65803 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65804 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65805 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65806 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65807 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_10
  simp only [← c65802, ← c65803, ← c65804] at e_1063_10
  have e_1063_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_11
  simp only [← c65805, ← c65806, k2, ← c65807] at e_1063_11
  have hr := e_1063_11
  simp only [e_1063_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8961)) := by
  have c65808 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65809 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65810 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65811 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65812 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65813 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_12
  simp only [← c65808, ← c65809, ← c65810] at e_1063_12
  have e_1063_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_13
  simp only [← c65811, ← c65812, k2, ← c65813] at e_1063_13
  have hr := e_1063_13
  simp only [e_1063_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8962)) := by
  have c65814 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65815 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65816 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65817 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65818 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65819 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_14
  simp only [← c65814, ← c65815, ← c65816] at e_1063_14
  have e_1063_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_15
  simp only [← c65817, ← c65818, k2, ← c65819] at e_1063_15
  have hr := e_1063_15
  simp only [e_1063_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8963)) := by
  have c65820 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65821 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65822 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65823 := (publicBatchWrapper89_copies2056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65824 := (publicBatchWrapper89_copies2057 a h).1
  have c65825 := (publicBatchWrapper89_copies2057 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_16
  simp only [← c65820, ← c65821, ← c65822] at e_1063_16
  have e_1063_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_17
  simp only [← c65823, ← c65824, k2, ← c65825] at e_1063_17
  have hr := e_1063_17
  simp only [e_1063_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1063 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8964)) := by
  have c65826 := (publicBatchWrapper89_copies2057 a h).2.2.1
  have c65827 := (publicBatchWrapper89_copies2057 a h).2.2.2.1
  have c65828 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.1
  have c65829 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.1
  have c65830 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.1
  have c65831 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1063_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_18
  simp only [← c65826, ← c65827, ← c65828] at e_1063_18
  have e_1063_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1063 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1063_19
  simp only [← c65829, ← c65830, k2, ← c65831] at e_1063_19
  have hr := e_1063_19
  simp only [e_1063_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8965)) := by
  have c65832 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.1
  have c65833 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.1
  have c65834 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65835 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65836 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65837 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_0
  simp only [← c65832, ← c65833, ← c65834] at e_1064_0
  have e_1064_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_1
  simp only [← c65835, ← c65836, k2, ← c65837] at e_1064_1
  have hr := e_1064_1
  simp only [e_1064_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8966)) := by
  have c65838 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65839 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65840 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65841 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65842 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65843 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_2
  simp only [← c65838, ← c65839, ← c65840] at e_1064_2
  have e_1064_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_3
  simp only [← c65841, ← c65842, k2, ← c65843] at e_1064_3
  have hr := e_1064_3
  simp only [e_1064_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8967)) := by
  have c65844 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65845 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65846 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65847 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65848 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65849 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_4
  simp only [← c65844, ← c65845, ← c65846] at e_1064_4
  have e_1064_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_5
  simp only [← c65847, ← c65848, k2, ← c65849] at e_1064_5
  have hr := e_1064_5
  simp only [e_1064_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8968)) := by
  have c65850 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65851 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65852 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65853 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65854 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65855 := (publicBatchWrapper89_copies2057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_6
  simp only [← c65850, ← c65851, ← c65852] at e_1064_6
  have e_1064_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_7
  simp only [← c65853, ← c65854, k2, ← c65855] at e_1064_7
  have hr := e_1064_7
  simp only [e_1064_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8969)) := by
  have c65856 := (publicBatchWrapper89_copies2058 a h).1
  have c65857 := (publicBatchWrapper89_copies2058 a h).2.1
  have c65858 := (publicBatchWrapper89_copies2058 a h).2.2.1
  have c65859 := (publicBatchWrapper89_copies2058 a h).2.2.2.1
  have c65860 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.1
  have c65861 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_8
  simp only [← c65856, ← c65857, ← c65858] at e_1064_8
  have e_1064_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_9
  simp only [← c65859, ← c65860, k2, ← c65861] at e_1064_9
  have hr := e_1064_9
  simp only [e_1064_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9010)) := by
  have c65862 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.1
  have c65863 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.1
  have c65864 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.1
  have c65865 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.1
  have c65866 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65867 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_10
  simp only [← c65862, ← c65863, ← c65864] at e_1064_10
  have e_1064_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_11
  simp only [← c65865, ← c65866, k2, ← c65867] at e_1064_11
  have hr := e_1064_11
  simp only [e_1064_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9011)) := by
  have c65868 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65869 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65870 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65871 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65872 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65873 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_12
  simp only [← c65868, ← c65869, ← c65870] at e_1064_12
  have e_1064_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_13
  simp only [← c65871, ← c65872, k2, ← c65873] at e_1064_13
  have hr := e_1064_13
  simp only [e_1064_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9012)) := by
  have c65874 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65875 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65876 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65877 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65878 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65879 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_14
  simp only [← c65874, ← c65875, ← c65876] at e_1064_14
  have e_1064_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_15
  simp only [← c65877, ← c65878, k2, ← c65879] at e_1064_15
  have hr := e_1064_15
  simp only [e_1064_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9013)) := by
  have c65880 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65881 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65882 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65883 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65884 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65885 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_16
  simp only [← c65880, ← c65881, ← c65882] at e_1064_16
  have e_1064_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_17
  simp only [← c65883, ← c65884, k2, ← c65885] at e_1064_17
  have hr := e_1064_17
  simp only [e_1064_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1064 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9014)) := by
  have c65886 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65887 := (publicBatchWrapper89_copies2058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65888 := (publicBatchWrapper89_copies2059 a h).1
  have c65889 := (publicBatchWrapper89_copies2059 a h).2.1
  have c65890 := (publicBatchWrapper89_copies2059 a h).2.2.1
  have c65891 := (publicBatchWrapper89_copies2059 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1064_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_18
  simp only [← c65886, ← c65887, ← c65888] at e_1064_18
  have e_1064_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1064 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1064_19
  simp only [← c65889, ← c65890, k2, ← c65891] at e_1064_19
  have hr := e_1064_19
  simp only [e_1064_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9015)) := by
  have c65892 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.1
  have c65893 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.1
  have c65894 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.1
  have c65895 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.1
  have c65896 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.1
  have c65897 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_0
  simp only [← c65892, ← c65893, ← c65894] at e_1065_0
  have e_1065_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_1
  simp only [← c65895, ← c65896, k2, ← c65897] at e_1065_1
  have hr := e_1065_1
  simp only [e_1065_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9016)) := by
  have c65898 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65899 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65900 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65901 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65902 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65903 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_2
  simp only [← c65898, ← c65899, ← c65900] at e_1065_2
  have e_1065_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_3
  simp only [← c65901, ← c65902, k2, ← c65903] at e_1065_3
  have hr := e_1065_3
  simp only [e_1065_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9017)) := by
  have c65904 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65905 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65906 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65907 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65908 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65909 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_4
  simp only [← c65904, ← c65905, ← c65906] at e_1065_4
  have e_1065_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_5
  simp only [← c65907, ← c65908, k2, ← c65909] at e_1065_5
  have hr := e_1065_5
  simp only [e_1065_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9018)) := by
  have c65910 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65911 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65912 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65913 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65914 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65915 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_6
  simp only [← c65910, ← c65911, ← c65912] at e_1065_6
  have e_1065_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_7
  simp only [← c65913, ← c65914, k2, ← c65915] at e_1065_7
  have hr := e_1065_7
  simp only [e_1065_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9019)) := by
  have c65916 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65917 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65918 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65919 := (publicBatchWrapper89_copies2059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65920 := (publicBatchWrapper89_copies2060 a h).1
  have c65921 := (publicBatchWrapper89_copies2060 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_8
  simp only [← c65916, ← c65917, ← c65918] at e_1065_8
  have e_1065_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_9
  simp only [← c65919, ← c65920, k2, ← c65921] at e_1065_9
  have hr := e_1065_9
  simp only [e_1065_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9020)) := by
  have c65922 := (publicBatchWrapper89_copies2060 a h).2.2.1
  have c65923 := (publicBatchWrapper89_copies2060 a h).2.2.2.1
  have c65924 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.1
  have c65925 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.1
  have c65926 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.1
  have c65927 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_10
  simp only [← c65922, ← c65923, ← c65924] at e_1065_10
  have e_1065_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_11
  simp only [← c65925, ← c65926, k2, ← c65927] at e_1065_11
  have hr := e_1065_11
  simp only [e_1065_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9021)) := by
  have c65928 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.1
  have c65929 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.1
  have c65930 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65931 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65932 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65933 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_12
  simp only [← c65928, ← c65929, ← c65930] at e_1065_12
  have e_1065_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_13
  simp only [← c65931, ← c65932, k2, ← c65933] at e_1065_13
  have hr := e_1065_13
  simp only [e_1065_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9022)) := by
  have c65934 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65935 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65936 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65937 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65938 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65939 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_14
  simp only [← c65934, ← c65935, ← c65936] at e_1065_14
  have e_1065_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_15
  simp only [← c65937, ← c65938, k2, ← c65939] at e_1065_15
  have hr := e_1065_15
  simp only [e_1065_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9023)) := by
  have c65940 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65941 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65942 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65943 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65944 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65945 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_16
  simp only [← c65940, ← c65941, ← c65942] at e_1065_16
  have e_1065_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_17
  simp only [← c65943, ← c65944, k2, ← c65945] at e_1065_17
  have hr := e_1065_17
  simp only [e_1065_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1065 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9024)) := by
  have c65946 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65947 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65948 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65949 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65950 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65951 := (publicBatchWrapper89_copies2060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1065_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_18
  simp only [← c65946, ← c65947, ← c65948] at e_1065_18
  have e_1065_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1065 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1065_19
  simp only [← c65949, ← c65950, k2, ← c65951] at e_1065_19
  have hr := e_1065_19
  simp only [e_1065_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9025)) := by
  have c65952 := (publicBatchWrapper89_copies2061 a h).1
  have c65953 := (publicBatchWrapper89_copies2061 a h).2.1
  have c65954 := (publicBatchWrapper89_copies2061 a h).2.2.1
  have c65955 := (publicBatchWrapper89_copies2061 a h).2.2.2.1
  have c65956 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.1
  have c65957 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_0
  simp only [← c65952, ← c65953, ← c65954] at e_1066_0
  have e_1066_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_1
  simp only [← c65955, ← c65956, k2, ← c65957] at e_1066_1
  have hr := e_1066_1
  simp only [e_1066_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9026)) := by
  have c65958 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.1
  have c65959 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.1
  have c65960 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.1
  have c65961 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.1
  have c65962 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65963 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_2
  simp only [← c65958, ← c65959, ← c65960] at e_1066_2
  have e_1066_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_3
  simp only [← c65961, ← c65962, k2, ← c65963] at e_1066_3
  have hr := e_1066_3
  simp only [e_1066_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9027)) := by
  have c65964 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65965 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65966 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65967 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65968 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65969 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_4
  simp only [← c65964, ← c65965, ← c65966] at e_1066_4
  have e_1066_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_5
  simp only [← c65967, ← c65968, k2, ← c65969] at e_1066_5
  have hr := e_1066_5
  simp only [e_1066_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9028)) := by
  have c65970 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65971 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65972 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65973 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65974 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65975 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_6
  simp only [← c65970, ← c65971, ← c65972] at e_1066_6
  have e_1066_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_7
  simp only [← c65973, ← c65974, k2, ← c65975] at e_1066_7
  have hr := e_1066_7
  simp only [e_1066_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9029)) := by
  have c65976 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65977 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65978 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65979 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65980 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65981 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_8
  simp only [← c65976, ← c65977, ← c65978] at e_1066_8
  have e_1066_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_9
  simp only [← c65979, ← c65980, k2, ← c65981] at e_1066_9
  have hr := e_1066_9
  simp only [e_1066_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9030)) := by
  have c65982 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65983 := (publicBatchWrapper89_copies2061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c65984 := (publicBatchWrapper89_copies2062 a h).1
  have c65985 := (publicBatchWrapper89_copies2062 a h).2.1
  have c65986 := (publicBatchWrapper89_copies2062 a h).2.2.1
  have c65987 := (publicBatchWrapper89_copies2062 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_10
  simp only [← c65982, ← c65983, ← c65984] at e_1066_10
  have e_1066_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_11
  simp only [← c65985, ← c65986, k2, ← c65987] at e_1066_11
  have hr := e_1066_11
  simp only [e_1066_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9031)) := by
  have c65988 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.1
  have c65989 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.1
  have c65990 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.1
  have c65991 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.1
  have c65992 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.1
  have c65993 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_12
  simp only [← c65988, ← c65989, ← c65990] at e_1066_12
  have e_1066_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_13
  simp only [← c65991, ← c65992, k2, ← c65993] at e_1066_13
  have hr := e_1066_13
  simp only [e_1066_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9032)) := by
  have c65994 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.1
  have c65995 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c65996 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65997 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65998 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c65999 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_14
  simp only [← c65994, ← c65995, ← c65996] at e_1066_14
  have e_1066_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_15
  simp only [← c65997, ← c65998, k2, ← c65999] at e_1066_15
  have hr := e_1066_15
  simp only [e_1066_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9033)) := by
  have c66000 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66001 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66002 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66003 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66004 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66005 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_16
  simp only [← c66000, ← c66001, ← c66002] at e_1066_16
  have e_1066_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_17
  simp only [← c66003, ← c66004, k2, ← c66005] at e_1066_17
  have hr := e_1066_17
  simp only [e_1066_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1066 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9034)) := by
  have c66006 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66007 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66008 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66009 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66010 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66011 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1066_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_18
  simp only [← c66006, ← c66007, ← c66008] at e_1066_18
  have e_1066_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1066 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1066_19
  simp only [← c66009, ← c66010, k2, ← c66011] at e_1066_19
  have hr := e_1066_19
  simp only [e_1066_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9035)) := by
  have c66012 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66013 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66014 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66015 := (publicBatchWrapper89_copies2062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66016 := (publicBatchWrapper89_copies2063 a h).1
  have c66017 := (publicBatchWrapper89_copies2063 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_0
  simp only [← c66012, ← c66013, ← c66014] at e_1067_0
  have e_1067_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_1
  simp only [← c66015, ← c66016, k2, ← c66017] at e_1067_1
  have hr := e_1067_1
  simp only [e_1067_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9036)) := by
  have c66018 := (publicBatchWrapper89_copies2063 a h).2.2.1
  have c66019 := (publicBatchWrapper89_copies2063 a h).2.2.2.1
  have c66020 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.1
  have c66021 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.1
  have c66022 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.1
  have c66023 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_2
  simp only [← c66018, ← c66019, ← c66020] at e_1067_2
  have e_1067_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_3
  simp only [← c66021, ← c66022, k2, ← c66023] at e_1067_3
  have hr := e_1067_3
  simp only [e_1067_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9037)) := by
  have c66024 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.1
  have c66025 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.1
  have c66026 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66027 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66028 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66029 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_4
  simp only [← c66024, ← c66025, ← c66026] at e_1067_4
  have e_1067_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_5
  simp only [← c66027, ← c66028, k2, ← c66029] at e_1067_5
  have hr := e_1067_5
  simp only [e_1067_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9038)) := by
  have c66030 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66031 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66032 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66033 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66034 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66035 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_6
  simp only [← c66030, ← c66031, ← c66032] at e_1067_6
  have e_1067_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_7
  simp only [← c66033, ← c66034, k2, ← c66035] at e_1067_7
  have hr := e_1067_7
  simp only [e_1067_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9039)) := by
  have c66036 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66037 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66038 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66039 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66040 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66041 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_8
  simp only [← c66036, ← c66037, ← c66038] at e_1067_8
  have e_1067_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_9
  simp only [← c66039, ← c66040, k2, ← c66041] at e_1067_9
  have hr := e_1067_9
  simp only [e_1067_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9040)) := by
  have c66042 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66043 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66044 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66045 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66046 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66047 := (publicBatchWrapper89_copies2063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_10
  simp only [← c66042, ← c66043, ← c66044] at e_1067_10
  have e_1067_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_11
  simp only [← c66045, ← c66046, k2, ← c66047] at e_1067_11
  have hr := e_1067_11
  simp only [e_1067_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9041)) := by
  have c66048 := (publicBatchWrapper89_copies2064 a h).1
  have c66049 := (publicBatchWrapper89_copies2064 a h).2.1
  have c66050 := (publicBatchWrapper89_copies2064 a h).2.2.1
  have c66051 := (publicBatchWrapper89_copies2064 a h).2.2.2.1
  have c66052 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.1
  have c66053 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_12
  simp only [← c66048, ← c66049, ← c66050] at e_1067_12
  have e_1067_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_13
  simp only [← c66051, ← c66052, k2, ← c66053] at e_1067_13
  have hr := e_1067_13
  simp only [e_1067_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9042)) := by
  have c66054 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.1
  have c66055 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.1
  have c66056 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.1
  have c66057 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.1
  have c66058 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66059 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_14
  simp only [← c66054, ← c66055, ← c66056] at e_1067_14
  have e_1067_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_15
  simp only [← c66057, ← c66058, k2, ← c66059] at e_1067_15
  have hr := e_1067_15
  simp only [e_1067_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9043)) := by
  have c66060 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66061 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66062 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66063 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66064 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66065 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_16
  simp only [← c66060, ← c66061, ← c66062] at e_1067_16
  have e_1067_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_17
  simp only [← c66063, ← c66064, k2, ← c66065] at e_1067_17
  have hr := e_1067_17
  simp only [e_1067_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1067 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9044)) := by
  have c66066 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66067 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66068 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66069 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66070 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66071 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1067_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_18
  simp only [← c66066, ← c66067, ← c66068] at e_1067_18
  have e_1067_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1067 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1067_19
  simp only [← c66069, ← c66070, k2, ← c66071] at e_1067_19
  have hr := e_1067_19
  simp only [e_1067_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9045)) := by
  have c66072 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66073 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66074 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66075 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66076 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66077 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_0
  simp only [← c66072, ← c66073, ← c66074] at e_1068_0
  have e_1068_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_1
  simp only [← c66075, ← c66076, k2, ← c66077] at e_1068_1
  have hr := e_1068_1
  simp only [e_1068_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9046)) := by
  have c66078 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66079 := (publicBatchWrapper89_copies2064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66080 := (publicBatchWrapper89_copies2065 a h).1
  have c66081 := (publicBatchWrapper89_copies2065 a h).2.1
  have c66082 := (publicBatchWrapper89_copies2065 a h).2.2.1
  have c66083 := (publicBatchWrapper89_copies2065 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_2
  simp only [← c66078, ← c66079, ← c66080] at e_1068_2
  have e_1068_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_3
  simp only [← c66081, ← c66082, k2, ← c66083] at e_1068_3
  have hr := e_1068_3
  simp only [e_1068_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9047)) := by
  have c66084 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.1
  have c66085 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.1
  have c66086 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.1
  have c66087 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.1
  have c66088 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.1
  have c66089 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_4
  simp only [← c66084, ← c66085, ← c66086] at e_1068_4
  have e_1068_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_5
  simp only [← c66087, ← c66088, k2, ← c66089] at e_1068_5
  have hr := e_1068_5
  simp only [e_1068_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9048)) := by
  have c66090 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66091 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66092 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66093 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66094 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66095 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_6
  simp only [← c66090, ← c66091, ← c66092] at e_1068_6
  have e_1068_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_7
  simp only [← c66093, ← c66094, k2, ← c66095] at e_1068_7
  have hr := e_1068_7
  simp only [e_1068_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9049)) := by
  have c66096 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66097 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66098 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66099 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66100 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66101 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_8
  simp only [← c66096, ← c66097, ← c66098] at e_1068_8
  have e_1068_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_9
  simp only [← c66099, ← c66100, k2, ← c66101] at e_1068_9
  have hr := e_1068_9
  simp only [e_1068_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9050)) := by
  have c66102 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66103 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66104 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66105 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66106 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66107 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_10
  simp only [← c66102, ← c66103, ← c66104] at e_1068_10
  have e_1068_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_11
  simp only [← c66105, ← c66106, k2, ← c66107] at e_1068_11
  have hr := e_1068_11
  simp only [e_1068_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9051)) := by
  have c66108 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66109 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66110 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66111 := (publicBatchWrapper89_copies2065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66112 := (publicBatchWrapper89_copies2066 a h).1
  have c66113 := (publicBatchWrapper89_copies2066 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_12
  simp only [← c66108, ← c66109, ← c66110] at e_1068_12
  have e_1068_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_13
  simp only [← c66111, ← c66112, k2, ← c66113] at e_1068_13
  have hr := e_1068_13
  simp only [e_1068_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9052)) := by
  have c66114 := (publicBatchWrapper89_copies2066 a h).2.2.1
  have c66115 := (publicBatchWrapper89_copies2066 a h).2.2.2.1
  have c66116 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.1
  have c66117 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.1
  have c66118 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.1
  have c66119 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_14
  simp only [← c66114, ← c66115, ← c66116] at e_1068_14
  have e_1068_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_15
  simp only [← c66117, ← c66118, k2, ← c66119] at e_1068_15
  have hr := e_1068_15
  simp only [e_1068_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9053)) := by
  have c66120 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.1
  have c66121 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.1
  have c66122 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66123 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66124 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66125 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_16
  simp only [← c66120, ← c66121, ← c66122] at e_1068_16
  have e_1068_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_17
  simp only [← c66123, ← c66124, k2, ← c66125] at e_1068_17
  have hr := e_1068_17
  simp only [e_1068_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1068 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9054)) := by
  have c66126 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66127 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66128 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66129 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66130 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66131 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1068_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_18
  simp only [← c66126, ← c66127, ← c66128] at e_1068_18
  have e_1068_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1068 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1068_19
  simp only [← c66129, ← c66130, k2, ← c66131] at e_1068_19
  have hr := e_1068_19
  simp only [e_1068_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9055)) := by
  have c66132 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66133 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66134 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66135 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66136 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66137 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_0
  simp only [← c66132, ← c66133, ← c66134] at e_1069_0
  have e_1069_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_1
  simp only [← c66135, ← c66136, k2, ← c66137] at e_1069_1
  have hr := e_1069_1
  simp only [e_1069_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9056)) := by
  have c66138 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66139 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66140 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66141 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66142 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66143 := (publicBatchWrapper89_copies2066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_2
  simp only [← c66138, ← c66139, ← c66140] at e_1069_2
  have e_1069_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_3
  simp only [← c66141, ← c66142, k2, ← c66143] at e_1069_3
  have hr := e_1069_3
  simp only [e_1069_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9057)) := by
  have c66144 := (publicBatchWrapper89_copies2067 a h).1
  have c66145 := (publicBatchWrapper89_copies2067 a h).2.1
  have c66146 := (publicBatchWrapper89_copies2067 a h).2.2.1
  have c66147 := (publicBatchWrapper89_copies2067 a h).2.2.2.1
  have c66148 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.1
  have c66149 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_4
  simp only [← c66144, ← c66145, ← c66146] at e_1069_4
  have e_1069_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_5
  simp only [← c66147, ← c66148, k2, ← c66149] at e_1069_5
  have hr := e_1069_5
  simp only [e_1069_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9058)) := by
  have c66150 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.1
  have c66151 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.1
  have c66152 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.1
  have c66153 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.1
  have c66154 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66155 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_6
  simp only [← c66150, ← c66151, ← c66152] at e_1069_6
  have e_1069_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_7
  simp only [← c66153, ← c66154, k2, ← c66155] at e_1069_7
  have hr := e_1069_7
  simp only [e_1069_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9059)) := by
  have c66156 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66157 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66158 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66159 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66160 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66161 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_8
  simp only [← c66156, ← c66157, ← c66158] at e_1069_8
  have e_1069_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_9
  simp only [← c66159, ← c66160, k2, ← c66161] at e_1069_9
  have hr := e_1069_9
  simp only [e_1069_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9060)) := by
  have c66162 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66163 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66164 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66165 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66166 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66167 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_10
  simp only [← c66162, ← c66163, ← c66164] at e_1069_10
  have e_1069_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_11
  simp only [← c66165, ← c66166, k2, ← c66167] at e_1069_11
  have hr := e_1069_11
  simp only [e_1069_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9061)) := by
  have c66168 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66169 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66170 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66171 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66172 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66173 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_12
  simp only [← c66168, ← c66169, ← c66170] at e_1069_12
  have e_1069_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_13
  simp only [← c66171, ← c66172, k2, ← c66173] at e_1069_13
  have hr := e_1069_13
  simp only [e_1069_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9062)) := by
  have c66174 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66175 := (publicBatchWrapper89_copies2067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66176 := (publicBatchWrapper89_copies2068 a h).1
  have c66177 := (publicBatchWrapper89_copies2068 a h).2.1
  have c66178 := (publicBatchWrapper89_copies2068 a h).2.2.1
  have c66179 := (publicBatchWrapper89_copies2068 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_14
  simp only [← c66174, ← c66175, ← c66176] at e_1069_14
  have e_1069_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_15
  simp only [← c66177, ← c66178, k2, ← c66179] at e_1069_15
  have hr := e_1069_15
  simp only [e_1069_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9063)) := by
  have c66180 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.1
  have c66181 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.1
  have c66182 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.1
  have c66183 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.1
  have c66184 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.1
  have c66185 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_16
  simp only [← c66180, ← c66181, ← c66182] at e_1069_16
  have e_1069_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_17
  simp only [← c66183, ← c66184, k2, ← c66185] at e_1069_17
  have hr := e_1069_17
  simp only [e_1069_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1069 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9064)) := by
  have c66186 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66187 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66188 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66189 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66190 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66191 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1069_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_18
  simp only [← c66186, ← c66187, ← c66188] at e_1069_18
  have e_1069_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1069 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1069_19
  simp only [← c66189, ← c66190, k2, ← c66191] at e_1069_19
  have hr := e_1069_19
  simp only [e_1069_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9065)) := by
  have c66192 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66193 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66194 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66195 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66196 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66197 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_0
  simp only [← c66192, ← c66193, ← c66194] at e_1070_0
  have e_1070_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_1
  simp only [← c66195, ← c66196, k2, ← c66197] at e_1070_1
  have hr := e_1070_1
  simp only [e_1070_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9066)) := by
  have c66198 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66199 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66200 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66201 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66202 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66203 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_2
  simp only [← c66198, ← c66199, ← c66200] at e_1070_2
  have e_1070_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_3
  simp only [← c66201, ← c66202, k2, ← c66203] at e_1070_3
  have hr := e_1070_3
  simp only [e_1070_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9067)) := by
  have c66204 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66205 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66206 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66207 := (publicBatchWrapper89_copies2068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66208 := (publicBatchWrapper89_copies2069 a h).1
  have c66209 := (publicBatchWrapper89_copies2069 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_4
  simp only [← c66204, ← c66205, ← c66206] at e_1070_4
  have e_1070_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_5
  simp only [← c66207, ← c66208, k2, ← c66209] at e_1070_5
  have hr := e_1070_5
  simp only [e_1070_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9068)) := by
  have c66210 := (publicBatchWrapper89_copies2069 a h).2.2.1
  have c66211 := (publicBatchWrapper89_copies2069 a h).2.2.2.1
  have c66212 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.1
  have c66213 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.1
  have c66214 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.1
  have c66215 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_6
  simp only [← c66210, ← c66211, ← c66212] at e_1070_6
  have e_1070_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_7
  simp only [← c66213, ← c66214, k2, ← c66215] at e_1070_7
  have hr := e_1070_7
  simp only [e_1070_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9069)) := by
  have c66216 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.1
  have c66217 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.1
  have c66218 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66219 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66220 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66221 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_8
  simp only [← c66216, ← c66217, ← c66218] at e_1070_8
  have e_1070_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_9
  simp only [← c66219, ← c66220, k2, ← c66221] at e_1070_9
  have hr := e_1070_9
  simp only [e_1070_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9070)) := by
  have c66222 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66223 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66224 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66225 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66226 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66227 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_10
  simp only [← c66222, ← c66223, ← c66224] at e_1070_10
  have e_1070_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_11
  simp only [← c66225, ← c66226, k2, ← c66227] at e_1070_11
  have hr := e_1070_11
  simp only [e_1070_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9071)) := by
  have c66228 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66229 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66230 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66231 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66232 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66233 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_12
  simp only [← c66228, ← c66229, ← c66230] at e_1070_12
  have e_1070_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_13
  simp only [← c66231, ← c66232, k2, ← c66233] at e_1070_13
  have hr := e_1070_13
  simp only [e_1070_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9072)) := by
  have c66234 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66235 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66236 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66237 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66238 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66239 := (publicBatchWrapper89_copies2069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_14
  simp only [← c66234, ← c66235, ← c66236] at e_1070_14
  have e_1070_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_15
  simp only [← c66237, ← c66238, k2, ← c66239] at e_1070_15
  have hr := e_1070_15
  simp only [e_1070_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9073)) := by
  have c66240 := (publicBatchWrapper89_copies2070 a h).1
  have c66241 := (publicBatchWrapper89_copies2070 a h).2.1
  have c66242 := (publicBatchWrapper89_copies2070 a h).2.2.1
  have c66243 := (publicBatchWrapper89_copies2070 a h).2.2.2.1
  have c66244 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.1
  have c66245 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_16
  simp only [← c66240, ← c66241, ← c66242] at e_1070_16
  have e_1070_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_17
  simp only [← c66243, ← c66244, k2, ← c66245] at e_1070_17
  have hr := e_1070_17
  simp only [e_1070_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1070 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9074)) := by
  have c66246 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.1
  have c66247 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.1
  have c66248 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.1
  have c66249 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.1
  have c66250 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66251 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1070_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_18
  simp only [← c66246, ← c66247, ← c66248] at e_1070_18
  have e_1070_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1070 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1070_19
  simp only [← c66249, ← c66250, k2, ← c66251] at e_1070_19
  have hr := e_1070_19
  simp only [e_1070_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9075)) := by
  have c66252 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66253 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66254 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66255 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66256 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66257 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_0
  simp only [← c66252, ← c66253, ← c66254] at e_1071_0
  have e_1071_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_1
  simp only [← c66255, ← c66256, k2, ← c66257] at e_1071_1
  have hr := e_1071_1
  simp only [e_1071_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9076)) := by
  have c66258 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66259 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66260 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66261 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66262 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66263 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_2
  simp only [← c66258, ← c66259, ← c66260] at e_1071_2
  have e_1071_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_3
  simp only [← c66261, ← c66262, k2, ← c66263] at e_1071_3
  have hr := e_1071_3
  simp only [e_1071_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9077)) := by
  have c66264 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66265 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66266 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66267 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66268 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66269 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_4
  simp only [← c66264, ← c66265, ← c66266] at e_1071_4
  have e_1071_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_5
  simp only [← c66267, ← c66268, k2, ← c66269] at e_1071_5
  have hr := e_1071_5
  simp only [e_1071_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9078)) := by
  have c66270 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66271 := (publicBatchWrapper89_copies2070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66272 := (publicBatchWrapper89_copies2071 a h).1
  have c66273 := (publicBatchWrapper89_copies2071 a h).2.1
  have c66274 := (publicBatchWrapper89_copies2071 a h).2.2.1
  have c66275 := (publicBatchWrapper89_copies2071 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_6
  simp only [← c66270, ← c66271, ← c66272] at e_1071_6
  have e_1071_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_7
  simp only [← c66273, ← c66274, k2, ← c66275] at e_1071_7
  have hr := e_1071_7
  simp only [e_1071_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9079)) := by
  have c66276 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.1
  have c66277 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.1
  have c66278 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.1
  have c66279 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.1
  have c66280 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.1
  have c66281 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_8
  simp only [← c66276, ← c66277, ← c66278] at e_1071_8
  have e_1071_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_9
  simp only [← c66279, ← c66280, k2, ← c66281] at e_1071_9
  have hr := e_1071_9
  simp only [e_1071_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9080)) := by
  have c66282 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66283 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66284 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66285 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66286 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66287 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_10
  simp only [← c66282, ← c66283, ← c66284] at e_1071_10
  have e_1071_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_11
  simp only [← c66285, ← c66286, k2, ← c66287] at e_1071_11
  have hr := e_1071_11
  simp only [e_1071_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9081)) := by
  have c66288 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66289 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66290 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66291 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66292 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66293 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_12
  simp only [← c66288, ← c66289, ← c66290] at e_1071_12
  have e_1071_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_13
  simp only [← c66291, ← c66292, k2, ← c66293] at e_1071_13
  have hr := e_1071_13
  simp only [e_1071_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9082)) := by
  have c66294 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66295 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66296 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66297 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66298 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66299 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_14
  simp only [← c66294, ← c66295, ← c66296] at e_1071_14
  have e_1071_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_15
  simp only [← c66297, ← c66298, k2, ← c66299] at e_1071_15
  have hr := e_1071_15
  simp only [e_1071_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9083)) := by
  have c66300 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66301 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66302 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66303 := (publicBatchWrapper89_copies2071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66304 := (publicBatchWrapper89_copies2072 a h).1
  have c66305 := (publicBatchWrapper89_copies2072 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_16
  simp only [← c66300, ← c66301, ← c66302] at e_1071_16
  have e_1071_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_17
  simp only [← c66303, ← c66304, k2, ← c66305] at e_1071_17
  have hr := e_1071_17
  simp only [e_1071_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1071 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9084)) := by
  have c66306 := (publicBatchWrapper89_copies2072 a h).2.2.1
  have c66307 := (publicBatchWrapper89_copies2072 a h).2.2.2.1
  have c66308 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.1
  have c66309 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.1
  have c66310 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.1
  have c66311 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1071_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_18
  simp only [← c66306, ← c66307, ← c66308] at e_1071_18
  have e_1071_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1071 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1071_19
  simp only [← c66309, ← c66310, k2, ← c66311] at e_1071_19
  have hr := e_1071_19
  simp only [e_1071_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9085)) := by
  have c66312 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.1
  have c66313 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.1
  have c66314 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66315 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66316 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66317 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_0
  simp only [← c66312, ← c66313, ← c66314] at e_1072_0
  have e_1072_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_1
  simp only [← c66315, ← c66316, k2, ← c66317] at e_1072_1
  have hr := e_1072_1
  simp only [e_1072_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9086)) := by
  have c66318 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66319 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66320 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66321 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66322 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66323 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_2
  simp only [← c66318, ← c66319, ← c66320] at e_1072_2
  have e_1072_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_3
  simp only [← c66321, ← c66322, k2, ← c66323] at e_1072_3
  have hr := e_1072_3
  simp only [e_1072_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9087)) := by
  have c66324 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66325 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66326 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66327 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66328 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66329 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_4
  simp only [← c66324, ← c66325, ← c66326] at e_1072_4
  have e_1072_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_5
  simp only [← c66327, ← c66328, k2, ← c66329] at e_1072_5
  have hr := e_1072_5
  simp only [e_1072_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9088)) := by
  have c66330 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66331 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66332 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66333 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66334 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66335 := (publicBatchWrapper89_copies2072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_6
  simp only [← c66330, ← c66331, ← c66332] at e_1072_6
  have e_1072_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_7
  simp only [← c66333, ← c66334, k2, ← c66335] at e_1072_7
  have hr := e_1072_7
  simp only [e_1072_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9089)) := by
  have c66336 := (publicBatchWrapper89_copies2073 a h).1
  have c66337 := (publicBatchWrapper89_copies2073 a h).2.1
  have c66338 := (publicBatchWrapper89_copies2073 a h).2.2.1
  have c66339 := (publicBatchWrapper89_copies2073 a h).2.2.2.1
  have c66340 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.1
  have c66341 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_8
  simp only [← c66336, ← c66337, ← c66338] at e_1072_8
  have e_1072_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_9
  simp only [← c66339, ← c66340, k2, ← c66341] at e_1072_9
  have hr := e_1072_9
  simp only [e_1072_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9090)) := by
  have c66342 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.1
  have c66343 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.1
  have c66344 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.1
  have c66345 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.1
  have c66346 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66347 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_10
  simp only [← c66342, ← c66343, ← c66344] at e_1072_10
  have e_1072_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_11
  simp only [← c66345, ← c66346, k2, ← c66347] at e_1072_11
  have hr := e_1072_11
  simp only [e_1072_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9091)) := by
  have c66348 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66349 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66350 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66351 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66352 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66353 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_12
  simp only [← c66348, ← c66349, ← c66350] at e_1072_12
  have e_1072_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_13
  simp only [← c66351, ← c66352, k2, ← c66353] at e_1072_13
  have hr := e_1072_13
  simp only [e_1072_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9092)) := by
  have c66354 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66355 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66356 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66357 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66358 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66359 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_14
  simp only [← c66354, ← c66355, ← c66356] at e_1072_14
  have e_1072_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_15
  simp only [← c66357, ← c66358, k2, ← c66359] at e_1072_15
  have hr := e_1072_15
  simp only [e_1072_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9093)) := by
  have c66360 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66361 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66362 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66363 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66364 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66365 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_16
  simp only [← c66360, ← c66361, ← c66362] at e_1072_16
  have e_1072_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_17
  simp only [← c66363, ← c66364, k2, ← c66365] at e_1072_17
  have hr := e_1072_17
  simp only [e_1072_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1072 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9094)) := by
  have c66366 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66367 := (publicBatchWrapper89_copies2073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66368 := (publicBatchWrapper89_copies2074 a h).1
  have c66369 := (publicBatchWrapper89_copies2074 a h).2.1
  have c66370 := (publicBatchWrapper89_copies2074 a h).2.2.1
  have c66371 := (publicBatchWrapper89_copies2074 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1072_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_18
  simp only [← c66366, ← c66367, ← c66368] at e_1072_18
  have e_1072_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1072 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1072_19
  simp only [← c66369, ← c66370, k2, ← c66371] at e_1072_19
  have hr := e_1072_19
  simp only [e_1072_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9095)) := by
  have c66372 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.1
  have c66373 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.1
  have c66374 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.1
  have c66375 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.1
  have c66376 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.1
  have c66377 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_0
  simp only [← c66372, ← c66373, ← c66374] at e_1073_0
  have e_1073_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_1
  simp only [← c66375, ← c66376, k2, ← c66377] at e_1073_1
  have hr := e_1073_1
  simp only [e_1073_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9096)) := by
  have c66378 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66379 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66380 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66381 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66382 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66383 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_2
  simp only [← c66378, ← c66379, ← c66380] at e_1073_2
  have e_1073_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_3
  simp only [← c66381, ← c66382, k2, ← c66383] at e_1073_3
  have hr := e_1073_3
  simp only [e_1073_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9097)) := by
  have c66384 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66385 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66386 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66387 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66388 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66389 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_4
  simp only [← c66384, ← c66385, ← c66386] at e_1073_4
  have e_1073_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_5
  simp only [← c66387, ← c66388, k2, ← c66389] at e_1073_5
  have hr := e_1073_5
  simp only [e_1073_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9098)) := by
  have c66390 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66391 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66392 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66393 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66394 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66395 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_6
  simp only [← c66390, ← c66391, ← c66392] at e_1073_6
  have e_1073_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_7
  simp only [← c66393, ← c66394, k2, ← c66395] at e_1073_7
  have hr := e_1073_7
  simp only [e_1073_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9099)) := by
  have c66396 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66397 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66398 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66399 := (publicBatchWrapper89_copies2074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66400 := (publicBatchWrapper89_copies2075 a h).1
  have c66401 := (publicBatchWrapper89_copies2075 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_8
  simp only [← c66396, ← c66397, ← c66398] at e_1073_8
  have e_1073_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_9
  simp only [← c66399, ← c66400, k2, ← c66401] at e_1073_9
  have hr := e_1073_9
  simp only [e_1073_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9100)) := by
  have c66402 := (publicBatchWrapper89_copies2075 a h).2.2.1
  have c66403 := (publicBatchWrapper89_copies2075 a h).2.2.2.1
  have c66404 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.1
  have c66405 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.1
  have c66406 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.1
  have c66407 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_10
  simp only [← c66402, ← c66403, ← c66404] at e_1073_10
  have e_1073_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_11
  simp only [← c66405, ← c66406, k2, ← c66407] at e_1073_11
  have hr := e_1073_11
  simp only [e_1073_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9101)) := by
  have c66408 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.1
  have c66409 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.1
  have c66410 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66411 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66412 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66413 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_12
  simp only [← c66408, ← c66409, ← c66410] at e_1073_12
  have e_1073_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_13
  simp only [← c66411, ← c66412, k2, ← c66413] at e_1073_13
  have hr := e_1073_13
  simp only [e_1073_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9102)) := by
  have c66414 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66415 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66416 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66417 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66418 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66419 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_14
  simp only [← c66414, ← c66415, ← c66416] at e_1073_14
  have e_1073_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_15
  simp only [← c66417, ← c66418, k2, ← c66419] at e_1073_15
  have hr := e_1073_15
  simp only [e_1073_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9103)) := by
  have c66420 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66421 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66422 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66423 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66424 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66425 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_16
  simp only [← c66420, ← c66421, ← c66422] at e_1073_16
  have e_1073_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_17
  simp only [← c66423, ← c66424, k2, ← c66425] at e_1073_17
  have hr := e_1073_17
  simp only [e_1073_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1073 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9104)) := by
  have c66426 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66427 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66428 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66429 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66430 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66431 := (publicBatchWrapper89_copies2075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1073_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_18
  simp only [← c66426, ← c66427, ← c66428] at e_1073_18
  have e_1073_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1073 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1073_19
  simp only [← c66429, ← c66430, k2, ← c66431] at e_1073_19
  have hr := e_1073_19
  simp only [e_1073_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9105)) := by
  have c66432 := (publicBatchWrapper89_copies2076 a h).1
  have c66433 := (publicBatchWrapper89_copies2076 a h).2.1
  have c66434 := (publicBatchWrapper89_copies2076 a h).2.2.1
  have c66435 := (publicBatchWrapper89_copies2076 a h).2.2.2.1
  have c66436 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.1
  have c66437 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_0
  simp only [← c66432, ← c66433, ← c66434] at e_1074_0
  have e_1074_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_1
  simp only [← c66435, ← c66436, k2, ← c66437] at e_1074_1
  have hr := e_1074_1
  simp only [e_1074_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9106)) := by
  have c66438 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.1
  have c66439 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.1
  have c66440 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.1
  have c66441 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.1
  have c66442 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66443 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_2
  simp only [← c66438, ← c66439, ← c66440] at e_1074_2
  have e_1074_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_3
  simp only [← c66441, ← c66442, k2, ← c66443] at e_1074_3
  have hr := e_1074_3
  simp only [e_1074_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9107)) := by
  have c66444 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66445 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66446 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66447 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66448 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66449 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_4
  simp only [← c66444, ← c66445, ← c66446] at e_1074_4
  have e_1074_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_5
  simp only [← c66447, ← c66448, k2, ← c66449] at e_1074_5
  have hr := e_1074_5
  simp only [e_1074_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9108)) := by
  have c66450 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66451 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66452 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66453 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66454 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66455 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_6
  simp only [← c66450, ← c66451, ← c66452] at e_1074_6
  have e_1074_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_7
  simp only [← c66453, ← c66454, k2, ← c66455] at e_1074_7
  have hr := e_1074_7
  simp only [e_1074_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9109)) := by
  have c66456 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66457 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66458 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66459 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66460 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66461 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_8
  simp only [← c66456, ← c66457, ← c66458] at e_1074_8
  have e_1074_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_9
  simp only [← c66459, ← c66460, k2, ← c66461] at e_1074_9
  have hr := e_1074_9
  simp only [e_1074_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9110)) := by
  have c66462 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66463 := (publicBatchWrapper89_copies2076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66464 := (publicBatchWrapper89_copies2077 a h).1
  have c66465 := (publicBatchWrapper89_copies2077 a h).2.1
  have c66466 := (publicBatchWrapper89_copies2077 a h).2.2.1
  have c66467 := (publicBatchWrapper89_copies2077 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_10
  simp only [← c66462, ← c66463, ← c66464] at e_1074_10
  have e_1074_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_11
  simp only [← c66465, ← c66466, k2, ← c66467] at e_1074_11
  have hr := e_1074_11
  simp only [e_1074_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9111)) := by
  have c66468 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.1
  have c66469 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.1
  have c66470 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.1
  have c66471 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.1
  have c66472 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.1
  have c66473 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_12
  simp only [← c66468, ← c66469, ← c66470] at e_1074_12
  have e_1074_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_13
  simp only [← c66471, ← c66472, k2, ← c66473] at e_1074_13
  have hr := e_1074_13
  simp only [e_1074_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9112)) := by
  have c66474 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66475 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66476 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66477 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66478 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66479 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_14
  simp only [← c66474, ← c66475, ← c66476] at e_1074_14
  have e_1074_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_15
  simp only [← c66477, ← c66478, k2, ← c66479] at e_1074_15
  have hr := e_1074_15
  simp only [e_1074_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9113)) := by
  have c66480 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66481 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66482 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66483 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66484 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66485 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_16
  simp only [← c66480, ← c66481, ← c66482] at e_1074_16
  have e_1074_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_17
  simp only [← c66483, ← c66484, k2, ← c66485] at e_1074_17
  have hr := e_1074_17
  simp only [e_1074_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1074 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9114)) := by
  have c66486 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66487 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66488 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66489 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66490 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66491 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1074_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_18
  simp only [← c66486, ← c66487, ← c66488] at e_1074_18
  have e_1074_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1074 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1074_19
  simp only [← c66489, ← c66490, k2, ← c66491] at e_1074_19
  have hr := e_1074_19
  simp only [e_1074_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9115)) := by
  have c66492 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66493 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66494 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66495 := (publicBatchWrapper89_copies2077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66496 := (publicBatchWrapper89_copies2078 a h).1
  have c66497 := (publicBatchWrapper89_copies2078 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_0
  simp only [← c66492, ← c66493, ← c66494] at e_1075_0
  have e_1075_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_1
  simp only [← c66495, ← c66496, k2, ← c66497] at e_1075_1
  have hr := e_1075_1
  simp only [e_1075_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9116)) := by
  have c66498 := (publicBatchWrapper89_copies2078 a h).2.2.1
  have c66499 := (publicBatchWrapper89_copies2078 a h).2.2.2.1
  have c66500 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.1
  have c66501 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.1
  have c66502 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.1
  have c66503 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_2
  simp only [← c66498, ← c66499, ← c66500] at e_1075_2
  have e_1075_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_3
  simp only [← c66501, ← c66502, k2, ← c66503] at e_1075_3
  have hr := e_1075_3
  simp only [e_1075_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9117)) := by
  have c66504 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.1
  have c66505 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.1
  have c66506 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66507 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66508 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66509 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_4
  simp only [← c66504, ← c66505, ← c66506] at e_1075_4
  have e_1075_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_5
  simp only [← c66507, ← c66508, k2, ← c66509] at e_1075_5
  have hr := e_1075_5
  simp only [e_1075_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9118)) := by
  have c66510 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66511 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66512 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66513 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66514 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66515 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_6
  simp only [← c66510, ← c66511, ← c66512] at e_1075_6
  have e_1075_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_7
  simp only [← c66513, ← c66514, k2, ← c66515] at e_1075_7
  have hr := e_1075_7
  simp only [e_1075_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9119)) := by
  have c66516 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66517 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66518 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66519 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66520 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66521 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_8
  simp only [← c66516, ← c66517, ← c66518] at e_1075_8
  have e_1075_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_9
  simp only [← c66519, ← c66520, k2, ← c66521] at e_1075_9
  have hr := e_1075_9
  simp only [e_1075_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9120)) := by
  have c66522 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66523 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66524 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66525 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66526 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66527 := (publicBatchWrapper89_copies2078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_10
  simp only [← c66522, ← c66523, ← c66524] at e_1075_10
  have e_1075_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_11
  simp only [← c66525, ← c66526, k2, ← c66527] at e_1075_11
  have hr := e_1075_11
  simp only [e_1075_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9121)) := by
  have c66528 := (publicBatchWrapper89_copies2079 a h).1
  have c66529 := (publicBatchWrapper89_copies2079 a h).2.1
  have c66530 := (publicBatchWrapper89_copies2079 a h).2.2.1
  have c66531 := (publicBatchWrapper89_copies2079 a h).2.2.2.1
  have c66532 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.1
  have c66533 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_12
  simp only [← c66528, ← c66529, ← c66530] at e_1075_12
  have e_1075_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_13
  simp only [← c66531, ← c66532, k2, ← c66533] at e_1075_13
  have hr := e_1075_13
  simp only [e_1075_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9122)) := by
  have c66534 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.1
  have c66535 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.1
  have c66536 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.1
  have c66537 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.1
  have c66538 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66539 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_14
  simp only [← c66534, ← c66535, ← c66536] at e_1075_14
  have e_1075_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_15
  simp only [← c66537, ← c66538, k2, ← c66539] at e_1075_15
  have hr := e_1075_15
  simp only [e_1075_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9123)) := by
  have c66540 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66541 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66542 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66543 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66544 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66545 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_16
  simp only [← c66540, ← c66541, ← c66542] at e_1075_16
  have e_1075_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_17
  simp only [← c66543, ← c66544, k2, ← c66545] at e_1075_17
  have hr := e_1075_17
  simp only [e_1075_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1075 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9124)) := by
  have c66546 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66547 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66548 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66549 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66550 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66551 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1075_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_18
  simp only [← c66546, ← c66547, ← c66548] at e_1075_18
  have e_1075_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1075 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1075_19
  simp only [← c66549, ← c66550, k2, ← c66551] at e_1075_19
  have hr := e_1075_19
  simp only [e_1075_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9125)) := by
  have c66552 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66553 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66554 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66555 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66556 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66557 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_0
  simp only [← c66552, ← c66553, ← c66554] at e_1076_0
  have e_1076_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_1
  simp only [← c66555, ← c66556, k2, ← c66557] at e_1076_1
  have hr := e_1076_1
  simp only [e_1076_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9126)) := by
  have c66558 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66559 := (publicBatchWrapper89_copies2079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66560 := (publicBatchWrapper89_copies2080 a h).1
  have c66561 := (publicBatchWrapper89_copies2080 a h).2.1
  have c66562 := (publicBatchWrapper89_copies2080 a h).2.2.1
  have c66563 := (publicBatchWrapper89_copies2080 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_2
  simp only [← c66558, ← c66559, ← c66560] at e_1076_2
  have e_1076_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_3
  simp only [← c66561, ← c66562, k2, ← c66563] at e_1076_3
  have hr := e_1076_3
  simp only [e_1076_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9127)) := by
  have c66564 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.1
  have c66565 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.1
  have c66566 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.1
  have c66567 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.1
  have c66568 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.1
  have c66569 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_4
  simp only [← c66564, ← c66565, ← c66566] at e_1076_4
  have e_1076_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_5
  simp only [← c66567, ← c66568, k2, ← c66569] at e_1076_5
  have hr := e_1076_5
  simp only [e_1076_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9128)) := by
  have c66570 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66571 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66572 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66573 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66574 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66575 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_6
  simp only [← c66570, ← c66571, ← c66572] at e_1076_6
  have e_1076_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_7
  simp only [← c66573, ← c66574, k2, ← c66575] at e_1076_7
  have hr := e_1076_7
  simp only [e_1076_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9129)) := by
  have c66576 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66577 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66578 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66579 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66580 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66581 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_8
  simp only [← c66576, ← c66577, ← c66578] at e_1076_8
  have e_1076_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_9
  simp only [← c66579, ← c66580, k2, ← c66581] at e_1076_9
  have hr := e_1076_9
  simp only [e_1076_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9130)) := by
  have c66582 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66583 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66584 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66585 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66586 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66587 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_10
  simp only [← c66582, ← c66583, ← c66584] at e_1076_10
  have e_1076_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_11
  simp only [← c66585, ← c66586, k2, ← c66587] at e_1076_11
  have hr := e_1076_11
  simp only [e_1076_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9131)) := by
  have c66588 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66589 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66590 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66591 := (publicBatchWrapper89_copies2080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66592 := (publicBatchWrapper89_copies2081 a h).1
  have c66593 := (publicBatchWrapper89_copies2081 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_12
  simp only [← c66588, ← c66589, ← c66590] at e_1076_12
  have e_1076_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_13
  simp only [← c66591, ← c66592, k2, ← c66593] at e_1076_13
  have hr := e_1076_13
  simp only [e_1076_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9132)) := by
  have c66594 := (publicBatchWrapper89_copies2081 a h).2.2.1
  have c66595 := (publicBatchWrapper89_copies2081 a h).2.2.2.1
  have c66596 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.1
  have c66597 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.1
  have c66598 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.1
  have c66599 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_14
  simp only [← c66594, ← c66595, ← c66596] at e_1076_14
  have e_1076_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_15
  simp only [← c66597, ← c66598, k2, ← c66599] at e_1076_15
  have hr := e_1076_15
  simp only [e_1076_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9133)) := by
  have c66600 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.1
  have c66601 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.1
  have c66602 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66603 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66604 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66605 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_16
  simp only [← c66600, ← c66601, ← c66602] at e_1076_16
  have e_1076_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_17
  simp only [← c66603, ← c66604, k2, ← c66605] at e_1076_17
  have hr := e_1076_17
  simp only [e_1076_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1076 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9134)) := by
  have c66606 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66607 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66608 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66609 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66610 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66611 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1076_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_18
  simp only [← c66606, ← c66607, ← c66608] at e_1076_18
  have e_1076_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1076 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1076_19
  simp only [← c66609, ← c66610, k2, ← c66611] at e_1076_19
  have hr := e_1076_19
  simp only [e_1076_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9135)) := by
  have c66612 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66613 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66614 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66615 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66616 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66617 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_0
  simp only [← c66612, ← c66613, ← c66614] at e_1077_0
  have e_1077_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_1
  simp only [← c66615, ← c66616, k2, ← c66617] at e_1077_1
  have hr := e_1077_1
  simp only [e_1077_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9136)) := by
  have c66618 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66619 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66620 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66621 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66622 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66623 := (publicBatchWrapper89_copies2081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_2
  simp only [← c66618, ← c66619, ← c66620] at e_1077_2
  have e_1077_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_3
  simp only [← c66621, ← c66622, k2, ← c66623] at e_1077_3
  have hr := e_1077_3
  simp only [e_1077_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9137)) := by
  have c66624 := (publicBatchWrapper89_copies2082 a h).1
  have c66625 := (publicBatchWrapper89_copies2082 a h).2.1
  have c66626 := (publicBatchWrapper89_copies2082 a h).2.2.1
  have c66627 := (publicBatchWrapper89_copies2082 a h).2.2.2.1
  have c66628 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.1
  have c66629 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_4
  simp only [← c66624, ← c66625, ← c66626] at e_1077_4
  have e_1077_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_5
  simp only [← c66627, ← c66628, k2, ← c66629] at e_1077_5
  have hr := e_1077_5
  simp only [e_1077_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9138)) := by
  have c66630 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.1
  have c66631 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.1
  have c66632 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.1
  have c66633 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.1
  have c66634 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66635 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_6
  simp only [← c66630, ← c66631, ← c66632] at e_1077_6
  have e_1077_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_7
  simp only [← c66633, ← c66634, k2, ← c66635] at e_1077_7
  have hr := e_1077_7
  simp only [e_1077_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9139)) := by
  have c66636 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66637 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66638 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66639 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66640 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66641 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_8
  simp only [← c66636, ← c66637, ← c66638] at e_1077_8
  have e_1077_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_9
  simp only [← c66639, ← c66640, k2, ← c66641] at e_1077_9
  have hr := e_1077_9
  simp only [e_1077_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9140)) := by
  have c66642 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66643 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66644 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66645 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66646 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66647 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_10
  simp only [← c66642, ← c66643, ← c66644] at e_1077_10
  have e_1077_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_11
  simp only [← c66645, ← c66646, k2, ← c66647] at e_1077_11
  have hr := e_1077_11
  simp only [e_1077_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9141)) := by
  have c66648 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66649 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66650 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66651 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66652 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66653 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_12
  simp only [← c66648, ← c66649, ← c66650] at e_1077_12
  have e_1077_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_13
  simp only [← c66651, ← c66652, k2, ← c66653] at e_1077_13
  have hr := e_1077_13
  simp only [e_1077_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9142)) := by
  have c66654 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66655 := (publicBatchWrapper89_copies2082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66656 := (publicBatchWrapper89_copies2083 a h).1
  have c66657 := (publicBatchWrapper89_copies2083 a h).2.1
  have c66658 := (publicBatchWrapper89_copies2083 a h).2.2.1
  have c66659 := (publicBatchWrapper89_copies2083 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_14
  simp only [← c66654, ← c66655, ← c66656] at e_1077_14
  have e_1077_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_15
  simp only [← c66657, ← c66658, k2, ← c66659] at e_1077_15
  have hr := e_1077_15
  simp only [e_1077_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9143)) := by
  have c66660 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.1
  have c66661 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.1
  have c66662 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.1
  have c66663 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.1
  have c66664 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.1
  have c66665 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_16
  simp only [← c66660, ← c66661, ← c66662] at e_1077_16
  have e_1077_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_17
  simp only [← c66663, ← c66664, k2, ← c66665] at e_1077_17
  have hr := e_1077_17
  simp only [e_1077_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1077 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9144)) := by
  have c66666 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66667 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66668 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66669 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66670 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66671 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1077_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_18
  simp only [← c66666, ← c66667, ← c66668] at e_1077_18
  have e_1077_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1077 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1077_19
  simp only [← c66669, ← c66670, k2, ← c66671] at e_1077_19
  have hr := e_1077_19
  simp only [e_1077_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9145)) := by
  have c66672 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66673 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66674 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66675 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66676 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66677 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_0
  simp only [← c66672, ← c66673, ← c66674] at e_1078_0
  have e_1078_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_1
  simp only [← c66675, ← c66676, k2, ← c66677] at e_1078_1
  have hr := e_1078_1
  simp only [e_1078_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9146)) := by
  have c66678 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66679 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66680 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66681 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66682 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66683 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_2
  simp only [← c66678, ← c66679, ← c66680] at e_1078_2
  have e_1078_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_3
  simp only [← c66681, ← c66682, k2, ← c66683] at e_1078_3
  have hr := e_1078_3
  simp only [e_1078_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9147)) := by
  have c66684 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66685 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66686 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66687 := (publicBatchWrapper89_copies2083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66688 := (publicBatchWrapper89_copies2084 a h).1
  have c66689 := (publicBatchWrapper89_copies2084 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_4
  simp only [← c66684, ← c66685, ← c66686] at e_1078_4
  have e_1078_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_5
  simp only [← c66687, ← c66688, k2, ← c66689] at e_1078_5
  have hr := e_1078_5
  simp only [e_1078_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9148)) := by
  have c66690 := (publicBatchWrapper89_copies2084 a h).2.2.1
  have c66691 := (publicBatchWrapper89_copies2084 a h).2.2.2.1
  have c66692 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.1
  have c66693 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.1
  have c66694 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.1
  have c66695 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_6
  simp only [← c66690, ← c66691, ← c66692] at e_1078_6
  have e_1078_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_7
  simp only [← c66693, ← c66694, k2, ← c66695] at e_1078_7
  have hr := e_1078_7
  simp only [e_1078_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9149)) := by
  have c66696 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.1
  have c66697 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.1
  have c66698 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66699 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66700 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66701 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_8
  simp only [← c66696, ← c66697, ← c66698] at e_1078_8
  have e_1078_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_9
  simp only [← c66699, ← c66700, k2, ← c66701] at e_1078_9
  have hr := e_1078_9
  simp only [e_1078_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9150)) := by
  have c66702 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66703 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66704 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66705 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66706 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66707 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_10
  simp only [← c66702, ← c66703, ← c66704] at e_1078_10
  have e_1078_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_11
  simp only [← c66705, ← c66706, k2, ← c66707] at e_1078_11
  have hr := e_1078_11
  simp only [e_1078_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9151)) := by
  have c66708 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66709 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66710 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66711 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66712 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66713 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_12
  simp only [← c66708, ← c66709, ← c66710] at e_1078_12
  have e_1078_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_13
  simp only [← c66711, ← c66712, k2, ← c66713] at e_1078_13
  have hr := e_1078_13
  simp only [e_1078_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9152)) := by
  have c66714 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66715 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66716 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66717 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66718 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66719 := (publicBatchWrapper89_copies2084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_14
  simp only [← c66714, ← c66715, ← c66716] at e_1078_14
  have e_1078_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_15
  simp only [← c66717, ← c66718, k2, ← c66719] at e_1078_15
  have hr := e_1078_15
  simp only [e_1078_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9153)) := by
  have c66720 := (publicBatchWrapper89_copies2085 a h).1
  have c66721 := (publicBatchWrapper89_copies2085 a h).2.1
  have c66722 := (publicBatchWrapper89_copies2085 a h).2.2.1
  have c66723 := (publicBatchWrapper89_copies2085 a h).2.2.2.1
  have c66724 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.1
  have c66725 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_16
  simp only [← c66720, ← c66721, ← c66722] at e_1078_16
  have e_1078_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_17
  simp only [← c66723, ← c66724, k2, ← c66725] at e_1078_17
  have hr := e_1078_17
  simp only [e_1078_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1078 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9154)) := by
  have c66726 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.1
  have c66727 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.1
  have c66728 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.1
  have c66729 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.1
  have c66730 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66731 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1078_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_18
  simp only [← c66726, ← c66727, ← c66728] at e_1078_18
  have e_1078_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1078 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1078_19
  simp only [← c66729, ← c66730, k2, ← c66731] at e_1078_19
  have hr := e_1078_19
  simp only [e_1078_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9155)) := by
  have c66732 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66733 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66734 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66735 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66736 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66737 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_0
  simp only [← c66732, ← c66733, ← c66734] at e_1079_0
  have e_1079_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_1
  simp only [← c66735, ← c66736, k2, ← c66737] at e_1079_1
  have hr := e_1079_1
  simp only [e_1079_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9156)) := by
  have c66738 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66739 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66740 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66741 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66742 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66743 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_2
  simp only [← c66738, ← c66739, ← c66740] at e_1079_2
  have e_1079_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_3
  simp only [← c66741, ← c66742, k2, ← c66743] at e_1079_3
  have hr := e_1079_3
  simp only [e_1079_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9157)) := by
  have c66744 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66745 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66746 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66747 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66748 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66749 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_4
  simp only [← c66744, ← c66745, ← c66746] at e_1079_4
  have e_1079_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_5
  simp only [← c66747, ← c66748, k2, ← c66749] at e_1079_5
  have hr := e_1079_5
  simp only [e_1079_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9158)) := by
  have c66750 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66751 := (publicBatchWrapper89_copies2085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66752 := (publicBatchWrapper89_copies2086 a h).1
  have c66753 := (publicBatchWrapper89_copies2086 a h).2.1
  have c66754 := (publicBatchWrapper89_copies2086 a h).2.2.1
  have c66755 := (publicBatchWrapper89_copies2086 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_6
  simp only [← c66750, ← c66751, ← c66752] at e_1079_6
  have e_1079_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_7
  simp only [← c66753, ← c66754, k2, ← c66755] at e_1079_7
  have hr := e_1079_7
  simp only [e_1079_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9159)) := by
  have c66756 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.1
  have c66757 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.1
  have c66758 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.1
  have c66759 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.1
  have c66760 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.1
  have c66761 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_8
  simp only [← c66756, ← c66757, ← c66758] at e_1079_8
  have e_1079_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_9
  simp only [← c66759, ← c66760, k2, ← c66761] at e_1079_9
  have hr := e_1079_9
  simp only [e_1079_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9160)) := by
  have c66762 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66763 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66764 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66765 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66766 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66767 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_10
  simp only [← c66762, ← c66763, ← c66764] at e_1079_10
  have e_1079_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_11
  simp only [← c66765, ← c66766, k2, ← c66767] at e_1079_11
  have hr := e_1079_11
  simp only [e_1079_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9161)) := by
  have c66768 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66769 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66770 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66771 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66772 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66773 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_12
  simp only [← c66768, ← c66769, ← c66770] at e_1079_12
  have e_1079_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_13
  simp only [← c66771, ← c66772, k2, ← c66773] at e_1079_13
  have hr := e_1079_13
  simp only [e_1079_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9162)) := by
  have c66774 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66775 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66776 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66777 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66778 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66779 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_14
  simp only [← c66774, ← c66775, ← c66776] at e_1079_14
  have e_1079_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_15
  simp only [← c66777, ← c66778, k2, ← c66779] at e_1079_15
  have hr := e_1079_15
  simp only [e_1079_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9163)) := by
  have c66780 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66781 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66782 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66783 := (publicBatchWrapper89_copies2086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66784 := (publicBatchWrapper89_copies2087 a h).1
  have c66785 := (publicBatchWrapper89_copies2087 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_16
  simp only [← c66780, ← c66781, ← c66782] at e_1079_16
  have e_1079_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_17
  simp only [← c66783, ← c66784, k2, ← c66785] at e_1079_17
  have hr := e_1079_17
  simp only [e_1079_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1079 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9164)) := by
  have c66786 := (publicBatchWrapper89_copies2087 a h).2.2.1
  have c66787 := (publicBatchWrapper89_copies2087 a h).2.2.2.1
  have c66788 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.1
  have c66789 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.1
  have c66790 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.1
  have c66791 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1079_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_18
  simp only [← c66786, ← c66787, ← c66788] at e_1079_18
  have e_1079_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1079 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1079_19
  simp only [← c66789, ← c66790, k2, ← c66791] at e_1079_19
  have hr := e_1079_19
  simp only [e_1079_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9165)) := by
  have c66792 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.1
  have c66793 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.1
  have c66794 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66795 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66796 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66797 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_0
  simp only [← c66792, ← c66793, ← c66794] at e_1080_0
  have e_1080_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_1
  simp only [← c66795, ← c66796, k2, ← c66797] at e_1080_1
  have hr := e_1080_1
  simp only [e_1080_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9166)) := by
  have c66798 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66799 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66800 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66801 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66802 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66803 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_2
  simp only [← c66798, ← c66799, ← c66800] at e_1080_2
  have e_1080_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_3
  simp only [← c66801, ← c66802, k2, ← c66803] at e_1080_3
  have hr := e_1080_3
  simp only [e_1080_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9167)) := by
  have c66804 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66805 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66806 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66807 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66808 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66809 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_4
  simp only [← c66804, ← c66805, ← c66806] at e_1080_4
  have e_1080_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_5
  simp only [← c66807, ← c66808, k2, ← c66809] at e_1080_5
  have hr := e_1080_5
  simp only [e_1080_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9168)) := by
  have c66810 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66811 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66812 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66813 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66814 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66815 := (publicBatchWrapper89_copies2087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_6
  simp only [← c66810, ← c66811, ← c66812] at e_1080_6
  have e_1080_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_7
  simp only [← c66813, ← c66814, k2, ← c66815] at e_1080_7
  have hr := e_1080_7
  simp only [e_1080_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9169)) := by
  have c66816 := (publicBatchWrapper89_copies2088 a h).1
  have c66817 := (publicBatchWrapper89_copies2088 a h).2.1
  have c66818 := (publicBatchWrapper89_copies2088 a h).2.2.1
  have c66819 := (publicBatchWrapper89_copies2088 a h).2.2.2.1
  have c66820 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.1
  have c66821 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_8
  simp only [← c66816, ← c66817, ← c66818] at e_1080_8
  have e_1080_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_9
  simp only [← c66819, ← c66820, k2, ← c66821] at e_1080_9
  have hr := e_1080_9
  simp only [e_1080_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9210)) := by
  have c66822 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.1
  have c66823 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.1
  have c66824 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.1
  have c66825 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.1
  have c66826 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66827 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_10
  simp only [← c66822, ← c66823, ← c66824] at e_1080_10
  have e_1080_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_11
  simp only [← c66825, ← c66826, k2, ← c66827] at e_1080_11
  have hr := e_1080_11
  simp only [e_1080_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9211)) := by
  have c66828 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66829 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66830 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66831 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66832 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66833 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_12
  simp only [← c66828, ← c66829, ← c66830] at e_1080_12
  have e_1080_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_13
  simp only [← c66831, ← c66832, k2, ← c66833] at e_1080_13
  have hr := e_1080_13
  simp only [e_1080_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9212)) := by
  have c66834 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66835 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66836 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66837 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66838 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66839 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_14
  simp only [← c66834, ← c66835, ← c66836] at e_1080_14
  have e_1080_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_15
  simp only [← c66837, ← c66838, k2, ← c66839] at e_1080_15
  have hr := e_1080_15
  simp only [e_1080_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9213)) := by
  have c66840 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66841 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66842 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66843 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66844 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66845 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_16
  simp only [← c66840, ← c66841, ← c66842] at e_1080_16
  have e_1080_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_17
  simp only [← c66843, ← c66844, k2, ← c66845] at e_1080_17
  have hr := e_1080_17
  simp only [e_1080_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1080 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9214)) := by
  have c66846 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66847 := (publicBatchWrapper89_copies2088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66848 := (publicBatchWrapper89_copies2089 a h).1
  have c66849 := (publicBatchWrapper89_copies2089 a h).2.1
  have c66850 := (publicBatchWrapper89_copies2089 a h).2.2.1
  have c66851 := (publicBatchWrapper89_copies2089 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1080_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_18
  simp only [← c66846, ← c66847, ← c66848] at e_1080_18
  have e_1080_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1080 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1080_19
  simp only [← c66849, ← c66850, k2, ← c66851] at e_1080_19
  have hr := e_1080_19
  simp only [e_1080_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9215)) := by
  have c66852 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.1
  have c66853 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.1
  have c66854 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.1
  have c66855 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.1
  have c66856 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.1
  have c66857 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_0
  simp only [← c66852, ← c66853, ← c66854] at e_1081_0
  have e_1081_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_1
  simp only [← c66855, ← c66856, k2, ← c66857] at e_1081_1
  have hr := e_1081_1
  simp only [e_1081_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9216)) := by
  have c66858 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66859 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66860 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66861 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66862 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66863 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_2
  simp only [← c66858, ← c66859, ← c66860] at e_1081_2
  have e_1081_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_3
  simp only [← c66861, ← c66862, k2, ← c66863] at e_1081_3
  have hr := e_1081_3
  simp only [e_1081_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9217)) := by
  have c66864 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66865 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66866 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66867 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66868 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66869 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_4
  simp only [← c66864, ← c66865, ← c66866] at e_1081_4
  have e_1081_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_5
  simp only [← c66867, ← c66868, k2, ← c66869] at e_1081_5
  have hr := e_1081_5
  simp only [e_1081_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9218)) := by
  have c66870 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66871 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66872 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66873 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66874 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66875 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_6
  simp only [← c66870, ← c66871, ← c66872] at e_1081_6
  have e_1081_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_7
  simp only [← c66873, ← c66874, k2, ← c66875] at e_1081_7
  have hr := e_1081_7
  simp only [e_1081_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9219)) := by
  have c66876 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66877 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66878 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66879 := (publicBatchWrapper89_copies2089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66880 := (publicBatchWrapper89_copies2090 a h).1
  have c66881 := (publicBatchWrapper89_copies2090 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_8
  simp only [← c66876, ← c66877, ← c66878] at e_1081_8
  have e_1081_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_9
  simp only [← c66879, ← c66880, k2, ← c66881] at e_1081_9
  have hr := e_1081_9
  simp only [e_1081_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9220)) := by
  have c66882 := (publicBatchWrapper89_copies2090 a h).2.2.1
  have c66883 := (publicBatchWrapper89_copies2090 a h).2.2.2.1
  have c66884 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.1
  have c66885 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.1
  have c66886 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.1
  have c66887 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_10
  simp only [← c66882, ← c66883, ← c66884] at e_1081_10
  have e_1081_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_11
  simp only [← c66885, ← c66886, k2, ← c66887] at e_1081_11
  have hr := e_1081_11
  simp only [e_1081_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9221)) := by
  have c66888 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.1
  have c66889 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.1
  have c66890 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66891 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66892 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66893 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_12
  simp only [← c66888, ← c66889, ← c66890] at e_1081_12
  have e_1081_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_13
  simp only [← c66891, ← c66892, k2, ← c66893] at e_1081_13
  have hr := e_1081_13
  simp only [e_1081_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9222)) := by
  have c66894 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66895 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66896 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66897 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66898 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66899 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_14
  simp only [← c66894, ← c66895, ← c66896] at e_1081_14
  have e_1081_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_15
  simp only [← c66897, ← c66898, k2, ← c66899] at e_1081_15
  have hr := e_1081_15
  simp only [e_1081_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9223)) := by
  have c66900 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66901 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66902 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66903 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66904 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66905 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_16
  simp only [← c66900, ← c66901, ← c66902] at e_1081_16
  have e_1081_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_17
  simp only [← c66903, ← c66904, k2, ← c66905] at e_1081_17
  have hr := e_1081_17
  simp only [e_1081_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1081 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9224)) := by
  have c66906 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66907 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66908 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66909 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66910 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66911 := (publicBatchWrapper89_copies2090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1081_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_18
  simp only [← c66906, ← c66907, ← c66908] at e_1081_18
  have e_1081_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1081 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1081_19
  simp only [← c66909, ← c66910, k2, ← c66911] at e_1081_19
  have hr := e_1081_19
  simp only [e_1081_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9225)) := by
  have c66912 := (publicBatchWrapper89_copies2091 a h).1
  have c66913 := (publicBatchWrapper89_copies2091 a h).2.1
  have c66914 := (publicBatchWrapper89_copies2091 a h).2.2.1
  have c66915 := (publicBatchWrapper89_copies2091 a h).2.2.2.1
  have c66916 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.1
  have c66917 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_0
  simp only [← c66912, ← c66913, ← c66914] at e_1082_0
  have e_1082_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_1
  simp only [← c66915, ← c66916, k2, ← c66917] at e_1082_1
  have hr := e_1082_1
  simp only [e_1082_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9226)) := by
  have c66918 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.1
  have c66919 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.1
  have c66920 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.1
  have c66921 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.1
  have c66922 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66923 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_2
  simp only [← c66918, ← c66919, ← c66920] at e_1082_2
  have e_1082_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_3
  simp only [← c66921, ← c66922, k2, ← c66923] at e_1082_3
  have hr := e_1082_3
  simp only [e_1082_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9227)) := by
  have c66924 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66925 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66926 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66927 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66928 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66929 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_4
  simp only [← c66924, ← c66925, ← c66926] at e_1082_4
  have e_1082_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_5
  simp only [← c66927, ← c66928, k2, ← c66929] at e_1082_5
  have hr := e_1082_5
  simp only [e_1082_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9228)) := by
  have c66930 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66931 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66932 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66933 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66934 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66935 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_6
  simp only [← c66930, ← c66931, ← c66932] at e_1082_6
  have e_1082_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_7
  simp only [← c66933, ← c66934, k2, ← c66935] at e_1082_7
  have hr := e_1082_7
  simp only [e_1082_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9229)) := by
  have c66936 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66937 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66938 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66939 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66940 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66941 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_8
  simp only [← c66936, ← c66937, ← c66938] at e_1082_8
  have e_1082_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_9
  simp only [← c66939, ← c66940, k2, ← c66941] at e_1082_9
  have hr := e_1082_9
  simp only [e_1082_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9230)) := by
  have c66942 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66943 := (publicBatchWrapper89_copies2091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66944 := (publicBatchWrapper89_copies2092 a h).1
  have c66945 := (publicBatchWrapper89_copies2092 a h).2.1
  have c66946 := (publicBatchWrapper89_copies2092 a h).2.2.1
  have c66947 := (publicBatchWrapper89_copies2092 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_10
  simp only [← c66942, ← c66943, ← c66944] at e_1082_10
  have e_1082_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_11
  simp only [← c66945, ← c66946, k2, ← c66947] at e_1082_11
  have hr := e_1082_11
  simp only [e_1082_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9231)) := by
  have c66948 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.1
  have c66949 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.1
  have c66950 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.1
  have c66951 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.1
  have c66952 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.1
  have c66953 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_12
  simp only [← c66948, ← c66949, ← c66950] at e_1082_12
  have e_1082_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_13
  simp only [← c66951, ← c66952, k2, ← c66953] at e_1082_13
  have hr := e_1082_13
  simp only [e_1082_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9232)) := by
  have c66954 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66955 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66956 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66957 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66958 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66959 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_14
  simp only [← c66954, ← c66955, ← c66956] at e_1082_14
  have e_1082_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_15
  simp only [← c66957, ← c66958, k2, ← c66959] at e_1082_15
  have hr := e_1082_15
  simp only [e_1082_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9233)) := by
  have c66960 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66961 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66962 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66963 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66964 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66965 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_16
  simp only [← c66960, ← c66961, ← c66962] at e_1082_16
  have e_1082_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_17
  simp only [← c66963, ← c66964, k2, ← c66965] at e_1082_17
  have hr := e_1082_17
  simp only [e_1082_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1082 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9234)) := by
  have c66966 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66967 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66968 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66969 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66970 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66971 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1082_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_18
  simp only [← c66966, ← c66967, ← c66968] at e_1082_18
  have e_1082_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1082 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1082_19
  simp only [← c66969, ← c66970, k2, ← c66971] at e_1082_19
  have hr := e_1082_19
  simp only [e_1082_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9235)) := by
  have c66972 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66973 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66974 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66975 := (publicBatchWrapper89_copies2092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c66976 := (publicBatchWrapper89_copies2093 a h).1
  have c66977 := (publicBatchWrapper89_copies2093 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_0
  simp only [← c66972, ← c66973, ← c66974] at e_1083_0
  have e_1083_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_1
  simp only [← c66975, ← c66976, k2, ← c66977] at e_1083_1
  have hr := e_1083_1
  simp only [e_1083_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9236)) := by
  have c66978 := (publicBatchWrapper89_copies2093 a h).2.2.1
  have c66979 := (publicBatchWrapper89_copies2093 a h).2.2.2.1
  have c66980 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.1
  have c66981 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.1
  have c66982 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.1
  have c66983 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_2
  simp only [← c66978, ← c66979, ← c66980] at e_1083_2
  have e_1083_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_3
  simp only [← c66981, ← c66982, k2, ← c66983] at e_1083_3
  have hr := e_1083_3
  simp only [e_1083_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9237)) := by
  have c66984 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.1
  have c66985 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.1
  have c66986 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.1
  have c66987 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c66988 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66989 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_4
  simp only [← c66984, ← c66985, ← c66986] at e_1083_4
  have e_1083_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_5
  simp only [← c66987, ← c66988, k2, ← c66989] at e_1083_5
  have hr := e_1083_5
  simp only [e_1083_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9238)) := by
  have c66990 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66991 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66992 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66993 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66994 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66995 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_6
  simp only [← c66990, ← c66991, ← c66992] at e_1083_6
  have e_1083_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_7
  simp only [← c66993, ← c66994, k2, ← c66995] at e_1083_7
  have hr := e_1083_7
  simp only [e_1083_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9239)) := by
  have c66996 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66997 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66998 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c66999 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67000 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67001 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_8
  simp only [← c66996, ← c66997, ← c66998] at e_1083_8
  have e_1083_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_9
  simp only [← c66999, ← c67000, k2, ← c67001] at e_1083_9
  have hr := e_1083_9
  simp only [e_1083_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9240)) := by
  have c67002 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67003 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67004 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67005 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67006 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67007 := (publicBatchWrapper89_copies2093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_10
  simp only [← c67002, ← c67003, ← c67004] at e_1083_10
  have e_1083_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_11
  simp only [← c67005, ← c67006, k2, ← c67007] at e_1083_11
  have hr := e_1083_11
  simp only [e_1083_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9241)) := by
  have c67008 := (publicBatchWrapper89_copies2094 a h).1
  have c67009 := (publicBatchWrapper89_copies2094 a h).2.1
  have c67010 := (publicBatchWrapper89_copies2094 a h).2.2.1
  have c67011 := (publicBatchWrapper89_copies2094 a h).2.2.2.1
  have c67012 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.1
  have c67013 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_12
  simp only [← c67008, ← c67009, ← c67010] at e_1083_12
  have e_1083_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_13
  simp only [← c67011, ← c67012, k2, ← c67013] at e_1083_13
  have hr := e_1083_13
  simp only [e_1083_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9242)) := by
  have c67014 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.1
  have c67015 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.1
  have c67016 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.1
  have c67017 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.1
  have c67018 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67019 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_14
  simp only [← c67014, ← c67015, ← c67016] at e_1083_14
  have e_1083_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_15
  simp only [← c67017, ← c67018, k2, ← c67019] at e_1083_15
  have hr := e_1083_15
  simp only [e_1083_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9243)) := by
  have c67020 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67021 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67022 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67023 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67024 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67025 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_16
  simp only [← c67020, ← c67021, ← c67022] at e_1083_16
  have e_1083_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_17
  simp only [← c67023, ← c67024, k2, ← c67025] at e_1083_17
  have hr := e_1083_17
  simp only [e_1083_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1083 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9244)) := by
  have c67026 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67027 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67028 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67029 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67030 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67031 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1083_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_18
  simp only [← c67026, ← c67027, ← c67028] at e_1083_18
  have e_1083_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1083 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1083_19
  simp only [← c67029, ← c67030, k2, ← c67031] at e_1083_19
  have hr := e_1083_19
  simp only [e_1083_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9245)) := by
  have c67032 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67033 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67034 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67035 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67036 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67037 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_0
  simp only [← c67032, ← c67033, ← c67034] at e_1084_0
  have e_1084_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_1
  simp only [← c67035, ← c67036, k2, ← c67037] at e_1084_1
  have hr := e_1084_1
  simp only [e_1084_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9246)) := by
  have c67038 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67039 := (publicBatchWrapper89_copies2094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67040 := (publicBatchWrapper89_copies2095 a h).1
  have c67041 := (publicBatchWrapper89_copies2095 a h).2.1
  have c67042 := (publicBatchWrapper89_copies2095 a h).2.2.1
  have c67043 := (publicBatchWrapper89_copies2095 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_2
  simp only [← c67038, ← c67039, ← c67040] at e_1084_2
  have e_1084_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_3
  simp only [← c67041, ← c67042, k2, ← c67043] at e_1084_3
  have hr := e_1084_3
  simp only [e_1084_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9247)) := by
  have c67044 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.1
  have c67045 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.1
  have c67046 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.1
  have c67047 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.1
  have c67048 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.1
  have c67049 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_4
  simp only [← c67044, ← c67045, ← c67046] at e_1084_4
  have e_1084_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_5
  simp only [← c67047, ← c67048, k2, ← c67049] at e_1084_5
  have hr := e_1084_5
  simp only [e_1084_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9248)) := by
  have c67050 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67051 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67052 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67053 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67054 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67055 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_6
  simp only [← c67050, ← c67051, ← c67052] at e_1084_6
  have e_1084_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_7
  simp only [← c67053, ← c67054, k2, ← c67055] at e_1084_7
  have hr := e_1084_7
  simp only [e_1084_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9249)) := by
  have c67056 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67057 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67058 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67059 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67060 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67061 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_8
  simp only [← c67056, ← c67057, ← c67058] at e_1084_8
  have e_1084_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_9
  simp only [← c67059, ← c67060, k2, ← c67061] at e_1084_9
  have hr := e_1084_9
  simp only [e_1084_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9250)) := by
  have c67062 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67063 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67064 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67065 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67066 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67067 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_10
  simp only [← c67062, ← c67063, ← c67064] at e_1084_10
  have e_1084_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_11
  simp only [← c67065, ← c67066, k2, ← c67067] at e_1084_11
  have hr := e_1084_11
  simp only [e_1084_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9251)) := by
  have c67068 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67069 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67070 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67071 := (publicBatchWrapper89_copies2095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67072 := (publicBatchWrapper89_copies2096 a h).1
  have c67073 := (publicBatchWrapper89_copies2096 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_12
  simp only [← c67068, ← c67069, ← c67070] at e_1084_12
  have e_1084_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_13
  simp only [← c67071, ← c67072, k2, ← c67073] at e_1084_13
  have hr := e_1084_13
  simp only [e_1084_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9252)) := by
  have c67074 := (publicBatchWrapper89_copies2096 a h).2.2.1
  have c67075 := (publicBatchWrapper89_copies2096 a h).2.2.2.1
  have c67076 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.1
  have c67077 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.1
  have c67078 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.1
  have c67079 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_14
  simp only [← c67074, ← c67075, ← c67076] at e_1084_14
  have e_1084_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_15
  simp only [← c67077, ← c67078, k2, ← c67079] at e_1084_15
  have hr := e_1084_15
  simp only [e_1084_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9253)) := by
  have c67080 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.1
  have c67081 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.1
  have c67082 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67083 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67084 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67085 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_16
  simp only [← c67080, ← c67081, ← c67082] at e_1084_16
  have e_1084_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_17
  simp only [← c67083, ← c67084, k2, ← c67085] at e_1084_17
  have hr := e_1084_17
  simp only [e_1084_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1084 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9254)) := by
  have c67086 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67087 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67088 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67089 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67090 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67091 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1084_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_18
  simp only [← c67086, ← c67087, ← c67088] at e_1084_18
  have e_1084_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1084 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1084_19
  simp only [← c67089, ← c67090, k2, ← c67091] at e_1084_19
  have hr := e_1084_19
  simp only [e_1084_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9255)) := by
  have c67092 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67093 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67094 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67095 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67096 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67097 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_0
  simp only [← c67092, ← c67093, ← c67094] at e_1085_0
  have e_1085_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_1
  simp only [← c67095, ← c67096, k2, ← c67097] at e_1085_1
  have hr := e_1085_1
  simp only [e_1085_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9256)) := by
  have c67098 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67099 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67100 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67101 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67102 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67103 := (publicBatchWrapper89_copies2096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_2
  simp only [← c67098, ← c67099, ← c67100] at e_1085_2
  have e_1085_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_3
  simp only [← c67101, ← c67102, k2, ← c67103] at e_1085_3
  have hr := e_1085_3
  simp only [e_1085_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9257)) := by
  have c67104 := (publicBatchWrapper89_copies2097 a h).1
  have c67105 := (publicBatchWrapper89_copies2097 a h).2.1
  have c67106 := (publicBatchWrapper89_copies2097 a h).2.2.1
  have c67107 := (publicBatchWrapper89_copies2097 a h).2.2.2.1
  have c67108 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.1
  have c67109 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_4
  simp only [← c67104, ← c67105, ← c67106] at e_1085_4
  have e_1085_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_5
  simp only [← c67107, ← c67108, k2, ← c67109] at e_1085_5
  have hr := e_1085_5
  simp only [e_1085_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9258)) := by
  have c67110 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.1
  have c67111 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.1
  have c67112 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.1
  have c67113 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.1
  have c67114 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67115 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_6
  simp only [← c67110, ← c67111, ← c67112] at e_1085_6
  have e_1085_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_7
  simp only [← c67113, ← c67114, k2, ← c67115] at e_1085_7
  have hr := e_1085_7
  simp only [e_1085_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9259)) := by
  have c67116 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67117 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67118 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67119 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67120 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67121 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_8
  simp only [← c67116, ← c67117, ← c67118] at e_1085_8
  have e_1085_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_9
  simp only [← c67119, ← c67120, k2, ← c67121] at e_1085_9
  have hr := e_1085_9
  simp only [e_1085_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9260)) := by
  have c67122 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67123 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67124 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67125 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67126 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67127 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_10
  simp only [← c67122, ← c67123, ← c67124] at e_1085_10
  have e_1085_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_11
  simp only [← c67125, ← c67126, k2, ← c67127] at e_1085_11
  have hr := e_1085_11
  simp only [e_1085_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9261)) := by
  have c67128 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67129 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67130 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67131 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67132 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67133 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_12
  simp only [← c67128, ← c67129, ← c67130] at e_1085_12
  have e_1085_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_13
  simp only [← c67131, ← c67132, k2, ← c67133] at e_1085_13
  have hr := e_1085_13
  simp only [e_1085_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9262)) := by
  have c67134 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67135 := (publicBatchWrapper89_copies2097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67136 := (publicBatchWrapper89_copies2098 a h).1
  have c67137 := (publicBatchWrapper89_copies2098 a h).2.1
  have c67138 := (publicBatchWrapper89_copies2098 a h).2.2.1
  have c67139 := (publicBatchWrapper89_copies2098 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_14
  simp only [← c67134, ← c67135, ← c67136] at e_1085_14
  have e_1085_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_15
  simp only [← c67137, ← c67138, k2, ← c67139] at e_1085_15
  have hr := e_1085_15
  simp only [e_1085_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9263)) := by
  have c67140 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.1
  have c67141 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.1
  have c67142 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.1
  have c67143 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.1
  have c67144 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.1
  have c67145 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_16
  simp only [← c67140, ← c67141, ← c67142] at e_1085_16
  have e_1085_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_17
  simp only [← c67143, ← c67144, k2, ← c67145] at e_1085_17
  have hr := e_1085_17
  simp only [e_1085_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1085 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9264)) := by
  have c67146 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67147 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67148 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67149 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67150 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67151 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1085_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_18
  simp only [← c67146, ← c67147, ← c67148] at e_1085_18
  have e_1085_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1085 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1085_19
  simp only [← c67149, ← c67150, k2, ← c67151] at e_1085_19
  have hr := e_1085_19
  simp only [e_1085_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9265)) := by
  have c67152 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67153 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67154 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67155 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67156 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67157 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_0
  simp only [← c67152, ← c67153, ← c67154] at e_1086_0
  have e_1086_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_1
  simp only [← c67155, ← c67156, k2, ← c67157] at e_1086_1
  have hr := e_1086_1
  simp only [e_1086_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9266)) := by
  have c67158 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67159 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67160 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67161 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67162 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67163 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_2
  simp only [← c67158, ← c67159, ← c67160] at e_1086_2
  have e_1086_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_3
  simp only [← c67161, ← c67162, k2, ← c67163] at e_1086_3
  have hr := e_1086_3
  simp only [e_1086_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9267)) := by
  have c67164 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67165 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67166 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67167 := (publicBatchWrapper89_copies2098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67168 := (publicBatchWrapper89_copies2099 a h).1
  have c67169 := (publicBatchWrapper89_copies2099 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_4
  simp only [← c67164, ← c67165, ← c67166] at e_1086_4
  have e_1086_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_5
  simp only [← c67167, ← c67168, k2, ← c67169] at e_1086_5
  have hr := e_1086_5
  simp only [e_1086_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9268)) := by
  have c67170 := (publicBatchWrapper89_copies2099 a h).2.2.1
  have c67171 := (publicBatchWrapper89_copies2099 a h).2.2.2.1
  have c67172 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.1
  have c67173 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.1
  have c67174 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.1
  have c67175 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_6
  simp only [← c67170, ← c67171, ← c67172] at e_1086_6
  have e_1086_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_7
  simp only [← c67173, ← c67174, k2, ← c67175] at e_1086_7
  have hr := e_1086_7
  simp only [e_1086_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9269)) := by
  have c67176 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.1
  have c67177 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.1
  have c67178 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67179 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67180 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67181 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_8
  simp only [← c67176, ← c67177, ← c67178] at e_1086_8
  have e_1086_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_9
  simp only [← c67179, ← c67180, k2, ← c67181] at e_1086_9
  have hr := e_1086_9
  simp only [e_1086_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9270)) := by
  have c67182 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67183 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67184 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67185 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67186 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67187 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_10
  simp only [← c67182, ← c67183, ← c67184] at e_1086_10
  have e_1086_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_11
  simp only [← c67185, ← c67186, k2, ← c67187] at e_1086_11
  have hr := e_1086_11
  simp only [e_1086_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9271)) := by
  have c67188 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67189 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67190 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67191 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67192 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67193 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_12
  simp only [← c67188, ← c67189, ← c67190] at e_1086_12
  have e_1086_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_13
  simp only [← c67191, ← c67192, k2, ← c67193] at e_1086_13
  have hr := e_1086_13
  simp only [e_1086_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9272)) := by
  have c67194 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67195 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67196 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67197 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67198 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67199 := (publicBatchWrapper89_copies2099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_14
  simp only [← c67194, ← c67195, ← c67196] at e_1086_14
  have e_1086_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_15
  simp only [← c67197, ← c67198, k2, ← c67199] at e_1086_15
  have hr := e_1086_15
  simp only [e_1086_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9273)) := by
  have c67200 := (publicBatchWrapper89_copies2100 a h).1
  have c67201 := (publicBatchWrapper89_copies2100 a h).2.1
  have c67202 := (publicBatchWrapper89_copies2100 a h).2.2.1
  have c67203 := (publicBatchWrapper89_copies2100 a h).2.2.2.1
  have c67204 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.1
  have c67205 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_16
  simp only [← c67200, ← c67201, ← c67202] at e_1086_16
  have e_1086_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_17
  simp only [← c67203, ← c67204, k2, ← c67205] at e_1086_17
  have hr := e_1086_17
  simp only [e_1086_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1086 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9274)) := by
  have c67206 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.1
  have c67207 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.1
  have c67208 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.1
  have c67209 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.1
  have c67210 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67211 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1086_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_18
  simp only [← c67206, ← c67207, ← c67208] at e_1086_18
  have e_1086_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1086 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1086_19
  simp only [← c67209, ← c67210, k2, ← c67211] at e_1086_19
  have hr := e_1086_19
  simp only [e_1086_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9275)) := by
  have c67212 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67213 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67214 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67215 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67216 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67217 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_0
  simp only [← c67212, ← c67213, ← c67214] at e_1087_0
  have e_1087_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_1
  simp only [← c67215, ← c67216, k2, ← c67217] at e_1087_1
  have hr := e_1087_1
  simp only [e_1087_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9276)) := by
  have c67218 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67219 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67220 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67221 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67222 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67223 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_2
  simp only [← c67218, ← c67219, ← c67220] at e_1087_2
  have e_1087_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_3
  simp only [← c67221, ← c67222, k2, ← c67223] at e_1087_3
  have hr := e_1087_3
  simp only [e_1087_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9277)) := by
  have c67224 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67225 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67226 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67227 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67228 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67229 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_4
  simp only [← c67224, ← c67225, ← c67226] at e_1087_4
  have e_1087_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_5
  simp only [← c67227, ← c67228, k2, ← c67229] at e_1087_5
  have hr := e_1087_5
  simp only [e_1087_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9278)) := by
  have c67230 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67231 := (publicBatchWrapper89_copies2100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67232 := (publicBatchWrapper89_copies2101 a h).1
  have c67233 := (publicBatchWrapper89_copies2101 a h).2.1
  have c67234 := (publicBatchWrapper89_copies2101 a h).2.2.1
  have c67235 := (publicBatchWrapper89_copies2101 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_6
  simp only [← c67230, ← c67231, ← c67232] at e_1087_6
  have e_1087_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_7
  simp only [← c67233, ← c67234, k2, ← c67235] at e_1087_7
  have hr := e_1087_7
  simp only [e_1087_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9279)) := by
  have c67236 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.1
  have c67237 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.1
  have c67238 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.1
  have c67239 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.1
  have c67240 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.1
  have c67241 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_8
  simp only [← c67236, ← c67237, ← c67238] at e_1087_8
  have e_1087_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_9
  simp only [← c67239, ← c67240, k2, ← c67241] at e_1087_9
  have hr := e_1087_9
  simp only [e_1087_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9280)) := by
  have c67242 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67243 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67244 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67245 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67246 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67247 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_10
  simp only [← c67242, ← c67243, ← c67244] at e_1087_10
  have e_1087_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_11
  simp only [← c67245, ← c67246, k2, ← c67247] at e_1087_11
  have hr := e_1087_11
  simp only [e_1087_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9281)) := by
  have c67248 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67249 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67250 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67251 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67252 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67253 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_12
  simp only [← c67248, ← c67249, ← c67250] at e_1087_12
  have e_1087_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_13
  simp only [← c67251, ← c67252, k2, ← c67253] at e_1087_13
  have hr := e_1087_13
  simp only [e_1087_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9282)) := by
  have c67254 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67255 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67256 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67257 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67258 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67259 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_14
  simp only [← c67254, ← c67255, ← c67256] at e_1087_14
  have e_1087_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_15
  simp only [← c67257, ← c67258, k2, ← c67259] at e_1087_15
  have hr := e_1087_15
  simp only [e_1087_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9283)) := by
  have c67260 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67261 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67262 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67263 := (publicBatchWrapper89_copies2101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67264 := (publicBatchWrapper89_copies2102 a h).1
  have c67265 := (publicBatchWrapper89_copies2102 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_16
  simp only [← c67260, ← c67261, ← c67262] at e_1087_16
  have e_1087_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_17
  simp only [← c67263, ← c67264, k2, ← c67265] at e_1087_17
  have hr := e_1087_17
  simp only [e_1087_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1087 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9284)) := by
  have c67266 := (publicBatchWrapper89_copies2102 a h).2.2.1
  have c67267 := (publicBatchWrapper89_copies2102 a h).2.2.2.1
  have c67268 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.1
  have c67269 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.1
  have c67270 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.1
  have c67271 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1087_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_18
  simp only [← c67266, ← c67267, ← c67268] at e_1087_18
  have e_1087_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1087 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1087_19
  simp only [← c67269, ← c67270, k2, ← c67271] at e_1087_19
  have hr := e_1087_19
  simp only [e_1087_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9285)) := by
  have c67272 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.1
  have c67273 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.1
  have c67274 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67275 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67276 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67277 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_0
  simp only [← c67272, ← c67273, ← c67274] at e_1088_0
  have e_1088_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_1
  simp only [← c67275, ← c67276, k2, ← c67277] at e_1088_1
  have hr := e_1088_1
  simp only [e_1088_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9286)) := by
  have c67278 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67279 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67280 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67281 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67282 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67283 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_2
  simp only [← c67278, ← c67279, ← c67280] at e_1088_2
  have e_1088_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_3
  simp only [← c67281, ← c67282, k2, ← c67283] at e_1088_3
  have hr := e_1088_3
  simp only [e_1088_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9287)) := by
  have c67284 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67285 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67286 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67287 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67288 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67289 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_4
  simp only [← c67284, ← c67285, ← c67286] at e_1088_4
  have e_1088_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_5
  simp only [← c67287, ← c67288, k2, ← c67289] at e_1088_5
  have hr := e_1088_5
  simp only [e_1088_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9288)) := by
  have c67290 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67291 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67292 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67293 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67294 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67295 := (publicBatchWrapper89_copies2102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_6
  simp only [← c67290, ← c67291, ← c67292] at e_1088_6
  have e_1088_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_7
  simp only [← c67293, ← c67294, k2, ← c67295] at e_1088_7
  have hr := e_1088_7
  simp only [e_1088_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9289)) := by
  have c67296 := (publicBatchWrapper89_copies2103 a h).1
  have c67297 := (publicBatchWrapper89_copies2103 a h).2.1
  have c67298 := (publicBatchWrapper89_copies2103 a h).2.2.1
  have c67299 := (publicBatchWrapper89_copies2103 a h).2.2.2.1
  have c67300 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.1
  have c67301 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_8
  simp only [← c67296, ← c67297, ← c67298] at e_1088_8
  have e_1088_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_9
  simp only [← c67299, ← c67300, k2, ← c67301] at e_1088_9
  have hr := e_1088_9
  simp only [e_1088_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9290)) := by
  have c67302 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.1
  have c67303 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.1
  have c67304 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.1
  have c67305 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.1
  have c67306 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67307 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_10
  simp only [← c67302, ← c67303, ← c67304] at e_1088_10
  have e_1088_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_11
  simp only [← c67305, ← c67306, k2, ← c67307] at e_1088_11
  have hr := e_1088_11
  simp only [e_1088_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9291)) := by
  have c67308 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67309 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67310 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67311 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67312 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67313 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_12
  simp only [← c67308, ← c67309, ← c67310] at e_1088_12
  have e_1088_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_13
  simp only [← c67311, ← c67312, k2, ← c67313] at e_1088_13
  have hr := e_1088_13
  simp only [e_1088_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9292)) := by
  have c67314 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67315 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67316 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67317 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67318 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67319 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_14
  simp only [← c67314, ← c67315, ← c67316] at e_1088_14
  have e_1088_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_15
  simp only [← c67317, ← c67318, k2, ← c67319] at e_1088_15
  have hr := e_1088_15
  simp only [e_1088_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9293)) := by
  have c67320 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67321 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67322 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67323 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67324 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67325 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_16
  simp only [← c67320, ← c67321, ← c67322] at e_1088_16
  have e_1088_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_17
  simp only [← c67323, ← c67324, k2, ← c67325] at e_1088_17
  have hr := e_1088_17
  simp only [e_1088_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1088 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9294)) := by
  have c67326 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67327 := (publicBatchWrapper89_copies2103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67328 := (publicBatchWrapper89_copies2104 a h).1
  have c67329 := (publicBatchWrapper89_copies2104 a h).2.1
  have c67330 := (publicBatchWrapper89_copies2104 a h).2.2.1
  have c67331 := (publicBatchWrapper89_copies2104 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1088_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_18
  simp only [← c67326, ← c67327, ← c67328] at e_1088_18
  have e_1088_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1088 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1088_19
  simp only [← c67329, ← c67330, k2, ← c67331] at e_1088_19
  have hr := e_1088_19
  simp only [e_1088_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9295)) := by
  have c67332 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.1
  have c67333 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.1
  have c67334 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.1
  have c67335 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.1
  have c67336 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.1
  have c67337 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_0
  simp only [← c67332, ← c67333, ← c67334] at e_1089_0
  have e_1089_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_1
  simp only [← c67335, ← c67336, k2, ← c67337] at e_1089_1
  have hr := e_1089_1
  simp only [e_1089_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9296)) := by
  have c67338 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67339 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67340 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67341 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67342 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67343 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_2
  simp only [← c67338, ← c67339, ← c67340] at e_1089_2
  have e_1089_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_3
  simp only [← c67341, ← c67342, k2, ← c67343] at e_1089_3
  have hr := e_1089_3
  simp only [e_1089_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9297)) := by
  have c67344 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67345 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67346 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67347 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67348 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67349 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_4
  simp only [← c67344, ← c67345, ← c67346] at e_1089_4
  have e_1089_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_5
  simp only [← c67347, ← c67348, k2, ← c67349] at e_1089_5
  have hr := e_1089_5
  simp only [e_1089_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9298)) := by
  have c67350 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67351 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67352 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67353 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67354 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67355 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_6
  simp only [← c67350, ← c67351, ← c67352] at e_1089_6
  have e_1089_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_7
  simp only [← c67353, ← c67354, k2, ← c67355] at e_1089_7
  have hr := e_1089_7
  simp only [e_1089_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9299)) := by
  have c67356 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67357 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67358 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67359 := (publicBatchWrapper89_copies2104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67360 := (publicBatchWrapper89_copies2105 a h).1
  have c67361 := (publicBatchWrapper89_copies2105 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_8
  simp only [← c67356, ← c67357, ← c67358] at e_1089_8
  have e_1089_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_9
  simp only [← c67359, ← c67360, k2, ← c67361] at e_1089_9
  have hr := e_1089_9
  simp only [e_1089_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9300)) := by
  have c67362 := (publicBatchWrapper89_copies2105 a h).2.2.1
  have c67363 := (publicBatchWrapper89_copies2105 a h).2.2.2.1
  have c67364 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.1
  have c67365 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.1
  have c67366 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.1
  have c67367 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_10
  simp only [← c67362, ← c67363, ← c67364] at e_1089_10
  have e_1089_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_11
  simp only [← c67365, ← c67366, k2, ← c67367] at e_1089_11
  have hr := e_1089_11
  simp only [e_1089_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9301)) := by
  have c67368 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.1
  have c67369 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.1
  have c67370 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67371 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67372 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67373 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_12
  simp only [← c67368, ← c67369, ← c67370] at e_1089_12
  have e_1089_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_13
  simp only [← c67371, ← c67372, k2, ← c67373] at e_1089_13
  have hr := e_1089_13
  simp only [e_1089_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9302)) := by
  have c67374 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67375 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67376 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67377 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67378 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67379 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_14
  simp only [← c67374, ← c67375, ← c67376] at e_1089_14
  have e_1089_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_15
  simp only [← c67377, ← c67378, k2, ← c67379] at e_1089_15
  have hr := e_1089_15
  simp only [e_1089_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9303)) := by
  have c67380 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67381 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67382 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67383 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67384 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67385 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_16
  simp only [← c67380, ← c67381, ← c67382] at e_1089_16
  have e_1089_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_17
  simp only [← c67383, ← c67384, k2, ← c67385] at e_1089_17
  have hr := e_1089_17
  simp only [e_1089_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1089 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9304)) := by
  have c67386 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67387 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67388 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67389 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67390 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67391 := (publicBatchWrapper89_copies2105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1089_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_18
  simp only [← c67386, ← c67387, ← c67388] at e_1089_18
  have e_1089_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1089 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1089_19
  simp only [← c67389, ← c67390, k2, ← c67391] at e_1089_19
  have hr := e_1089_19
  simp only [e_1089_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9305)) := by
  have c67392 := (publicBatchWrapper89_copies2106 a h).1
  have c67393 := (publicBatchWrapper89_copies2106 a h).2.1
  have c67394 := (publicBatchWrapper89_copies2106 a h).2.2.1
  have c67395 := (publicBatchWrapper89_copies2106 a h).2.2.2.1
  have c67396 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.1
  have c67397 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_0
  simp only [← c67392, ← c67393, ← c67394] at e_1090_0
  have e_1090_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_1
  simp only [← c67395, ← c67396, k2, ← c67397] at e_1090_1
  have hr := e_1090_1
  simp only [e_1090_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9306)) := by
  have c67398 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.1
  have c67399 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.1
  have c67400 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.1
  have c67401 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.1
  have c67402 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67403 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_2
  simp only [← c67398, ← c67399, ← c67400] at e_1090_2
  have e_1090_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_3
  simp only [← c67401, ← c67402, k2, ← c67403] at e_1090_3
  have hr := e_1090_3
  simp only [e_1090_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9307)) := by
  have c67404 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67405 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67406 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67407 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67408 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67409 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_4
  simp only [← c67404, ← c67405, ← c67406] at e_1090_4
  have e_1090_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_5
  simp only [← c67407, ← c67408, k2, ← c67409] at e_1090_5
  have hr := e_1090_5
  simp only [e_1090_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9308)) := by
  have c67410 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67411 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67412 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67413 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67414 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67415 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_6
  simp only [← c67410, ← c67411, ← c67412] at e_1090_6
  have e_1090_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_7
  simp only [← c67413, ← c67414, k2, ← c67415] at e_1090_7
  have hr := e_1090_7
  simp only [e_1090_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9309)) := by
  have c67416 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67417 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67418 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67419 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67420 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67421 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_8
  simp only [← c67416, ← c67417, ← c67418] at e_1090_8
  have e_1090_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_9
  simp only [← c67419, ← c67420, k2, ← c67421] at e_1090_9
  have hr := e_1090_9
  simp only [e_1090_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9310)) := by
  have c67422 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67423 := (publicBatchWrapper89_copies2106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67424 := (publicBatchWrapper89_copies2107 a h).1
  have c67425 := (publicBatchWrapper89_copies2107 a h).2.1
  have c67426 := (publicBatchWrapper89_copies2107 a h).2.2.1
  have c67427 := (publicBatchWrapper89_copies2107 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_10
  simp only [← c67422, ← c67423, ← c67424] at e_1090_10
  have e_1090_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_11
  simp only [← c67425, ← c67426, k2, ← c67427] at e_1090_11
  have hr := e_1090_11
  simp only [e_1090_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9311)) := by
  have c67428 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.1
  have c67429 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.1
  have c67430 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.1
  have c67431 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.1
  have c67432 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.1
  have c67433 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_12
  simp only [← c67428, ← c67429, ← c67430] at e_1090_12
  have e_1090_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_13
  simp only [← c67431, ← c67432, k2, ← c67433] at e_1090_13
  have hr := e_1090_13
  simp only [e_1090_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9312)) := by
  have c67434 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67435 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67436 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67437 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67438 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67439 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_14
  simp only [← c67434, ← c67435, ← c67436] at e_1090_14
  have e_1090_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_15
  simp only [← c67437, ← c67438, k2, ← c67439] at e_1090_15
  have hr := e_1090_15
  simp only [e_1090_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9313)) := by
  have c67440 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67441 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67442 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67443 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67444 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67445 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_16
  simp only [← c67440, ← c67441, ← c67442] at e_1090_16
  have e_1090_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_17
  simp only [← c67443, ← c67444, k2, ← c67445] at e_1090_17
  have hr := e_1090_17
  simp only [e_1090_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1090 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9314)) := by
  have c67446 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67447 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67448 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67449 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67450 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67451 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1090_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_18
  simp only [← c67446, ← c67447, ← c67448] at e_1090_18
  have e_1090_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1090 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1090_19
  simp only [← c67449, ← c67450, k2, ← c67451] at e_1090_19
  have hr := e_1090_19
  simp only [e_1090_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9315)) := by
  have c67452 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67453 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67454 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67455 := (publicBatchWrapper89_copies2107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67456 := (publicBatchWrapper89_copies2108 a h).1
  have c67457 := (publicBatchWrapper89_copies2108 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_0
  simp only [← c67452, ← c67453, ← c67454] at e_1091_0
  have e_1091_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_1
  simp only [← c67455, ← c67456, k2, ← c67457] at e_1091_1
  have hr := e_1091_1
  simp only [e_1091_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9316)) := by
  have c67458 := (publicBatchWrapper89_copies2108 a h).2.2.1
  have c67459 := (publicBatchWrapper89_copies2108 a h).2.2.2.1
  have c67460 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.1
  have c67461 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.1
  have c67462 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.1
  have c67463 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_2
  simp only [← c67458, ← c67459, ← c67460] at e_1091_2
  have e_1091_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_3
  simp only [← c67461, ← c67462, k2, ← c67463] at e_1091_3
  have hr := e_1091_3
  simp only [e_1091_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9317)) := by
  have c67464 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.1
  have c67465 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.1
  have c67466 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67467 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67468 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67469 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_4
  simp only [← c67464, ← c67465, ← c67466] at e_1091_4
  have e_1091_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_5
  simp only [← c67467, ← c67468, k2, ← c67469] at e_1091_5
  have hr := e_1091_5
  simp only [e_1091_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9318)) := by
  have c67470 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67471 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67472 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67473 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67474 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67475 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_6
  simp only [← c67470, ← c67471, ← c67472] at e_1091_6
  have e_1091_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_7
  simp only [← c67473, ← c67474, k2, ← c67475] at e_1091_7
  have hr := e_1091_7
  simp only [e_1091_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9319)) := by
  have c67476 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67477 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67478 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67479 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67480 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67481 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_8
  simp only [← c67476, ← c67477, ← c67478] at e_1091_8
  have e_1091_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_9
  simp only [← c67479, ← c67480, k2, ← c67481] at e_1091_9
  have hr := e_1091_9
  simp only [e_1091_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9320)) := by
  have c67482 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67483 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67484 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67485 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67486 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67487 := (publicBatchWrapper89_copies2108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_10
  simp only [← c67482, ← c67483, ← c67484] at e_1091_10
  have e_1091_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_11
  simp only [← c67485, ← c67486, k2, ← c67487] at e_1091_11
  have hr := e_1091_11
  simp only [e_1091_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9321)) := by
  have c67488 := (publicBatchWrapper89_copies2109 a h).1
  have c67489 := (publicBatchWrapper89_copies2109 a h).2.1
  have c67490 := (publicBatchWrapper89_copies2109 a h).2.2.1
  have c67491 := (publicBatchWrapper89_copies2109 a h).2.2.2.1
  have c67492 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.1
  have c67493 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_12
  simp only [← c67488, ← c67489, ← c67490] at e_1091_12
  have e_1091_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_13
  simp only [← c67491, ← c67492, k2, ← c67493] at e_1091_13
  have hr := e_1091_13
  simp only [e_1091_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9322)) := by
  have c67494 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.1
  have c67495 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.1
  have c67496 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.1
  have c67497 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.1
  have c67498 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67499 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_14
  simp only [← c67494, ← c67495, ← c67496] at e_1091_14
  have e_1091_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_15
  simp only [← c67497, ← c67498, k2, ← c67499] at e_1091_15
  have hr := e_1091_15
  simp only [e_1091_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9323)) := by
  have c67500 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67501 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67502 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67503 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67504 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67505 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_16
  simp only [← c67500, ← c67501, ← c67502] at e_1091_16
  have e_1091_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_17
  simp only [← c67503, ← c67504, k2, ← c67505] at e_1091_17
  have hr := e_1091_17
  simp only [e_1091_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1091 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9324)) := by
  have c67506 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67507 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67508 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67509 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67510 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67511 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1091_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_18
  simp only [← c67506, ← c67507, ← c67508] at e_1091_18
  have e_1091_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1091 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1091_19
  simp only [← c67509, ← c67510, k2, ← c67511] at e_1091_19
  have hr := e_1091_19
  simp only [e_1091_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9325)) := by
  have c67512 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67513 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67514 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67515 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67516 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67517 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_0
  simp only [← c67512, ← c67513, ← c67514] at e_1092_0
  have e_1092_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_1
  simp only [← c67515, ← c67516, k2, ← c67517] at e_1092_1
  have hr := e_1092_1
  simp only [e_1092_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9326)) := by
  have c67518 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67519 := (publicBatchWrapper89_copies2109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67520 := (publicBatchWrapper89_copies2110 a h).1
  have c67521 := (publicBatchWrapper89_copies2110 a h).2.1
  have c67522 := (publicBatchWrapper89_copies2110 a h).2.2.1
  have c67523 := (publicBatchWrapper89_copies2110 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_2
  simp only [← c67518, ← c67519, ← c67520] at e_1092_2
  have e_1092_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_3
  simp only [← c67521, ← c67522, k2, ← c67523] at e_1092_3
  have hr := e_1092_3
  simp only [e_1092_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9327)) := by
  have c67524 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.1
  have c67525 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.1
  have c67526 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.1
  have c67527 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.1
  have c67528 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.1
  have c67529 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_4
  simp only [← c67524, ← c67525, ← c67526] at e_1092_4
  have e_1092_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_5
  simp only [← c67527, ← c67528, k2, ← c67529] at e_1092_5
  have hr := e_1092_5
  simp only [e_1092_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9328)) := by
  have c67530 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67531 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67532 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67533 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67534 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67535 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_6
  simp only [← c67530, ← c67531, ← c67532] at e_1092_6
  have e_1092_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_7
  simp only [← c67533, ← c67534, k2, ← c67535] at e_1092_7
  have hr := e_1092_7
  simp only [e_1092_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9329)) := by
  have c67536 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67537 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67538 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67539 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67540 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67541 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_8
  simp only [← c67536, ← c67537, ← c67538] at e_1092_8
  have e_1092_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_9
  simp only [← c67539, ← c67540, k2, ← c67541] at e_1092_9
  have hr := e_1092_9
  simp only [e_1092_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9330)) := by
  have c67542 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67543 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67544 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67545 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67546 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67547 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_10
  simp only [← c67542, ← c67543, ← c67544] at e_1092_10
  have e_1092_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_11
  simp only [← c67545, ← c67546, k2, ← c67547] at e_1092_11
  have hr := e_1092_11
  simp only [e_1092_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9331)) := by
  have c67548 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67549 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67550 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67551 := (publicBatchWrapper89_copies2110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67552 := (publicBatchWrapper89_copies2111 a h).1
  have c67553 := (publicBatchWrapper89_copies2111 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_12
  simp only [← c67548, ← c67549, ← c67550] at e_1092_12
  have e_1092_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_13
  simp only [← c67551, ← c67552, k2, ← c67553] at e_1092_13
  have hr := e_1092_13
  simp only [e_1092_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9332)) := by
  have c67554 := (publicBatchWrapper89_copies2111 a h).2.2.1
  have c67555 := (publicBatchWrapper89_copies2111 a h).2.2.2.1
  have c67556 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.1
  have c67557 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.1
  have c67558 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.1
  have c67559 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_14
  simp only [← c67554, ← c67555, ← c67556] at e_1092_14
  have e_1092_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_15
  simp only [← c67557, ← c67558, k2, ← c67559] at e_1092_15
  have hr := e_1092_15
  simp only [e_1092_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9333)) := by
  have c67560 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.1
  have c67561 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.1
  have c67562 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67563 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67564 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67565 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_16
  simp only [← c67560, ← c67561, ← c67562] at e_1092_16
  have e_1092_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_17
  simp only [← c67563, ← c67564, k2, ← c67565] at e_1092_17
  have hr := e_1092_17
  simp only [e_1092_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1092 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9334)) := by
  have c67566 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67567 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67568 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67569 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67570 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67571 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1092_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_18
  simp only [← c67566, ← c67567, ← c67568] at e_1092_18
  have e_1092_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1092 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1092_19
  simp only [← c67569, ← c67570, k2, ← c67571] at e_1092_19
  have hr := e_1092_19
  simp only [e_1092_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9335)) := by
  have c67572 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67573 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67574 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67575 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67576 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67577 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_0
  simp only [← c67572, ← c67573, ← c67574] at e_1093_0
  have e_1093_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_1
  simp only [← c67575, ← c67576, k2, ← c67577] at e_1093_1
  have hr := e_1093_1
  simp only [e_1093_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9336)) := by
  have c67578 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67579 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67580 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67581 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67582 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67583 := (publicBatchWrapper89_copies2111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_2
  simp only [← c67578, ← c67579, ← c67580] at e_1093_2
  have e_1093_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_3
  simp only [← c67581, ← c67582, k2, ← c67583] at e_1093_3
  have hr := e_1093_3
  simp only [e_1093_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9337)) := by
  have c67584 := (publicBatchWrapper89_copies2112 a h).1
  have c67585 := (publicBatchWrapper89_copies2112 a h).2.1
  have c67586 := (publicBatchWrapper89_copies2112 a h).2.2.1
  have c67587 := (publicBatchWrapper89_copies2112 a h).2.2.2.1
  have c67588 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.1
  have c67589 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_4
  simp only [← c67584, ← c67585, ← c67586] at e_1093_4
  have e_1093_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_5
  simp only [← c67587, ← c67588, k2, ← c67589] at e_1093_5
  have hr := e_1093_5
  simp only [e_1093_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9338)) := by
  have c67590 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.1
  have c67591 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.1
  have c67592 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.1
  have c67593 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.1
  have c67594 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67595 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_6
  simp only [← c67590, ← c67591, ← c67592] at e_1093_6
  have e_1093_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_7
  simp only [← c67593, ← c67594, k2, ← c67595] at e_1093_7
  have hr := e_1093_7
  simp only [e_1093_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9339)) := by
  have c67596 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67597 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67598 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67599 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67600 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67601 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_8
  simp only [← c67596, ← c67597, ← c67598] at e_1093_8
  have e_1093_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_9
  simp only [← c67599, ← c67600, k2, ← c67601] at e_1093_9
  have hr := e_1093_9
  simp only [e_1093_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9340)) := by
  have c67602 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67603 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67604 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67605 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67606 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67607 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_10
  simp only [← c67602, ← c67603, ← c67604] at e_1093_10
  have e_1093_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_11
  simp only [← c67605, ← c67606, k2, ← c67607] at e_1093_11
  have hr := e_1093_11
  simp only [e_1093_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9341)) := by
  have c67608 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67609 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67610 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67611 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67612 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67613 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_12
  simp only [← c67608, ← c67609, ← c67610] at e_1093_12
  have e_1093_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_13
  simp only [← c67611, ← c67612, k2, ← c67613] at e_1093_13
  have hr := e_1093_13
  simp only [e_1093_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9342)) := by
  have c67614 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67615 := (publicBatchWrapper89_copies2112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67616 := (publicBatchWrapper89_copies2113 a h).1
  have c67617 := (publicBatchWrapper89_copies2113 a h).2.1
  have c67618 := (publicBatchWrapper89_copies2113 a h).2.2.1
  have c67619 := (publicBatchWrapper89_copies2113 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_14
  simp only [← c67614, ← c67615, ← c67616] at e_1093_14
  have e_1093_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_15
  simp only [← c67617, ← c67618, k2, ← c67619] at e_1093_15
  have hr := e_1093_15
  simp only [e_1093_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9343)) := by
  have c67620 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.1
  have c67621 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.1
  have c67622 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.1
  have c67623 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.1
  have c67624 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.1
  have c67625 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_16
  simp only [← c67620, ← c67621, ← c67622] at e_1093_16
  have e_1093_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_17
  simp only [← c67623, ← c67624, k2, ← c67625] at e_1093_17
  have hr := e_1093_17
  simp only [e_1093_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1093 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9344)) := by
  have c67626 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67627 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67628 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67629 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67630 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67631 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1093_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_18
  simp only [← c67626, ← c67627, ← c67628] at e_1093_18
  have e_1093_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1093 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1093_19
  simp only [← c67629, ← c67630, k2, ← c67631] at e_1093_19
  have hr := e_1093_19
  simp only [e_1093_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9345)) := by
  have c67632 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67633 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67634 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67635 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67636 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67637 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_0
  simp only [← c67632, ← c67633, ← c67634] at e_1094_0
  have e_1094_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_1
  simp only [← c67635, ← c67636, k2, ← c67637] at e_1094_1
  have hr := e_1094_1
  simp only [e_1094_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9346)) := by
  have c67638 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67639 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67640 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67641 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67642 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67643 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_2
  simp only [← c67638, ← c67639, ← c67640] at e_1094_2
  have e_1094_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_3
  simp only [← c67641, ← c67642, k2, ← c67643] at e_1094_3
  have hr := e_1094_3
  simp only [e_1094_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9347)) := by
  have c67644 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67645 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67646 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67647 := (publicBatchWrapper89_copies2113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67648 := (publicBatchWrapper89_copies2114 a h).1
  have c67649 := (publicBatchWrapper89_copies2114 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_4
  simp only [← c67644, ← c67645, ← c67646] at e_1094_4
  have e_1094_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_5
  simp only [← c67647, ← c67648, k2, ← c67649] at e_1094_5
  have hr := e_1094_5
  simp only [e_1094_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9348)) := by
  have c67650 := (publicBatchWrapper89_copies2114 a h).2.2.1
  have c67651 := (publicBatchWrapper89_copies2114 a h).2.2.2.1
  have c67652 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.1
  have c67653 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.1
  have c67654 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.1
  have c67655 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_6
  simp only [← c67650, ← c67651, ← c67652] at e_1094_6
  have e_1094_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_7
  simp only [← c67653, ← c67654, k2, ← c67655] at e_1094_7
  have hr := e_1094_7
  simp only [e_1094_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9349)) := by
  have c67656 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.1
  have c67657 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.1
  have c67658 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67659 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67660 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67661 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_8
  simp only [← c67656, ← c67657, ← c67658] at e_1094_8
  have e_1094_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_9
  simp only [← c67659, ← c67660, k2, ← c67661] at e_1094_9
  have hr := e_1094_9
  simp only [e_1094_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9350)) := by
  have c67662 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67663 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67664 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67665 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67666 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67667 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_10
  simp only [← c67662, ← c67663, ← c67664] at e_1094_10
  have e_1094_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_11
  simp only [← c67665, ← c67666, k2, ← c67667] at e_1094_11
  have hr := e_1094_11
  simp only [e_1094_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9351)) := by
  have c67668 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67669 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67670 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67671 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67672 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67673 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_12
  simp only [← c67668, ← c67669, ← c67670] at e_1094_12
  have e_1094_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_13
  simp only [← c67671, ← c67672, k2, ← c67673] at e_1094_13
  have hr := e_1094_13
  simp only [e_1094_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9352)) := by
  have c67674 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67675 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67676 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67677 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67678 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67679 := (publicBatchWrapper89_copies2114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_14
  simp only [← c67674, ← c67675, ← c67676] at e_1094_14
  have e_1094_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_15
  simp only [← c67677, ← c67678, k2, ← c67679] at e_1094_15
  have hr := e_1094_15
  simp only [e_1094_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9353)) := by
  have c67680 := (publicBatchWrapper89_copies2115 a h).1
  have c67681 := (publicBatchWrapper89_copies2115 a h).2.1
  have c67682 := (publicBatchWrapper89_copies2115 a h).2.2.1
  have c67683 := (publicBatchWrapper89_copies2115 a h).2.2.2.1
  have c67684 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.1
  have c67685 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_16
  simp only [← c67680, ← c67681, ← c67682] at e_1094_16
  have e_1094_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_17
  simp only [← c67683, ← c67684, k2, ← c67685] at e_1094_17
  have hr := e_1094_17
  simp only [e_1094_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1094 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9354)) := by
  have c67686 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.1
  have c67687 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.1
  have c67688 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.1
  have c67689 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.1
  have c67690 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67691 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1094_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_18
  simp only [← c67686, ← c67687, ← c67688] at e_1094_18
  have e_1094_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1094 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1094_19
  simp only [← c67689, ← c67690, k2, ← c67691] at e_1094_19
  have hr := e_1094_19
  simp only [e_1094_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9355)) := by
  have c67692 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67693 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67694 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67695 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67696 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67697 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_0
  simp only [← c67692, ← c67693, ← c67694] at e_1095_0
  have e_1095_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_1
  simp only [← c67695, ← c67696, k2, ← c67697] at e_1095_1
  have hr := e_1095_1
  simp only [e_1095_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9356)) := by
  have c67698 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67699 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67700 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67701 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67702 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67703 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_2
  simp only [← c67698, ← c67699, ← c67700] at e_1095_2
  have e_1095_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_3
  simp only [← c67701, ← c67702, k2, ← c67703] at e_1095_3
  have hr := e_1095_3
  simp only [e_1095_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9357)) := by
  have c67704 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67705 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67706 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67707 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67708 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67709 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_4
  simp only [← c67704, ← c67705, ← c67706] at e_1095_4
  have e_1095_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_5
  simp only [← c67707, ← c67708, k2, ← c67709] at e_1095_5
  have hr := e_1095_5
  simp only [e_1095_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9358)) := by
  have c67710 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67711 := (publicBatchWrapper89_copies2115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67712 := (publicBatchWrapper89_copies2116 a h).1
  have c67713 := (publicBatchWrapper89_copies2116 a h).2.1
  have c67714 := (publicBatchWrapper89_copies2116 a h).2.2.1
  have c67715 := (publicBatchWrapper89_copies2116 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_6
  simp only [← c67710, ← c67711, ← c67712] at e_1095_6
  have e_1095_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_7
  simp only [← c67713, ← c67714, k2, ← c67715] at e_1095_7
  have hr := e_1095_7
  simp only [e_1095_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9359)) := by
  have c67716 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.1
  have c67717 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.1
  have c67718 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.1
  have c67719 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.1
  have c67720 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.1
  have c67721 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_8
  simp only [← c67716, ← c67717, ← c67718] at e_1095_8
  have e_1095_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_9
  simp only [← c67719, ← c67720, k2, ← c67721] at e_1095_9
  have hr := e_1095_9
  simp only [e_1095_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9360)) := by
  have c67722 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67723 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67724 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67725 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67726 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67727 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_10
  simp only [← c67722, ← c67723, ← c67724] at e_1095_10
  have e_1095_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_11
  simp only [← c67725, ← c67726, k2, ← c67727] at e_1095_11
  have hr := e_1095_11
  simp only [e_1095_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9361)) := by
  have c67728 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67729 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67730 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67731 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67732 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67733 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_12
  simp only [← c67728, ← c67729, ← c67730] at e_1095_12
  have e_1095_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_13
  simp only [← c67731, ← c67732, k2, ← c67733] at e_1095_13
  have hr := e_1095_13
  simp only [e_1095_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9362)) := by
  have c67734 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67735 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67736 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67737 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67738 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67739 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_14
  simp only [← c67734, ← c67735, ← c67736] at e_1095_14
  have e_1095_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_15
  simp only [← c67737, ← c67738, k2, ← c67739] at e_1095_15
  have hr := e_1095_15
  simp only [e_1095_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9363)) := by
  have c67740 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67741 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67742 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67743 := (publicBatchWrapper89_copies2116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67744 := (publicBatchWrapper89_copies2117 a h).1
  have c67745 := (publicBatchWrapper89_copies2117 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_16
  simp only [← c67740, ← c67741, ← c67742] at e_1095_16
  have e_1095_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_17
  simp only [← c67743, ← c67744, k2, ← c67745] at e_1095_17
  have hr := e_1095_17
  simp only [e_1095_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1095 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9364)) := by
  have c67746 := (publicBatchWrapper89_copies2117 a h).2.2.1
  have c67747 := (publicBatchWrapper89_copies2117 a h).2.2.2.1
  have c67748 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.1
  have c67749 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.1
  have c67750 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.1
  have c67751 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1095_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_18
  simp only [← c67746, ← c67747, ← c67748] at e_1095_18
  have e_1095_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1095 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1095_19
  simp only [← c67749, ← c67750, k2, ← c67751] at e_1095_19
  have hr := e_1095_19
  simp only [e_1095_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9365)) := by
  have c67752 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.1
  have c67753 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.1
  have c67754 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67755 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67756 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67757 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_0
  simp only [← c67752, ← c67753, ← c67754] at e_1096_0
  have e_1096_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_1
  simp only [← c67755, ← c67756, k2, ← c67757] at e_1096_1
  have hr := e_1096_1
  simp only [e_1096_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9366)) := by
  have c67758 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67759 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67760 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67761 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67762 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67763 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_2
  simp only [← c67758, ← c67759, ← c67760] at e_1096_2
  have e_1096_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_3
  simp only [← c67761, ← c67762, k2, ← c67763] at e_1096_3
  have hr := e_1096_3
  simp only [e_1096_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9367)) := by
  have c67764 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67765 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67766 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67767 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67768 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67769 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_4
  simp only [← c67764, ← c67765, ← c67766] at e_1096_4
  have e_1096_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_5
  simp only [← c67767, ← c67768, k2, ← c67769] at e_1096_5
  have hr := e_1096_5
  simp only [e_1096_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9368)) := by
  have c67770 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67771 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67772 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67773 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67774 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67775 := (publicBatchWrapper89_copies2117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_6
  simp only [← c67770, ← c67771, ← c67772] at e_1096_6
  have e_1096_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_7
  simp only [← c67773, ← c67774, k2, ← c67775] at e_1096_7
  have hr := e_1096_7
  simp only [e_1096_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9369)) := by
  have c67776 := (publicBatchWrapper89_copies2118 a h).1
  have c67777 := (publicBatchWrapper89_copies2118 a h).2.1
  have c67778 := (publicBatchWrapper89_copies2118 a h).2.2.1
  have c67779 := (publicBatchWrapper89_copies2118 a h).2.2.2.1
  have c67780 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.1
  have c67781 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_8
  simp only [← c67776, ← c67777, ← c67778] at e_1096_8
  have e_1096_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_9
  simp only [← c67779, ← c67780, k2, ← c67781] at e_1096_9
  have hr := e_1096_9
  simp only [e_1096_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9410)) := by
  have c67782 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.1
  have c67783 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.1
  have c67784 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.1
  have c67785 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.1
  have c67786 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67787 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_10
  simp only [← c67782, ← c67783, ← c67784] at e_1096_10
  have e_1096_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_11
  simp only [← c67785, ← c67786, k2, ← c67787] at e_1096_11
  have hr := e_1096_11
  simp only [e_1096_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9411)) := by
  have c67788 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67789 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67790 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67791 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67792 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67793 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_12
  simp only [← c67788, ← c67789, ← c67790] at e_1096_12
  have e_1096_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_13
  simp only [← c67791, ← c67792, k2, ← c67793] at e_1096_13
  have hr := e_1096_13
  simp only [e_1096_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9412)) := by
  have c67794 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67795 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67796 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67797 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67798 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67799 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_14
  simp only [← c67794, ← c67795, ← c67796] at e_1096_14
  have e_1096_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_15
  simp only [← c67797, ← c67798, k2, ← c67799] at e_1096_15
  have hr := e_1096_15
  simp only [e_1096_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9413)) := by
  have c67800 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67801 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67802 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67803 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67804 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67805 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_16
  simp only [← c67800, ← c67801, ← c67802] at e_1096_16
  have e_1096_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_17
  simp only [← c67803, ← c67804, k2, ← c67805] at e_1096_17
  have hr := e_1096_17
  simp only [e_1096_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1096 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9414)) := by
  have c67806 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67807 := (publicBatchWrapper89_copies2118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67808 := (publicBatchWrapper89_copies2119 a h).1
  have c67809 := (publicBatchWrapper89_copies2119 a h).2.1
  have c67810 := (publicBatchWrapper89_copies2119 a h).2.2.1
  have c67811 := (publicBatchWrapper89_copies2119 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1096_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_18
  simp only [← c67806, ← c67807, ← c67808] at e_1096_18
  have e_1096_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1096 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1096_19
  simp only [← c67809, ← c67810, k2, ← c67811] at e_1096_19
  have hr := e_1096_19
  simp only [e_1096_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9415)) := by
  have c67812 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.1
  have c67813 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.1
  have c67814 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.1
  have c67815 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.1
  have c67816 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.1
  have c67817 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_0
  simp only [← c67812, ← c67813, ← c67814] at e_1097_0
  have e_1097_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_1
  simp only [← c67815, ← c67816, k2, ← c67817] at e_1097_1
  have hr := e_1097_1
  simp only [e_1097_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9416)) := by
  have c67818 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67819 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67820 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67821 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67822 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67823 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_2
  simp only [← c67818, ← c67819, ← c67820] at e_1097_2
  have e_1097_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_3
  simp only [← c67821, ← c67822, k2, ← c67823] at e_1097_3
  have hr := e_1097_3
  simp only [e_1097_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9417)) := by
  have c67824 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67825 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67826 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67827 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67828 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67829 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_4
  simp only [← c67824, ← c67825, ← c67826] at e_1097_4
  have e_1097_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_5
  simp only [← c67827, ← c67828, k2, ← c67829] at e_1097_5
  have hr := e_1097_5
  simp only [e_1097_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9418)) := by
  have c67830 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67831 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67832 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67833 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67834 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67835 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_6
  simp only [← c67830, ← c67831, ← c67832] at e_1097_6
  have e_1097_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_7
  simp only [← c67833, ← c67834, k2, ← c67835] at e_1097_7
  have hr := e_1097_7
  simp only [e_1097_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9419)) := by
  have c67836 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67837 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67838 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67839 := (publicBatchWrapper89_copies2119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c67840 := (publicBatchWrapper89_copies2120 a h).1
  have c67841 := (publicBatchWrapper89_copies2120 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_8
  simp only [← c67836, ← c67837, ← c67838] at e_1097_8
  have e_1097_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_9
  simp only [← c67839, ← c67840, k2, ← c67841] at e_1097_9
  have hr := e_1097_9
  simp only [e_1097_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9420)) := by
  have c67842 := (publicBatchWrapper89_copies2120 a h).2.2.1
  have c67843 := (publicBatchWrapper89_copies2120 a h).2.2.2.1
  have c67844 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.1
  have c67845 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.1
  have c67846 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.1
  have c67847 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_10
  simp only [← c67842, ← c67843, ← c67844] at e_1097_10
  have e_1097_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_11
  simp only [← c67845, ← c67846, k2, ← c67847] at e_1097_11
  have hr := e_1097_11
  simp only [e_1097_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9421)) := by
  have c67848 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.1
  have c67849 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.1
  have c67850 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c67851 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c67852 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67853 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_12
  simp only [← c67848, ← c67849, ← c67850] at e_1097_12
  have e_1097_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_13
  simp only [← c67851, ← c67852, k2, ← c67853] at e_1097_13
  have hr := e_1097_13
  simp only [e_1097_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f10558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1097 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9422)) := by
  have c67854 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67855 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67856 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67857 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67858 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c67859 := (publicBatchWrapper89_copies2120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1097_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_14
  simp only [← c67854, ← c67855, ← c67856] at e_1097_14
  have e_1097_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1097 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1097_15
  simp only [← c67857, ← c67858, k2, ← c67859] at e_1097_15
  have hr := e_1097_15
  simp only [e_1097_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
