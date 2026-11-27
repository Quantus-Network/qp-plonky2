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

theorem publicBatchWrapper89_f13635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13259)) := by
  have c86316 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86317 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86318 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86319 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86320 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86321 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_8
  simp only [← c86316, ← c86317, ← c86318] at e_1405_8
  have e_1405_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_9
  simp only [← c86319, ← c86320, k2, ← c86321] at e_1405_9
  have hr := e_1405_9
  simp only [e_1405_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13260)) := by
  have c86322 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86323 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86324 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86325 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86326 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86327 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_10
  simp only [← c86322, ← c86323, ← c86324] at e_1405_10
  have e_1405_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_11
  simp only [← c86325, ← c86326, k2, ← c86327] at e_1405_11
  have hr := e_1405_11
  simp only [e_1405_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13261)) := by
  have c86328 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86329 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86330 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86331 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86332 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86333 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_12
  simp only [← c86328, ← c86329, ← c86330] at e_1405_12
  have e_1405_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_13
  simp only [← c86331, ← c86332, k2, ← c86333] at e_1405_13
  have hr := e_1405_13
  simp only [e_1405_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13262)) := by
  have c86334 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86335 := (publicBatchWrapper89_copies2697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86336 := (publicBatchWrapper89_copies2698 a h).1
  have c86337 := (publicBatchWrapper89_copies2698 a h).2.1
  have c86338 := (publicBatchWrapper89_copies2698 a h).2.2.1
  have c86339 := (publicBatchWrapper89_copies2698 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_14
  simp only [← c86334, ← c86335, ← c86336] at e_1405_14
  have e_1405_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_15
  simp only [← c86337, ← c86338, k2, ← c86339] at e_1405_15
  have hr := e_1405_15
  simp only [e_1405_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13263)) := by
  have c86340 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.1
  have c86341 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.1
  have c86342 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.1
  have c86343 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.1
  have c86344 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.1
  have c86345 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_16
  simp only [← c86340, ← c86341, ← c86342] at e_1405_16
  have e_1405_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_17
  simp only [← c86343, ← c86344, k2, ← c86345] at e_1405_17
  have hr := e_1405_17
  simp only [e_1405_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1405 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13264)) := by
  have c86346 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86347 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86348 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86349 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86350 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86351 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1405_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_18
  simp only [← c86346, ← c86347, ← c86348] at e_1405_18
  have e_1405_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1405 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1405_19
  simp only [← c86349, ← c86350, k2, ← c86351] at e_1405_19
  have hr := e_1405_19
  simp only [e_1405_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13265)) := by
  have c86352 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86353 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86354 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86355 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86356 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86357 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_0
  simp only [← c86352, ← c86353, ← c86354] at e_1406_0
  have e_1406_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_1
  simp only [← c86355, ← c86356, k2, ← c86357] at e_1406_1
  have hr := e_1406_1
  simp only [e_1406_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13266)) := by
  have c86358 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86359 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86360 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86361 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86362 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86363 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_2
  simp only [← c86358, ← c86359, ← c86360] at e_1406_2
  have e_1406_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_3
  simp only [← c86361, ← c86362, k2, ← c86363] at e_1406_3
  have hr := e_1406_3
  simp only [e_1406_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13267)) := by
  have c86364 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86365 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86366 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86367 := (publicBatchWrapper89_copies2698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86368 := (publicBatchWrapper89_copies2699 a h).1
  have c86369 := (publicBatchWrapper89_copies2699 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_4
  simp only [← c86364, ← c86365, ← c86366] at e_1406_4
  have e_1406_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_5
  simp only [← c86367, ← c86368, k2, ← c86369] at e_1406_5
  have hr := e_1406_5
  simp only [e_1406_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13268)) := by
  have c86370 := (publicBatchWrapper89_copies2699 a h).2.2.1
  have c86371 := (publicBatchWrapper89_copies2699 a h).2.2.2.1
  have c86372 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.1
  have c86373 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.1
  have c86374 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.1
  have c86375 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_6
  simp only [← c86370, ← c86371, ← c86372] at e_1406_6
  have e_1406_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_7
  simp only [← c86373, ← c86374, k2, ← c86375] at e_1406_7
  have hr := e_1406_7
  simp only [e_1406_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13269)) := by
  have c86376 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.1
  have c86377 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.1
  have c86378 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86379 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86380 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86381 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_8
  simp only [← c86376, ← c86377, ← c86378] at e_1406_8
  have e_1406_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_9
  simp only [← c86379, ← c86380, k2, ← c86381] at e_1406_9
  have hr := e_1406_9
  simp only [e_1406_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13270)) := by
  have c86382 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86383 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86384 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86385 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86386 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86387 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_10
  simp only [← c86382, ← c86383, ← c86384] at e_1406_10
  have e_1406_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_11
  simp only [← c86385, ← c86386, k2, ← c86387] at e_1406_11
  have hr := e_1406_11
  simp only [e_1406_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13271)) := by
  have c86388 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86389 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86390 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86391 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86392 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86393 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_12
  simp only [← c86388, ← c86389, ← c86390] at e_1406_12
  have e_1406_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_13
  simp only [← c86391, ← c86392, k2, ← c86393] at e_1406_13
  have hr := e_1406_13
  simp only [e_1406_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13272)) := by
  have c86394 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86395 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86396 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86397 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86398 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86399 := (publicBatchWrapper89_copies2699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_14
  simp only [← c86394, ← c86395, ← c86396] at e_1406_14
  have e_1406_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_15
  simp only [← c86397, ← c86398, k2, ← c86399] at e_1406_15
  have hr := e_1406_15
  simp only [e_1406_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13273)) := by
  have c86400 := (publicBatchWrapper89_copies2700 a h).1
  have c86401 := (publicBatchWrapper89_copies2700 a h).2.1
  have c86402 := (publicBatchWrapper89_copies2700 a h).2.2.1
  have c86403 := (publicBatchWrapper89_copies2700 a h).2.2.2.1
  have c86404 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.1
  have c86405 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_16
  simp only [← c86400, ← c86401, ← c86402] at e_1406_16
  have e_1406_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_17
  simp only [← c86403, ← c86404, k2, ← c86405] at e_1406_17
  have hr := e_1406_17
  simp only [e_1406_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1406 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13274)) := by
  have c86406 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.1
  have c86407 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.1
  have c86408 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.1
  have c86409 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.1
  have c86410 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86411 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1406_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_18
  simp only [← c86406, ← c86407, ← c86408] at e_1406_18
  have e_1406_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1406 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1406_19
  simp only [← c86409, ← c86410, k2, ← c86411] at e_1406_19
  have hr := e_1406_19
  simp only [e_1406_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13275)) := by
  have c86412 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86413 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86414 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86415 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86416 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86417 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_0
  simp only [← c86412, ← c86413, ← c86414] at e_1407_0
  have e_1407_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_1
  simp only [← c86415, ← c86416, k2, ← c86417] at e_1407_1
  have hr := e_1407_1
  simp only [e_1407_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13276)) := by
  have c86418 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86419 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86420 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86421 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86422 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86423 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_2
  simp only [← c86418, ← c86419, ← c86420] at e_1407_2
  have e_1407_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_3
  simp only [← c86421, ← c86422, k2, ← c86423] at e_1407_3
  have hr := e_1407_3
  simp only [e_1407_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13277)) := by
  have c86424 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86425 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86426 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86427 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86428 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86429 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_4
  simp only [← c86424, ← c86425, ← c86426] at e_1407_4
  have e_1407_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_5
  simp only [← c86427, ← c86428, k2, ← c86429] at e_1407_5
  have hr := e_1407_5
  simp only [e_1407_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13278)) := by
  have c86430 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86431 := (publicBatchWrapper89_copies2700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86432 := (publicBatchWrapper89_copies2701 a h).1
  have c86433 := (publicBatchWrapper89_copies2701 a h).2.1
  have c86434 := (publicBatchWrapper89_copies2701 a h).2.2.1
  have c86435 := (publicBatchWrapper89_copies2701 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_6
  simp only [← c86430, ← c86431, ← c86432] at e_1407_6
  have e_1407_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_7
  simp only [← c86433, ← c86434, k2, ← c86435] at e_1407_7
  have hr := e_1407_7
  simp only [e_1407_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13279)) := by
  have c86436 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.1
  have c86437 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.1
  have c86438 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.1
  have c86439 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.1
  have c86440 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.1
  have c86441 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_8
  simp only [← c86436, ← c86437, ← c86438] at e_1407_8
  have e_1407_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_9
  simp only [← c86439, ← c86440, k2, ← c86441] at e_1407_9
  have hr := e_1407_9
  simp only [e_1407_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13280)) := by
  have c86442 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86443 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86444 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86445 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86446 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86447 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_10
  simp only [← c86442, ← c86443, ← c86444] at e_1407_10
  have e_1407_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_11
  simp only [← c86445, ← c86446, k2, ← c86447] at e_1407_11
  have hr := e_1407_11
  simp only [e_1407_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13281)) := by
  have c86448 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86449 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86450 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86451 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86452 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86453 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_12
  simp only [← c86448, ← c86449, ← c86450] at e_1407_12
  have e_1407_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_13
  simp only [← c86451, ← c86452, k2, ← c86453] at e_1407_13
  have hr := e_1407_13
  simp only [e_1407_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13282)) := by
  have c86454 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86455 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86456 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86457 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86458 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86459 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_14
  simp only [← c86454, ← c86455, ← c86456] at e_1407_14
  have e_1407_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_15
  simp only [← c86457, ← c86458, k2, ← c86459] at e_1407_15
  have hr := e_1407_15
  simp only [e_1407_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13283)) := by
  have c86460 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86461 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86462 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86463 := (publicBatchWrapper89_copies2701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86464 := (publicBatchWrapper89_copies2702 a h).1
  have c86465 := (publicBatchWrapper89_copies2702 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_16
  simp only [← c86460, ← c86461, ← c86462] at e_1407_16
  have e_1407_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_17
  simp only [← c86463, ← c86464, k2, ← c86465] at e_1407_17
  have hr := e_1407_17
  simp only [e_1407_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1407 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13284)) := by
  have c86466 := (publicBatchWrapper89_copies2702 a h).2.2.1
  have c86467 := (publicBatchWrapper89_copies2702 a h).2.2.2.1
  have c86468 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.1
  have c86469 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.1
  have c86470 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.1
  have c86471 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1407_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_18
  simp only [← c86466, ← c86467, ← c86468] at e_1407_18
  have e_1407_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1407 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1407_19
  simp only [← c86469, ← c86470, k2, ← c86471] at e_1407_19
  have hr := e_1407_19
  simp only [e_1407_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13285)) := by
  have c86472 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.1
  have c86473 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.1
  have c86474 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86475 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86476 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86477 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_0
  simp only [← c86472, ← c86473, ← c86474] at e_1408_0
  have e_1408_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_1
  simp only [← c86475, ← c86476, k2, ← c86477] at e_1408_1
  have hr := e_1408_1
  simp only [e_1408_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13286)) := by
  have c86478 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86479 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86480 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86481 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86482 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86483 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_2
  simp only [← c86478, ← c86479, ← c86480] at e_1408_2
  have e_1408_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_3
  simp only [← c86481, ← c86482, k2, ← c86483] at e_1408_3
  have hr := e_1408_3
  simp only [e_1408_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13287)) := by
  have c86484 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86485 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86486 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86487 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86488 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86489 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_4
  simp only [← c86484, ← c86485, ← c86486] at e_1408_4
  have e_1408_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_5
  simp only [← c86487, ← c86488, k2, ← c86489] at e_1408_5
  have hr := e_1408_5
  simp only [e_1408_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13288)) := by
  have c86490 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86491 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86492 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86493 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86494 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86495 := (publicBatchWrapper89_copies2702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_6
  simp only [← c86490, ← c86491, ← c86492] at e_1408_6
  have e_1408_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_7
  simp only [← c86493, ← c86494, k2, ← c86495] at e_1408_7
  have hr := e_1408_7
  simp only [e_1408_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13289)) := by
  have c86496 := (publicBatchWrapper89_copies2703 a h).1
  have c86497 := (publicBatchWrapper89_copies2703 a h).2.1
  have c86498 := (publicBatchWrapper89_copies2703 a h).2.2.1
  have c86499 := (publicBatchWrapper89_copies2703 a h).2.2.2.1
  have c86500 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.1
  have c86501 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_8
  simp only [← c86496, ← c86497, ← c86498] at e_1408_8
  have e_1408_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_9
  simp only [← c86499, ← c86500, k2, ← c86501] at e_1408_9
  have hr := e_1408_9
  simp only [e_1408_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13290)) := by
  have c86502 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.1
  have c86503 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.1
  have c86504 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.1
  have c86505 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.1
  have c86506 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86507 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_10
  simp only [← c86502, ← c86503, ← c86504] at e_1408_10
  have e_1408_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_11
  simp only [← c86505, ← c86506, k2, ← c86507] at e_1408_11
  have hr := e_1408_11
  simp only [e_1408_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13291)) := by
  have c86508 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86509 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86510 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86511 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86512 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86513 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_12
  simp only [← c86508, ← c86509, ← c86510] at e_1408_12
  have e_1408_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_13
  simp only [← c86511, ← c86512, k2, ← c86513] at e_1408_13
  have hr := e_1408_13
  simp only [e_1408_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13292)) := by
  have c86514 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86515 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86516 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86517 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86518 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86519 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_14
  simp only [← c86514, ← c86515, ← c86516] at e_1408_14
  have e_1408_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_15
  simp only [← c86517, ← c86518, k2, ← c86519] at e_1408_15
  have hr := e_1408_15
  simp only [e_1408_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13293)) := by
  have c86520 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86521 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86522 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86523 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86524 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86525 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_16
  simp only [← c86520, ← c86521, ← c86522] at e_1408_16
  have e_1408_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_17
  simp only [← c86523, ← c86524, k2, ← c86525] at e_1408_17
  have hr := e_1408_17
  simp only [e_1408_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1408 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13294)) := by
  have c86526 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86527 := (publicBatchWrapper89_copies2703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86528 := (publicBatchWrapper89_copies2704 a h).1
  have c86529 := (publicBatchWrapper89_copies2704 a h).2.1
  have c86530 := (publicBatchWrapper89_copies2704 a h).2.2.1
  have c86531 := (publicBatchWrapper89_copies2704 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1408_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_18
  simp only [← c86526, ← c86527, ← c86528] at e_1408_18
  have e_1408_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1408 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1408_19
  simp only [← c86529, ← c86530, k2, ← c86531] at e_1408_19
  have hr := e_1408_19
  simp only [e_1408_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13295)) := by
  have c86532 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.1
  have c86533 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.1
  have c86534 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.1
  have c86535 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.1
  have c86536 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.1
  have c86537 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_0
  simp only [← c86532, ← c86533, ← c86534] at e_1409_0
  have e_1409_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_1
  simp only [← c86535, ← c86536, k2, ← c86537] at e_1409_1
  have hr := e_1409_1
  simp only [e_1409_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13296)) := by
  have c86538 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86539 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86540 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86541 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86542 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86543 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_2
  simp only [← c86538, ← c86539, ← c86540] at e_1409_2
  have e_1409_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_3
  simp only [← c86541, ← c86542, k2, ← c86543] at e_1409_3
  have hr := e_1409_3
  simp only [e_1409_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13297)) := by
  have c86544 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86545 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86546 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86547 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86548 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86549 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_4
  simp only [← c86544, ← c86545, ← c86546] at e_1409_4
  have e_1409_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_5
  simp only [← c86547, ← c86548, k2, ← c86549] at e_1409_5
  have hr := e_1409_5
  simp only [e_1409_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13298)) := by
  have c86550 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86551 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86552 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86553 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86554 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86555 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_6
  simp only [← c86550, ← c86551, ← c86552] at e_1409_6
  have e_1409_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_7
  simp only [← c86553, ← c86554, k2, ← c86555] at e_1409_7
  have hr := e_1409_7
  simp only [e_1409_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13299)) := by
  have c86556 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86557 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86558 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86559 := (publicBatchWrapper89_copies2704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86560 := (publicBatchWrapper89_copies2705 a h).1
  have c86561 := (publicBatchWrapper89_copies2705 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_8
  simp only [← c86556, ← c86557, ← c86558] at e_1409_8
  have e_1409_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_9
  simp only [← c86559, ← c86560, k2, ← c86561] at e_1409_9
  have hr := e_1409_9
  simp only [e_1409_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13300)) := by
  have c86562 := (publicBatchWrapper89_copies2705 a h).2.2.1
  have c86563 := (publicBatchWrapper89_copies2705 a h).2.2.2.1
  have c86564 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.1
  have c86565 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.1
  have c86566 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.1
  have c86567 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_10
  simp only [← c86562, ← c86563, ← c86564] at e_1409_10
  have e_1409_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_11
  simp only [← c86565, ← c86566, k2, ← c86567] at e_1409_11
  have hr := e_1409_11
  simp only [e_1409_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13301)) := by
  have c86568 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.1
  have c86569 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.1
  have c86570 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86571 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86572 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86573 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_12
  simp only [← c86568, ← c86569, ← c86570] at e_1409_12
  have e_1409_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_13
  simp only [← c86571, ← c86572, k2, ← c86573] at e_1409_13
  have hr := e_1409_13
  simp only [e_1409_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13302)) := by
  have c86574 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86575 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86576 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86577 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86578 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86579 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_14
  simp only [← c86574, ← c86575, ← c86576] at e_1409_14
  have e_1409_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_15
  simp only [← c86577, ← c86578, k2, ← c86579] at e_1409_15
  have hr := e_1409_15
  simp only [e_1409_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13303)) := by
  have c86580 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86581 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86582 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86583 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86584 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86585 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_16
  simp only [← c86580, ← c86581, ← c86582] at e_1409_16
  have e_1409_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_17
  simp only [← c86583, ← c86584, k2, ← c86585] at e_1409_17
  have hr := e_1409_17
  simp only [e_1409_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1409 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13304)) := by
  have c86586 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86587 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86588 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86589 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86590 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86591 := (publicBatchWrapper89_copies2705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1409_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_18
  simp only [← c86586, ← c86587, ← c86588] at e_1409_18
  have e_1409_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1409 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1409_19
  simp only [← c86589, ← c86590, k2, ← c86591] at e_1409_19
  have hr := e_1409_19
  simp only [e_1409_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13305)) := by
  have c86592 := (publicBatchWrapper89_copies2706 a h).1
  have c86593 := (publicBatchWrapper89_copies2706 a h).2.1
  have c86594 := (publicBatchWrapper89_copies2706 a h).2.2.1
  have c86595 := (publicBatchWrapper89_copies2706 a h).2.2.2.1
  have c86596 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.1
  have c86597 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_0
  simp only [← c86592, ← c86593, ← c86594] at e_1410_0
  have e_1410_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_1
  simp only [← c86595, ← c86596, k2, ← c86597] at e_1410_1
  have hr := e_1410_1
  simp only [e_1410_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13306)) := by
  have c86598 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.1
  have c86599 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.1
  have c86600 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.1
  have c86601 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.1
  have c86602 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86603 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_2
  simp only [← c86598, ← c86599, ← c86600] at e_1410_2
  have e_1410_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_3
  simp only [← c86601, ← c86602, k2, ← c86603] at e_1410_3
  have hr := e_1410_3
  simp only [e_1410_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13307)) := by
  have c86604 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86605 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86606 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86607 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86608 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86609 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_4
  simp only [← c86604, ← c86605, ← c86606] at e_1410_4
  have e_1410_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_5
  simp only [← c86607, ← c86608, k2, ← c86609] at e_1410_5
  have hr := e_1410_5
  simp only [e_1410_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13308)) := by
  have c86610 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86611 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86612 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86613 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86614 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86615 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_6
  simp only [← c86610, ← c86611, ← c86612] at e_1410_6
  have e_1410_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_7
  simp only [← c86613, ← c86614, k2, ← c86615] at e_1410_7
  have hr := e_1410_7
  simp only [e_1410_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13309)) := by
  have c86616 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86617 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86618 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86619 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86620 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86621 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_8
  simp only [← c86616, ← c86617, ← c86618] at e_1410_8
  have e_1410_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_9
  simp only [← c86619, ← c86620, k2, ← c86621] at e_1410_9
  have hr := e_1410_9
  simp only [e_1410_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13310)) := by
  have c86622 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86623 := (publicBatchWrapper89_copies2706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86624 := (publicBatchWrapper89_copies2707 a h).1
  have c86625 := (publicBatchWrapper89_copies2707 a h).2.1
  have c86626 := (publicBatchWrapper89_copies2707 a h).2.2.1
  have c86627 := (publicBatchWrapper89_copies2707 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_10
  simp only [← c86622, ← c86623, ← c86624] at e_1410_10
  have e_1410_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_11
  simp only [← c86625, ← c86626, k2, ← c86627] at e_1410_11
  have hr := e_1410_11
  simp only [e_1410_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13311)) := by
  have c86628 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.1
  have c86629 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.1
  have c86630 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.1
  have c86631 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.1
  have c86632 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.1
  have c86633 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_12
  simp only [← c86628, ← c86629, ← c86630] at e_1410_12
  have e_1410_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_13
  simp only [← c86631, ← c86632, k2, ← c86633] at e_1410_13
  have hr := e_1410_13
  simp only [e_1410_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13312)) := by
  have c86634 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86635 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86636 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86637 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86638 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86639 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_14
  simp only [← c86634, ← c86635, ← c86636] at e_1410_14
  have e_1410_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_15
  simp only [← c86637, ← c86638, k2, ← c86639] at e_1410_15
  have hr := e_1410_15
  simp only [e_1410_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13313)) := by
  have c86640 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86641 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86642 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86643 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86644 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86645 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_16
  simp only [← c86640, ← c86641, ← c86642] at e_1410_16
  have e_1410_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_17
  simp only [← c86643, ← c86644, k2, ← c86645] at e_1410_17
  have hr := e_1410_17
  simp only [e_1410_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1410 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13314)) := by
  have c86646 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86647 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86648 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86649 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86650 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86651 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1410_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_18
  simp only [← c86646, ← c86647, ← c86648] at e_1410_18
  have e_1410_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1410 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1410_19
  simp only [← c86649, ← c86650, k2, ← c86651] at e_1410_19
  have hr := e_1410_19
  simp only [e_1410_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13315)) := by
  have c86652 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86653 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86654 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86655 := (publicBatchWrapper89_copies2707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86656 := (publicBatchWrapper89_copies2708 a h).1
  have c86657 := (publicBatchWrapper89_copies2708 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_0
  simp only [← c86652, ← c86653, ← c86654] at e_1411_0
  have e_1411_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_1
  simp only [← c86655, ← c86656, k2, ← c86657] at e_1411_1
  have hr := e_1411_1
  simp only [e_1411_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13316)) := by
  have c86658 := (publicBatchWrapper89_copies2708 a h).2.2.1
  have c86659 := (publicBatchWrapper89_copies2708 a h).2.2.2.1
  have c86660 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.1
  have c86661 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.1
  have c86662 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.1
  have c86663 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_2
  simp only [← c86658, ← c86659, ← c86660] at e_1411_2
  have e_1411_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_3
  simp only [← c86661, ← c86662, k2, ← c86663] at e_1411_3
  have hr := e_1411_3
  simp only [e_1411_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13317)) := by
  have c86664 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.1
  have c86665 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.1
  have c86666 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86667 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86668 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86669 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_4
  simp only [← c86664, ← c86665, ← c86666] at e_1411_4
  have e_1411_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_5
  simp only [← c86667, ← c86668, k2, ← c86669] at e_1411_5
  have hr := e_1411_5
  simp only [e_1411_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13318)) := by
  have c86670 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86671 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86672 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86673 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86674 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86675 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_6
  simp only [← c86670, ← c86671, ← c86672] at e_1411_6
  have e_1411_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_7
  simp only [← c86673, ← c86674, k2, ← c86675] at e_1411_7
  have hr := e_1411_7
  simp only [e_1411_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13319)) := by
  have c86676 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86677 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86678 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86679 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86680 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86681 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_8
  simp only [← c86676, ← c86677, ← c86678] at e_1411_8
  have e_1411_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_9
  simp only [← c86679, ← c86680, k2, ← c86681] at e_1411_9
  have hr := e_1411_9
  simp only [e_1411_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13320)) := by
  have c86682 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86683 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86684 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86685 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86686 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86687 := (publicBatchWrapper89_copies2708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_10
  simp only [← c86682, ← c86683, ← c86684] at e_1411_10
  have e_1411_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_11
  simp only [← c86685, ← c86686, k2, ← c86687] at e_1411_11
  have hr := e_1411_11
  simp only [e_1411_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13321)) := by
  have c86688 := (publicBatchWrapper89_copies2709 a h).1
  have c86689 := (publicBatchWrapper89_copies2709 a h).2.1
  have c86690 := (publicBatchWrapper89_copies2709 a h).2.2.1
  have c86691 := (publicBatchWrapper89_copies2709 a h).2.2.2.1
  have c86692 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.1
  have c86693 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_12
  simp only [← c86688, ← c86689, ← c86690] at e_1411_12
  have e_1411_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_13
  simp only [← c86691, ← c86692, k2, ← c86693] at e_1411_13
  have hr := e_1411_13
  simp only [e_1411_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13322)) := by
  have c86694 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.1
  have c86695 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.1
  have c86696 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.1
  have c86697 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.1
  have c86698 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86699 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_14
  simp only [← c86694, ← c86695, ← c86696] at e_1411_14
  have e_1411_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_15
  simp only [← c86697, ← c86698, k2, ← c86699] at e_1411_15
  have hr := e_1411_15
  simp only [e_1411_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13323)) := by
  have c86700 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86701 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86702 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86703 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86704 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86705 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_16
  simp only [← c86700, ← c86701, ← c86702] at e_1411_16
  have e_1411_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_17
  simp only [← c86703, ← c86704, k2, ← c86705] at e_1411_17
  have hr := e_1411_17
  simp only [e_1411_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1411 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13324)) := by
  have c86706 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86707 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86708 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86709 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86710 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86711 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1411_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_18
  simp only [← c86706, ← c86707, ← c86708] at e_1411_18
  have e_1411_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1411 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1411_19
  simp only [← c86709, ← c86710, k2, ← c86711] at e_1411_19
  have hr := e_1411_19
  simp only [e_1411_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13325)) := by
  have c86712 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86713 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86714 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86715 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86716 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86717 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_0
  simp only [← c86712, ← c86713, ← c86714] at e_1412_0
  have e_1412_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_1
  simp only [← c86715, ← c86716, k2, ← c86717] at e_1412_1
  have hr := e_1412_1
  simp only [e_1412_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13326)) := by
  have c86718 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86719 := (publicBatchWrapper89_copies2709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86720 := (publicBatchWrapper89_copies2710 a h).1
  have c86721 := (publicBatchWrapper89_copies2710 a h).2.1
  have c86722 := (publicBatchWrapper89_copies2710 a h).2.2.1
  have c86723 := (publicBatchWrapper89_copies2710 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_2
  simp only [← c86718, ← c86719, ← c86720] at e_1412_2
  have e_1412_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_3
  simp only [← c86721, ← c86722, k2, ← c86723] at e_1412_3
  have hr := e_1412_3
  simp only [e_1412_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13327)) := by
  have c86724 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.1
  have c86725 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.1
  have c86726 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.1
  have c86727 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.1
  have c86728 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.1
  have c86729 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_4
  simp only [← c86724, ← c86725, ← c86726] at e_1412_4
  have e_1412_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_5
  simp only [← c86727, ← c86728, k2, ← c86729] at e_1412_5
  have hr := e_1412_5
  simp only [e_1412_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13328)) := by
  have c86730 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86731 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86732 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86733 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86734 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86735 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_6
  simp only [← c86730, ← c86731, ← c86732] at e_1412_6
  have e_1412_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_7
  simp only [← c86733, ← c86734, k2, ← c86735] at e_1412_7
  have hr := e_1412_7
  simp only [e_1412_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13329)) := by
  have c86736 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86737 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86738 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86739 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86740 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86741 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_8
  simp only [← c86736, ← c86737, ← c86738] at e_1412_8
  have e_1412_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_9
  simp only [← c86739, ← c86740, k2, ← c86741] at e_1412_9
  have hr := e_1412_9
  simp only [e_1412_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13330)) := by
  have c86742 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86743 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86744 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86745 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86746 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86747 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_10
  simp only [← c86742, ← c86743, ← c86744] at e_1412_10
  have e_1412_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_11
  simp only [← c86745, ← c86746, k2, ← c86747] at e_1412_11
  have hr := e_1412_11
  simp only [e_1412_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13331)) := by
  have c86748 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86749 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86750 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86751 := (publicBatchWrapper89_copies2710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86752 := (publicBatchWrapper89_copies2711 a h).1
  have c86753 := (publicBatchWrapper89_copies2711 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_12
  simp only [← c86748, ← c86749, ← c86750] at e_1412_12
  have e_1412_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_13
  simp only [← c86751, ← c86752, k2, ← c86753] at e_1412_13
  have hr := e_1412_13
  simp only [e_1412_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13332)) := by
  have c86754 := (publicBatchWrapper89_copies2711 a h).2.2.1
  have c86755 := (publicBatchWrapper89_copies2711 a h).2.2.2.1
  have c86756 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.1
  have c86757 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.1
  have c86758 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.1
  have c86759 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_14
  simp only [← c86754, ← c86755, ← c86756] at e_1412_14
  have e_1412_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_15
  simp only [← c86757, ← c86758, k2, ← c86759] at e_1412_15
  have hr := e_1412_15
  simp only [e_1412_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13333)) := by
  have c86760 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.1
  have c86761 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.1
  have c86762 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86763 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86764 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86765 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_16
  simp only [← c86760, ← c86761, ← c86762] at e_1412_16
  have e_1412_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_17
  simp only [← c86763, ← c86764, k2, ← c86765] at e_1412_17
  have hr := e_1412_17
  simp only [e_1412_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1412 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13334)) := by
  have c86766 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86767 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86768 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86769 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86770 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86771 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1412_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_18
  simp only [← c86766, ← c86767, ← c86768] at e_1412_18
  have e_1412_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1412 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1412_19
  simp only [← c86769, ← c86770, k2, ← c86771] at e_1412_19
  have hr := e_1412_19
  simp only [e_1412_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13335)) := by
  have c86772 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86773 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86774 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86775 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86776 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86777 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_0
  simp only [← c86772, ← c86773, ← c86774] at e_1413_0
  have e_1413_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_1
  simp only [← c86775, ← c86776, k2, ← c86777] at e_1413_1
  have hr := e_1413_1
  simp only [e_1413_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13336)) := by
  have c86778 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86779 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86780 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86781 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86782 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86783 := (publicBatchWrapper89_copies2711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_2
  simp only [← c86778, ← c86779, ← c86780] at e_1413_2
  have e_1413_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_3
  simp only [← c86781, ← c86782, k2, ← c86783] at e_1413_3
  have hr := e_1413_3
  simp only [e_1413_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13337)) := by
  have c86784 := (publicBatchWrapper89_copies2712 a h).1
  have c86785 := (publicBatchWrapper89_copies2712 a h).2.1
  have c86786 := (publicBatchWrapper89_copies2712 a h).2.2.1
  have c86787 := (publicBatchWrapper89_copies2712 a h).2.2.2.1
  have c86788 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.1
  have c86789 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_4
  simp only [← c86784, ← c86785, ← c86786] at e_1413_4
  have e_1413_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_5
  simp only [← c86787, ← c86788, k2, ← c86789] at e_1413_5
  have hr := e_1413_5
  simp only [e_1413_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13338)) := by
  have c86790 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.1
  have c86791 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.1
  have c86792 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.1
  have c86793 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.1
  have c86794 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86795 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_6
  simp only [← c86790, ← c86791, ← c86792] at e_1413_6
  have e_1413_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_7
  simp only [← c86793, ← c86794, k2, ← c86795] at e_1413_7
  have hr := e_1413_7
  simp only [e_1413_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13339)) := by
  have c86796 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86797 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86798 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86799 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86800 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86801 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_8
  simp only [← c86796, ← c86797, ← c86798] at e_1413_8
  have e_1413_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_9
  simp only [← c86799, ← c86800, k2, ← c86801] at e_1413_9
  have hr := e_1413_9
  simp only [e_1413_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13340)) := by
  have c86802 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86803 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86804 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86805 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86806 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86807 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_10
  simp only [← c86802, ← c86803, ← c86804] at e_1413_10
  have e_1413_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_11
  simp only [← c86805, ← c86806, k2, ← c86807] at e_1413_11
  have hr := e_1413_11
  simp only [e_1413_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13341)) := by
  have c86808 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86809 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86810 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86811 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86812 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86813 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_12
  simp only [← c86808, ← c86809, ← c86810] at e_1413_12
  have e_1413_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_13
  simp only [← c86811, ← c86812, k2, ← c86813] at e_1413_13
  have hr := e_1413_13
  simp only [e_1413_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13342)) := by
  have c86814 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86815 := (publicBatchWrapper89_copies2712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86816 := (publicBatchWrapper89_copies2713 a h).1
  have c86817 := (publicBatchWrapper89_copies2713 a h).2.1
  have c86818 := (publicBatchWrapper89_copies2713 a h).2.2.1
  have c86819 := (publicBatchWrapper89_copies2713 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_14
  simp only [← c86814, ← c86815, ← c86816] at e_1413_14
  have e_1413_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_15
  simp only [← c86817, ← c86818, k2, ← c86819] at e_1413_15
  have hr := e_1413_15
  simp only [e_1413_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13343)) := by
  have c86820 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.1
  have c86821 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.1
  have c86822 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.1
  have c86823 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.1
  have c86824 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.1
  have c86825 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_16
  simp only [← c86820, ← c86821, ← c86822] at e_1413_16
  have e_1413_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_17
  simp only [← c86823, ← c86824, k2, ← c86825] at e_1413_17
  have hr := e_1413_17
  simp only [e_1413_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1413 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13344)) := by
  have c86826 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86827 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86828 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86829 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86830 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86831 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1413_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_18
  simp only [← c86826, ← c86827, ← c86828] at e_1413_18
  have e_1413_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1413 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1413_19
  simp only [← c86829, ← c86830, k2, ← c86831] at e_1413_19
  have hr := e_1413_19
  simp only [e_1413_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13345)) := by
  have c86832 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86833 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86834 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86835 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86836 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86837 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_0
  simp only [← c86832, ← c86833, ← c86834] at e_1414_0
  have e_1414_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_1
  simp only [← c86835, ← c86836, k2, ← c86837] at e_1414_1
  have hr := e_1414_1
  simp only [e_1414_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13346)) := by
  have c86838 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86839 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86840 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86841 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86842 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86843 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_2
  simp only [← c86838, ← c86839, ← c86840] at e_1414_2
  have e_1414_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_3
  simp only [← c86841, ← c86842, k2, ← c86843] at e_1414_3
  have hr := e_1414_3
  simp only [e_1414_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13347)) := by
  have c86844 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86845 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86846 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86847 := (publicBatchWrapper89_copies2713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86848 := (publicBatchWrapper89_copies2714 a h).1
  have c86849 := (publicBatchWrapper89_copies2714 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_4
  simp only [← c86844, ← c86845, ← c86846] at e_1414_4
  have e_1414_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_5
  simp only [← c86847, ← c86848, k2, ← c86849] at e_1414_5
  have hr := e_1414_5
  simp only [e_1414_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13348)) := by
  have c86850 := (publicBatchWrapper89_copies2714 a h).2.2.1
  have c86851 := (publicBatchWrapper89_copies2714 a h).2.2.2.1
  have c86852 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.1
  have c86853 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.1
  have c86854 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.1
  have c86855 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_6
  simp only [← c86850, ← c86851, ← c86852] at e_1414_6
  have e_1414_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_7
  simp only [← c86853, ← c86854, k2, ← c86855] at e_1414_7
  have hr := e_1414_7
  simp only [e_1414_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13349)) := by
  have c86856 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.1
  have c86857 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.1
  have c86858 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86859 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86860 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86861 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_8
  simp only [← c86856, ← c86857, ← c86858] at e_1414_8
  have e_1414_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_9
  simp only [← c86859, ← c86860, k2, ← c86861] at e_1414_9
  have hr := e_1414_9
  simp only [e_1414_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13350)) := by
  have c86862 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86863 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86864 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86865 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86866 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86867 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_10
  simp only [← c86862, ← c86863, ← c86864] at e_1414_10
  have e_1414_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_11
  simp only [← c86865, ← c86866, k2, ← c86867] at e_1414_11
  have hr := e_1414_11
  simp only [e_1414_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13351)) := by
  have c86868 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86869 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86870 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86871 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86872 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86873 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_12
  simp only [← c86868, ← c86869, ← c86870] at e_1414_12
  have e_1414_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_13
  simp only [← c86871, ← c86872, k2, ← c86873] at e_1414_13
  have hr := e_1414_13
  simp only [e_1414_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13352)) := by
  have c86874 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86875 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86876 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86877 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86878 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86879 := (publicBatchWrapper89_copies2714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_14
  simp only [← c86874, ← c86875, ← c86876] at e_1414_14
  have e_1414_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_15
  simp only [← c86877, ← c86878, k2, ← c86879] at e_1414_15
  have hr := e_1414_15
  simp only [e_1414_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13353)) := by
  have c86880 := (publicBatchWrapper89_copies2715 a h).1
  have c86881 := (publicBatchWrapper89_copies2715 a h).2.1
  have c86882 := (publicBatchWrapper89_copies2715 a h).2.2.1
  have c86883 := (publicBatchWrapper89_copies2715 a h).2.2.2.1
  have c86884 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.1
  have c86885 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_16
  simp only [← c86880, ← c86881, ← c86882] at e_1414_16
  have e_1414_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_17
  simp only [← c86883, ← c86884, k2, ← c86885] at e_1414_17
  have hr := e_1414_17
  simp only [e_1414_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1414 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13354)) := by
  have c86886 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.1
  have c86887 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.1
  have c86888 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.1
  have c86889 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.1
  have c86890 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86891 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1414_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_18
  simp only [← c86886, ← c86887, ← c86888] at e_1414_18
  have e_1414_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1414 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1414_19
  simp only [← c86889, ← c86890, k2, ← c86891] at e_1414_19
  have hr := e_1414_19
  simp only [e_1414_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13355)) := by
  have c86892 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86893 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86894 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86895 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86896 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86897 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_0
  simp only [← c86892, ← c86893, ← c86894] at e_1415_0
  have e_1415_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_1
  simp only [← c86895, ← c86896, k2, ← c86897] at e_1415_1
  have hr := e_1415_1
  simp only [e_1415_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13356)) := by
  have c86898 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86899 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86900 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86901 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86902 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86903 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_2
  simp only [← c86898, ← c86899, ← c86900] at e_1415_2
  have e_1415_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_3
  simp only [← c86901, ← c86902, k2, ← c86903] at e_1415_3
  have hr := e_1415_3
  simp only [e_1415_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13357)) := by
  have c86904 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86905 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86906 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86907 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86908 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86909 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_4
  simp only [← c86904, ← c86905, ← c86906] at e_1415_4
  have e_1415_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_5
  simp only [← c86907, ← c86908, k2, ← c86909] at e_1415_5
  have hr := e_1415_5
  simp only [e_1415_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13358)) := by
  have c86910 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86911 := (publicBatchWrapper89_copies2715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86912 := (publicBatchWrapper89_copies2716 a h).1
  have c86913 := (publicBatchWrapper89_copies2716 a h).2.1
  have c86914 := (publicBatchWrapper89_copies2716 a h).2.2.1
  have c86915 := (publicBatchWrapper89_copies2716 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_6
  simp only [← c86910, ← c86911, ← c86912] at e_1415_6
  have e_1415_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_7
  simp only [← c86913, ← c86914, k2, ← c86915] at e_1415_7
  have hr := e_1415_7
  simp only [e_1415_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13359)) := by
  have c86916 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.1
  have c86917 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.1
  have c86918 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.1
  have c86919 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.1
  have c86920 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.1
  have c86921 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_8
  simp only [← c86916, ← c86917, ← c86918] at e_1415_8
  have e_1415_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_9
  simp only [← c86919, ← c86920, k2, ← c86921] at e_1415_9
  have hr := e_1415_9
  simp only [e_1415_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13360)) := by
  have c86922 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86923 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86924 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86925 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86926 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86927 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_10
  simp only [← c86922, ← c86923, ← c86924] at e_1415_10
  have e_1415_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_11
  simp only [← c86925, ← c86926, k2, ← c86927] at e_1415_11
  have hr := e_1415_11
  simp only [e_1415_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13361)) := by
  have c86928 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86929 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86930 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86931 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86932 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86933 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_12
  simp only [← c86928, ← c86929, ← c86930] at e_1415_12
  have e_1415_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_13
  simp only [← c86931, ← c86932, k2, ← c86933] at e_1415_13
  have hr := e_1415_13
  simp only [e_1415_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13362)) := by
  have c86934 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86935 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86936 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86937 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86938 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86939 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_14
  simp only [← c86934, ← c86935, ← c86936] at e_1415_14
  have e_1415_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_15
  simp only [← c86937, ← c86938, k2, ← c86939] at e_1415_15
  have hr := e_1415_15
  simp only [e_1415_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13363)) := by
  have c86940 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86941 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86942 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86943 := (publicBatchWrapper89_copies2716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c86944 := (publicBatchWrapper89_copies2717 a h).1
  have c86945 := (publicBatchWrapper89_copies2717 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_16
  simp only [← c86940, ← c86941, ← c86942] at e_1415_16
  have e_1415_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_17
  simp only [← c86943, ← c86944, k2, ← c86945] at e_1415_17
  have hr := e_1415_17
  simp only [e_1415_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1415 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13364)) := by
  have c86946 := (publicBatchWrapper89_copies2717 a h).2.2.1
  have c86947 := (publicBatchWrapper89_copies2717 a h).2.2.2.1
  have c86948 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.1
  have c86949 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.1
  have c86950 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.1
  have c86951 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1415_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_18
  simp only [← c86946, ← c86947, ← c86948] at e_1415_18
  have e_1415_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1415 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1415_19
  simp only [← c86949, ← c86950, k2, ← c86951] at e_1415_19
  have hr := e_1415_19
  simp only [e_1415_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13365)) := by
  have c86952 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.1
  have c86953 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.1
  have c86954 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86955 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c86956 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86957 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_0
  simp only [← c86952, ← c86953, ← c86954] at e_1416_0
  have e_1416_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_1
  simp only [← c86955, ← c86956, k2, ← c86957] at e_1416_1
  have hr := e_1416_1
  simp only [e_1416_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13366)) := by
  have c86958 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86959 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86960 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86961 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86962 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86963 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_2
  simp only [← c86958, ← c86959, ← c86960] at e_1416_2
  have e_1416_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_3
  simp only [← c86961, ← c86962, k2, ← c86963] at e_1416_3
  have hr := e_1416_3
  simp only [e_1416_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13367)) := by
  have c86964 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86965 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86966 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86967 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86968 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86969 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_4
  simp only [← c86964, ← c86965, ← c86966] at e_1416_4
  have e_1416_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_5
  simp only [← c86967, ← c86968, k2, ← c86969] at e_1416_5
  have hr := e_1416_5
  simp only [e_1416_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13368)) := by
  have c86970 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86971 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86972 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86973 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86974 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86975 := (publicBatchWrapper89_copies2717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_6
  simp only [← c86970, ← c86971, ← c86972] at e_1416_6
  have e_1416_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_7
  simp only [← c86973, ← c86974, k2, ← c86975] at e_1416_7
  have hr := e_1416_7
  simp only [e_1416_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13369)) := by
  have c86976 := (publicBatchWrapper89_copies2718 a h).1
  have c86977 := (publicBatchWrapper89_copies2718 a h).2.1
  have c86978 := (publicBatchWrapper89_copies2718 a h).2.2.1
  have c86979 := (publicBatchWrapper89_copies2718 a h).2.2.2.1
  have c86980 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.1
  have c86981 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_8
  simp only [← c86976, ← c86977, ← c86978] at e_1416_8
  have e_1416_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_9
  simp only [← c86979, ← c86980, k2, ← c86981] at e_1416_9
  have hr := e_1416_9
  simp only [e_1416_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13410)) := by
  have c86982 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.1
  have c86983 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.1
  have c86984 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.1
  have c86985 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.1
  have c86986 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c86987 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_10
  simp only [← c86982, ← c86983, ← c86984] at e_1416_10
  have e_1416_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_11
  simp only [← c86985, ← c86986, k2, ← c86987] at e_1416_11
  have hr := e_1416_11
  simp only [e_1416_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13411)) := by
  have c86988 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86989 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86990 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86991 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86992 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86993 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_12
  simp only [← c86988, ← c86989, ← c86990] at e_1416_12
  have e_1416_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_13
  simp only [← c86991, ← c86992, k2, ← c86993] at e_1416_13
  have hr := e_1416_13
  simp only [e_1416_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13412)) := by
  have c86994 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86995 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86996 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86997 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86998 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c86999 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_14
  simp only [← c86994, ← c86995, ← c86996] at e_1416_14
  have e_1416_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_15
  simp only [← c86997, ← c86998, k2, ← c86999] at e_1416_15
  have hr := e_1416_15
  simp only [e_1416_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13413)) := by
  have c87000 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87001 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87002 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87003 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87004 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87005 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_16
  simp only [← c87000, ← c87001, ← c87002] at e_1416_16
  have e_1416_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_17
  simp only [← c87003, ← c87004, k2, ← c87005] at e_1416_17
  have hr := e_1416_17
  simp only [e_1416_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1416 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13414)) := by
  have c87006 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87007 := (publicBatchWrapper89_copies2718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87008 := (publicBatchWrapper89_copies2719 a h).1
  have c87009 := (publicBatchWrapper89_copies2719 a h).2.1
  have c87010 := (publicBatchWrapper89_copies2719 a h).2.2.1
  have c87011 := (publicBatchWrapper89_copies2719 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1416_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_18
  simp only [← c87006, ← c87007, ← c87008] at e_1416_18
  have e_1416_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1416 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1416_19
  simp only [← c87009, ← c87010, k2, ← c87011] at e_1416_19
  have hr := e_1416_19
  simp only [e_1416_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13415)) := by
  have c87012 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.1
  have c87013 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.1
  have c87014 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.1
  have c87015 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.1
  have c87016 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.1
  have c87017 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_0
  simp only [← c87012, ← c87013, ← c87014] at e_1417_0
  have e_1417_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_1
  simp only [← c87015, ← c87016, k2, ← c87017] at e_1417_1
  have hr := e_1417_1
  simp only [e_1417_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13416)) := by
  have c87018 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87019 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87020 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87021 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87022 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87023 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_2
  simp only [← c87018, ← c87019, ← c87020] at e_1417_2
  have e_1417_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_3
  simp only [← c87021, ← c87022, k2, ← c87023] at e_1417_3
  have hr := e_1417_3
  simp only [e_1417_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13417)) := by
  have c87024 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87025 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87026 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87027 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87028 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87029 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_4
  simp only [← c87024, ← c87025, ← c87026] at e_1417_4
  have e_1417_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_5
  simp only [← c87027, ← c87028, k2, ← c87029] at e_1417_5
  have hr := e_1417_5
  simp only [e_1417_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13418)) := by
  have c87030 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87031 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87032 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87033 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87034 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87035 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_6
  simp only [← c87030, ← c87031, ← c87032] at e_1417_6
  have e_1417_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_7
  simp only [← c87033, ← c87034, k2, ← c87035] at e_1417_7
  have hr := e_1417_7
  simp only [e_1417_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13419)) := by
  have c87036 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87037 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87038 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87039 := (publicBatchWrapper89_copies2719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87040 := (publicBatchWrapper89_copies2720 a h).1
  have c87041 := (publicBatchWrapper89_copies2720 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_8
  simp only [← c87036, ← c87037, ← c87038] at e_1417_8
  have e_1417_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_9
  simp only [← c87039, ← c87040, k2, ← c87041] at e_1417_9
  have hr := e_1417_9
  simp only [e_1417_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13420)) := by
  have c87042 := (publicBatchWrapper89_copies2720 a h).2.2.1
  have c87043 := (publicBatchWrapper89_copies2720 a h).2.2.2.1
  have c87044 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.1
  have c87045 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.1
  have c87046 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.1
  have c87047 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_10
  simp only [← c87042, ← c87043, ← c87044] at e_1417_10
  have e_1417_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_11
  simp only [← c87045, ← c87046, k2, ← c87047] at e_1417_11
  have hr := e_1417_11
  simp only [e_1417_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13421)) := by
  have c87048 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.1
  have c87049 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.1
  have c87050 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87051 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87052 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87053 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_12
  simp only [← c87048, ← c87049, ← c87050] at e_1417_12
  have e_1417_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_13
  simp only [← c87051, ← c87052, k2, ← c87053] at e_1417_13
  have hr := e_1417_13
  simp only [e_1417_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13422)) := by
  have c87054 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87055 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87056 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87057 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87058 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87059 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_14
  simp only [← c87054, ← c87055, ← c87056] at e_1417_14
  have e_1417_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_15
  simp only [← c87057, ← c87058, k2, ← c87059] at e_1417_15
  have hr := e_1417_15
  simp only [e_1417_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13423)) := by
  have c87060 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87061 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87062 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87063 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87064 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87065 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_16
  simp only [← c87060, ← c87061, ← c87062] at e_1417_16
  have e_1417_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_17
  simp only [← c87063, ← c87064, k2, ← c87065] at e_1417_17
  have hr := e_1417_17
  simp only [e_1417_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1417 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13424)) := by
  have c87066 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87067 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87068 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87069 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87070 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87071 := (publicBatchWrapper89_copies2720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1417_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_18
  simp only [← c87066, ← c87067, ← c87068] at e_1417_18
  have e_1417_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1417 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1417_19
  simp only [← c87069, ← c87070, k2, ← c87071] at e_1417_19
  have hr := e_1417_19
  simp only [e_1417_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13425)) := by
  have c87072 := (publicBatchWrapper89_copies2721 a h).1
  have c87073 := (publicBatchWrapper89_copies2721 a h).2.1
  have c87074 := (publicBatchWrapper89_copies2721 a h).2.2.1
  have c87075 := (publicBatchWrapper89_copies2721 a h).2.2.2.1
  have c87076 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.1
  have c87077 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_0
  simp only [← c87072, ← c87073, ← c87074] at e_1418_0
  have e_1418_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_1
  simp only [← c87075, ← c87076, k2, ← c87077] at e_1418_1
  have hr := e_1418_1
  simp only [e_1418_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13426)) := by
  have c87078 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.1
  have c87079 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.1
  have c87080 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.1
  have c87081 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.1
  have c87082 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87083 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_2
  simp only [← c87078, ← c87079, ← c87080] at e_1418_2
  have e_1418_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_3
  simp only [← c87081, ← c87082, k2, ← c87083] at e_1418_3
  have hr := e_1418_3
  simp only [e_1418_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13427)) := by
  have c87084 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87085 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87086 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87087 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87088 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87089 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_4
  simp only [← c87084, ← c87085, ← c87086] at e_1418_4
  have e_1418_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_5
  simp only [← c87087, ← c87088, k2, ← c87089] at e_1418_5
  have hr := e_1418_5
  simp only [e_1418_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13428)) := by
  have c87090 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87091 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87092 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87093 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87094 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87095 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_6
  simp only [← c87090, ← c87091, ← c87092] at e_1418_6
  have e_1418_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_7
  simp only [← c87093, ← c87094, k2, ← c87095] at e_1418_7
  have hr := e_1418_7
  simp only [e_1418_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13429)) := by
  have c87096 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87097 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87098 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87099 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87100 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87101 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_8
  simp only [← c87096, ← c87097, ← c87098] at e_1418_8
  have e_1418_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_9
  simp only [← c87099, ← c87100, k2, ← c87101] at e_1418_9
  have hr := e_1418_9
  simp only [e_1418_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13430)) := by
  have c87102 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87103 := (publicBatchWrapper89_copies2721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87104 := (publicBatchWrapper89_copies2722 a h).1
  have c87105 := (publicBatchWrapper89_copies2722 a h).2.1
  have c87106 := (publicBatchWrapper89_copies2722 a h).2.2.1
  have c87107 := (publicBatchWrapper89_copies2722 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_10
  simp only [← c87102, ← c87103, ← c87104] at e_1418_10
  have e_1418_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_11
  simp only [← c87105, ← c87106, k2, ← c87107] at e_1418_11
  have hr := e_1418_11
  simp only [e_1418_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13431)) := by
  have c87108 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.1
  have c87109 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.1
  have c87110 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.1
  have c87111 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.1
  have c87112 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.1
  have c87113 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_12
  simp only [← c87108, ← c87109, ← c87110] at e_1418_12
  have e_1418_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_13
  simp only [← c87111, ← c87112, k2, ← c87113] at e_1418_13
  have hr := e_1418_13
  simp only [e_1418_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13432)) := by
  have c87114 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87115 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87116 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87117 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87118 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87119 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_14
  simp only [← c87114, ← c87115, ← c87116] at e_1418_14
  have e_1418_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_15
  simp only [← c87117, ← c87118, k2, ← c87119] at e_1418_15
  have hr := e_1418_15
  simp only [e_1418_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13433)) := by
  have c87120 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87121 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87122 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87123 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87124 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87125 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_16
  simp only [← c87120, ← c87121, ← c87122] at e_1418_16
  have e_1418_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_17
  simp only [← c87123, ← c87124, k2, ← c87125] at e_1418_17
  have hr := e_1418_17
  simp only [e_1418_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1418 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13434)) := by
  have c87126 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87127 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87128 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87129 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87130 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87131 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1418_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_18
  simp only [← c87126, ← c87127, ← c87128] at e_1418_18
  have e_1418_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1418 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1418_19
  simp only [← c87129, ← c87130, k2, ← c87131] at e_1418_19
  have hr := e_1418_19
  simp only [e_1418_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13435)) := by
  have c87132 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87133 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87134 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87135 := (publicBatchWrapper89_copies2722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87136 := (publicBatchWrapper89_copies2723 a h).1
  have c87137 := (publicBatchWrapper89_copies2723 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_0
  simp only [← c87132, ← c87133, ← c87134] at e_1419_0
  have e_1419_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_1
  simp only [← c87135, ← c87136, k2, ← c87137] at e_1419_1
  have hr := e_1419_1
  simp only [e_1419_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13436)) := by
  have c87138 := (publicBatchWrapper89_copies2723 a h).2.2.1
  have c87139 := (publicBatchWrapper89_copies2723 a h).2.2.2.1
  have c87140 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.1
  have c87141 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.1
  have c87142 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.1
  have c87143 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_2
  simp only [← c87138, ← c87139, ← c87140] at e_1419_2
  have e_1419_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_3
  simp only [← c87141, ← c87142, k2, ← c87143] at e_1419_3
  have hr := e_1419_3
  simp only [e_1419_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13437)) := by
  have c87144 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.1
  have c87145 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.1
  have c87146 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87147 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87148 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87149 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_4
  simp only [← c87144, ← c87145, ← c87146] at e_1419_4
  have e_1419_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_5
  simp only [← c87147, ← c87148, k2, ← c87149] at e_1419_5
  have hr := e_1419_5
  simp only [e_1419_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13438)) := by
  have c87150 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87151 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87152 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87153 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87154 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87155 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_6
  simp only [← c87150, ← c87151, ← c87152] at e_1419_6
  have e_1419_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_7
  simp only [← c87153, ← c87154, k2, ← c87155] at e_1419_7
  have hr := e_1419_7
  simp only [e_1419_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13439)) := by
  have c87156 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87157 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87158 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87159 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87160 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87161 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_8
  simp only [← c87156, ← c87157, ← c87158] at e_1419_8
  have e_1419_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_9
  simp only [← c87159, ← c87160, k2, ← c87161] at e_1419_9
  have hr := e_1419_9
  simp only [e_1419_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13440)) := by
  have c87162 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87163 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87164 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87165 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87166 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87167 := (publicBatchWrapper89_copies2723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_10
  simp only [← c87162, ← c87163, ← c87164] at e_1419_10
  have e_1419_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_11
  simp only [← c87165, ← c87166, k2, ← c87167] at e_1419_11
  have hr := e_1419_11
  simp only [e_1419_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13441)) := by
  have c87168 := (publicBatchWrapper89_copies2724 a h).1
  have c87169 := (publicBatchWrapper89_copies2724 a h).2.1
  have c87170 := (publicBatchWrapper89_copies2724 a h).2.2.1
  have c87171 := (publicBatchWrapper89_copies2724 a h).2.2.2.1
  have c87172 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.1
  have c87173 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_12
  simp only [← c87168, ← c87169, ← c87170] at e_1419_12
  have e_1419_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_13
  simp only [← c87171, ← c87172, k2, ← c87173] at e_1419_13
  have hr := e_1419_13
  simp only [e_1419_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13442)) := by
  have c87174 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.1
  have c87175 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.1
  have c87176 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.1
  have c87177 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.1
  have c87178 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87179 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_14
  simp only [← c87174, ← c87175, ← c87176] at e_1419_14
  have e_1419_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_15
  simp only [← c87177, ← c87178, k2, ← c87179] at e_1419_15
  have hr := e_1419_15
  simp only [e_1419_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13443)) := by
  have c87180 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87181 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87182 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87183 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87184 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87185 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_16
  simp only [← c87180, ← c87181, ← c87182] at e_1419_16
  have e_1419_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_17
  simp only [← c87183, ← c87184, k2, ← c87185] at e_1419_17
  have hr := e_1419_17
  simp only [e_1419_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1419 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13444)) := by
  have c87186 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87187 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87188 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87189 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87190 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87191 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1419_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_18
  simp only [← c87186, ← c87187, ← c87188] at e_1419_18
  have e_1419_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1419 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1419_19
  simp only [← c87189, ← c87190, k2, ← c87191] at e_1419_19
  have hr := e_1419_19
  simp only [e_1419_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13445)) := by
  have c87192 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87193 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87194 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87195 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87196 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87197 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_0
  simp only [← c87192, ← c87193, ← c87194] at e_1420_0
  have e_1420_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_1
  simp only [← c87195, ← c87196, k2, ← c87197] at e_1420_1
  have hr := e_1420_1
  simp only [e_1420_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13446)) := by
  have c87198 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87199 := (publicBatchWrapper89_copies2724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87200 := (publicBatchWrapper89_copies2725 a h).1
  have c87201 := (publicBatchWrapper89_copies2725 a h).2.1
  have c87202 := (publicBatchWrapper89_copies2725 a h).2.2.1
  have c87203 := (publicBatchWrapper89_copies2725 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_2
  simp only [← c87198, ← c87199, ← c87200] at e_1420_2
  have e_1420_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_3
  simp only [← c87201, ← c87202, k2, ← c87203] at e_1420_3
  have hr := e_1420_3
  simp only [e_1420_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13447)) := by
  have c87204 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.1
  have c87205 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.1
  have c87206 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.1
  have c87207 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.1
  have c87208 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.1
  have c87209 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_4
  simp only [← c87204, ← c87205, ← c87206] at e_1420_4
  have e_1420_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_5
  simp only [← c87207, ← c87208, k2, ← c87209] at e_1420_5
  have hr := e_1420_5
  simp only [e_1420_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13448)) := by
  have c87210 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87211 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87212 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87213 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87214 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87215 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_6
  simp only [← c87210, ← c87211, ← c87212] at e_1420_6
  have e_1420_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_7
  simp only [← c87213, ← c87214, k2, ← c87215] at e_1420_7
  have hr := e_1420_7
  simp only [e_1420_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13449)) := by
  have c87216 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87217 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87218 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87219 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87220 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87221 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_8
  simp only [← c87216, ← c87217, ← c87218] at e_1420_8
  have e_1420_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_9
  simp only [← c87219, ← c87220, k2, ← c87221] at e_1420_9
  have hr := e_1420_9
  simp only [e_1420_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13450)) := by
  have c87222 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87223 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87224 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87225 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87226 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87227 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_10
  simp only [← c87222, ← c87223, ← c87224] at e_1420_10
  have e_1420_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_11
  simp only [← c87225, ← c87226, k2, ← c87227] at e_1420_11
  have hr := e_1420_11
  simp only [e_1420_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13451)) := by
  have c87228 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87229 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87230 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87231 := (publicBatchWrapper89_copies2725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87232 := (publicBatchWrapper89_copies2726 a h).1
  have c87233 := (publicBatchWrapper89_copies2726 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_12
  simp only [← c87228, ← c87229, ← c87230] at e_1420_12
  have e_1420_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_13
  simp only [← c87231, ← c87232, k2, ← c87233] at e_1420_13
  have hr := e_1420_13
  simp only [e_1420_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13452)) := by
  have c87234 := (publicBatchWrapper89_copies2726 a h).2.2.1
  have c87235 := (publicBatchWrapper89_copies2726 a h).2.2.2.1
  have c87236 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.1
  have c87237 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.1
  have c87238 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.1
  have c87239 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_14
  simp only [← c87234, ← c87235, ← c87236] at e_1420_14
  have e_1420_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_15
  simp only [← c87237, ← c87238, k2, ← c87239] at e_1420_15
  have hr := e_1420_15
  simp only [e_1420_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13453)) := by
  have c87240 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.1
  have c87241 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.1
  have c87242 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87243 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87244 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87245 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_16
  simp only [← c87240, ← c87241, ← c87242] at e_1420_16
  have e_1420_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_17
  simp only [← c87243, ← c87244, k2, ← c87245] at e_1420_17
  have hr := e_1420_17
  simp only [e_1420_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1420 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13454)) := by
  have c87246 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87247 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87248 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87249 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87250 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87251 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1420_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_18
  simp only [← c87246, ← c87247, ← c87248] at e_1420_18
  have e_1420_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1420 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1420_19
  simp only [← c87249, ← c87250, k2, ← c87251] at e_1420_19
  have hr := e_1420_19
  simp only [e_1420_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13455)) := by
  have c87252 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87253 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87254 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87255 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87256 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87257 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_0
  simp only [← c87252, ← c87253, ← c87254] at e_1421_0
  have e_1421_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_1
  simp only [← c87255, ← c87256, k2, ← c87257] at e_1421_1
  have hr := e_1421_1
  simp only [e_1421_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13456)) := by
  have c87258 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87259 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87260 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87261 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87262 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87263 := (publicBatchWrapper89_copies2726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_2
  simp only [← c87258, ← c87259, ← c87260] at e_1421_2
  have e_1421_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_3
  simp only [← c87261, ← c87262, k2, ← c87263] at e_1421_3
  have hr := e_1421_3
  simp only [e_1421_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13457)) := by
  have c87264 := (publicBatchWrapper89_copies2727 a h).1
  have c87265 := (publicBatchWrapper89_copies2727 a h).2.1
  have c87266 := (publicBatchWrapper89_copies2727 a h).2.2.1
  have c87267 := (publicBatchWrapper89_copies2727 a h).2.2.2.1
  have c87268 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.1
  have c87269 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_4
  simp only [← c87264, ← c87265, ← c87266] at e_1421_4
  have e_1421_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_5
  simp only [← c87267, ← c87268, k2, ← c87269] at e_1421_5
  have hr := e_1421_5
  simp only [e_1421_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13458)) := by
  have c87270 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.1
  have c87271 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.1
  have c87272 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.1
  have c87273 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.1
  have c87274 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87275 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_6
  simp only [← c87270, ← c87271, ← c87272] at e_1421_6
  have e_1421_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_7
  simp only [← c87273, ← c87274, k2, ← c87275] at e_1421_7
  have hr := e_1421_7
  simp only [e_1421_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13459)) := by
  have c87276 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87277 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87278 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87279 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87280 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87281 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_8
  simp only [← c87276, ← c87277, ← c87278] at e_1421_8
  have e_1421_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_9
  simp only [← c87279, ← c87280, k2, ← c87281] at e_1421_9
  have hr := e_1421_9
  simp only [e_1421_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13460)) := by
  have c87282 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87283 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87284 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87285 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87286 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87287 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_10
  simp only [← c87282, ← c87283, ← c87284] at e_1421_10
  have e_1421_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_11
  simp only [← c87285, ← c87286, k2, ← c87287] at e_1421_11
  have hr := e_1421_11
  simp only [e_1421_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13461)) := by
  have c87288 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87289 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87290 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87291 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87292 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87293 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_12
  simp only [← c87288, ← c87289, ← c87290] at e_1421_12
  have e_1421_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_13
  simp only [← c87291, ← c87292, k2, ← c87293] at e_1421_13
  have hr := e_1421_13
  simp only [e_1421_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13462)) := by
  have c87294 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87295 := (publicBatchWrapper89_copies2727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87296 := (publicBatchWrapper89_copies2728 a h).1
  have c87297 := (publicBatchWrapper89_copies2728 a h).2.1
  have c87298 := (publicBatchWrapper89_copies2728 a h).2.2.1
  have c87299 := (publicBatchWrapper89_copies2728 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_14
  simp only [← c87294, ← c87295, ← c87296] at e_1421_14
  have e_1421_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_15
  simp only [← c87297, ← c87298, k2, ← c87299] at e_1421_15
  have hr := e_1421_15
  simp only [e_1421_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13463)) := by
  have c87300 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.1
  have c87301 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.1
  have c87302 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.1
  have c87303 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.1
  have c87304 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.1
  have c87305 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_16
  simp only [← c87300, ← c87301, ← c87302] at e_1421_16
  have e_1421_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_17
  simp only [← c87303, ← c87304, k2, ← c87305] at e_1421_17
  have hr := e_1421_17
  simp only [e_1421_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1421 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13464)) := by
  have c87306 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87307 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87308 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87309 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87310 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87311 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1421_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_18
  simp only [← c87306, ← c87307, ← c87308] at e_1421_18
  have e_1421_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1421 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1421_19
  simp only [← c87309, ← c87310, k2, ← c87311] at e_1421_19
  have hr := e_1421_19
  simp only [e_1421_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13465)) := by
  have c87312 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87313 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87314 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87315 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87316 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87317 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_0
  simp only [← c87312, ← c87313, ← c87314] at e_1422_0
  have e_1422_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_1
  simp only [← c87315, ← c87316, k2, ← c87317] at e_1422_1
  have hr := e_1422_1
  simp only [e_1422_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13466)) := by
  have c87318 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87319 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87320 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87321 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87322 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87323 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_2
  simp only [← c87318, ← c87319, ← c87320] at e_1422_2
  have e_1422_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_3
  simp only [← c87321, ← c87322, k2, ← c87323] at e_1422_3
  have hr := e_1422_3
  simp only [e_1422_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13467)) := by
  have c87324 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87325 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87326 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87327 := (publicBatchWrapper89_copies2728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87328 := (publicBatchWrapper89_copies2729 a h).1
  have c87329 := (publicBatchWrapper89_copies2729 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_4
  simp only [← c87324, ← c87325, ← c87326] at e_1422_4
  have e_1422_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_5
  simp only [← c87327, ← c87328, k2, ← c87329] at e_1422_5
  have hr := e_1422_5
  simp only [e_1422_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13468)) := by
  have c87330 := (publicBatchWrapper89_copies2729 a h).2.2.1
  have c87331 := (publicBatchWrapper89_copies2729 a h).2.2.2.1
  have c87332 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.1
  have c87333 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.1
  have c87334 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.1
  have c87335 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_6
  simp only [← c87330, ← c87331, ← c87332] at e_1422_6
  have e_1422_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_7
  simp only [← c87333, ← c87334, k2, ← c87335] at e_1422_7
  have hr := e_1422_7
  simp only [e_1422_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13469)) := by
  have c87336 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.1
  have c87337 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.1
  have c87338 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87339 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87340 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87341 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_8
  simp only [← c87336, ← c87337, ← c87338] at e_1422_8
  have e_1422_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_9
  simp only [← c87339, ← c87340, k2, ← c87341] at e_1422_9
  have hr := e_1422_9
  simp only [e_1422_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13470)) := by
  have c87342 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87343 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87344 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87345 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87346 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87347 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_10
  simp only [← c87342, ← c87343, ← c87344] at e_1422_10
  have e_1422_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_11
  simp only [← c87345, ← c87346, k2, ← c87347] at e_1422_11
  have hr := e_1422_11
  simp only [e_1422_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13471)) := by
  have c87348 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87349 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87350 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87351 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87352 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87353 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_12
  simp only [← c87348, ← c87349, ← c87350] at e_1422_12
  have e_1422_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_13
  simp only [← c87351, ← c87352, k2, ← c87353] at e_1422_13
  have hr := e_1422_13
  simp only [e_1422_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13472)) := by
  have c87354 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87355 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87356 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87357 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87358 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87359 := (publicBatchWrapper89_copies2729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_14
  simp only [← c87354, ← c87355, ← c87356] at e_1422_14
  have e_1422_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_15
  simp only [← c87357, ← c87358, k2, ← c87359] at e_1422_15
  have hr := e_1422_15
  simp only [e_1422_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13473)) := by
  have c87360 := (publicBatchWrapper89_copies2730 a h).1
  have c87361 := (publicBatchWrapper89_copies2730 a h).2.1
  have c87362 := (publicBatchWrapper89_copies2730 a h).2.2.1
  have c87363 := (publicBatchWrapper89_copies2730 a h).2.2.2.1
  have c87364 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.1
  have c87365 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_16
  simp only [← c87360, ← c87361, ← c87362] at e_1422_16
  have e_1422_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_17
  simp only [← c87363, ← c87364, k2, ← c87365] at e_1422_17
  have hr := e_1422_17
  simp only [e_1422_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1422 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13474)) := by
  have c87366 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.1
  have c87367 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.1
  have c87368 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.1
  have c87369 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.1
  have c87370 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87371 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1422_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_18
  simp only [← c87366, ← c87367, ← c87368] at e_1422_18
  have e_1422_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1422 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1422_19
  simp only [← c87369, ← c87370, k2, ← c87371] at e_1422_19
  have hr := e_1422_19
  simp only [e_1422_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13475)) := by
  have c87372 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87373 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87374 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87375 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87376 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87377 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_0
  simp only [← c87372, ← c87373, ← c87374] at e_1423_0
  have e_1423_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_1
  simp only [← c87375, ← c87376, k2, ← c87377] at e_1423_1
  have hr := e_1423_1
  simp only [e_1423_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13476)) := by
  have c87378 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87379 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87380 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87381 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87382 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87383 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_2
  simp only [← c87378, ← c87379, ← c87380] at e_1423_2
  have e_1423_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_3
  simp only [← c87381, ← c87382, k2, ← c87383] at e_1423_3
  have hr := e_1423_3
  simp only [e_1423_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13477)) := by
  have c87384 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87385 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87386 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87387 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87388 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87389 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_4
  simp only [← c87384, ← c87385, ← c87386] at e_1423_4
  have e_1423_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_5
  simp only [← c87387, ← c87388, k2, ← c87389] at e_1423_5
  have hr := e_1423_5
  simp only [e_1423_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13478)) := by
  have c87390 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87391 := (publicBatchWrapper89_copies2730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87392 := (publicBatchWrapper89_copies2731 a h).1
  have c87393 := (publicBatchWrapper89_copies2731 a h).2.1
  have c87394 := (publicBatchWrapper89_copies2731 a h).2.2.1
  have c87395 := (publicBatchWrapper89_copies2731 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_6
  simp only [← c87390, ← c87391, ← c87392] at e_1423_6
  have e_1423_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_7
  simp only [← c87393, ← c87394, k2, ← c87395] at e_1423_7
  have hr := e_1423_7
  simp only [e_1423_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13479)) := by
  have c87396 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.1
  have c87397 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.1
  have c87398 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.1
  have c87399 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.1
  have c87400 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.1
  have c87401 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_8
  simp only [← c87396, ← c87397, ← c87398] at e_1423_8
  have e_1423_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_9
  simp only [← c87399, ← c87400, k2, ← c87401] at e_1423_9
  have hr := e_1423_9
  simp only [e_1423_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13480)) := by
  have c87402 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87403 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87404 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87405 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87406 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87407 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_10
  simp only [← c87402, ← c87403, ← c87404] at e_1423_10
  have e_1423_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_11
  simp only [← c87405, ← c87406, k2, ← c87407] at e_1423_11
  have hr := e_1423_11
  simp only [e_1423_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13481)) := by
  have c87408 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87409 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87410 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87411 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87412 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87413 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_12
  simp only [← c87408, ← c87409, ← c87410] at e_1423_12
  have e_1423_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_13
  simp only [← c87411, ← c87412, k2, ← c87413] at e_1423_13
  have hr := e_1423_13
  simp only [e_1423_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13482)) := by
  have c87414 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87415 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87416 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87417 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87418 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87419 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_14
  simp only [← c87414, ← c87415, ← c87416] at e_1423_14
  have e_1423_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_15
  simp only [← c87417, ← c87418, k2, ← c87419] at e_1423_15
  have hr := e_1423_15
  simp only [e_1423_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13483)) := by
  have c87420 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87421 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87422 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87423 := (publicBatchWrapper89_copies2731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87424 := (publicBatchWrapper89_copies2732 a h).1
  have c87425 := (publicBatchWrapper89_copies2732 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_16
  simp only [← c87420, ← c87421, ← c87422] at e_1423_16
  have e_1423_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_17
  simp only [← c87423, ← c87424, k2, ← c87425] at e_1423_17
  have hr := e_1423_17
  simp only [e_1423_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1423 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13484)) := by
  have c87426 := (publicBatchWrapper89_copies2732 a h).2.2.1
  have c87427 := (publicBatchWrapper89_copies2732 a h).2.2.2.1
  have c87428 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.1
  have c87429 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.1
  have c87430 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.1
  have c87431 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1423_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_18
  simp only [← c87426, ← c87427, ← c87428] at e_1423_18
  have e_1423_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1423 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1423_19
  simp only [← c87429, ← c87430, k2, ← c87431] at e_1423_19
  have hr := e_1423_19
  simp only [e_1423_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13485)) := by
  have c87432 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.1
  have c87433 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.1
  have c87434 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87435 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87436 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87437 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_0
  simp only [← c87432, ← c87433, ← c87434] at e_1424_0
  have e_1424_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_1
  simp only [← c87435, ← c87436, k2, ← c87437] at e_1424_1
  have hr := e_1424_1
  simp only [e_1424_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13486)) := by
  have c87438 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87439 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87440 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87441 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87442 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87443 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_2
  simp only [← c87438, ← c87439, ← c87440] at e_1424_2
  have e_1424_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_3
  simp only [← c87441, ← c87442, k2, ← c87443] at e_1424_3
  have hr := e_1424_3
  simp only [e_1424_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13487)) := by
  have c87444 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87445 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87446 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87447 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87448 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87449 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_4
  simp only [← c87444, ← c87445, ← c87446] at e_1424_4
  have e_1424_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_5
  simp only [← c87447, ← c87448, k2, ← c87449] at e_1424_5
  have hr := e_1424_5
  simp only [e_1424_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13488)) := by
  have c87450 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87451 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87452 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87453 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87454 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87455 := (publicBatchWrapper89_copies2732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_6
  simp only [← c87450, ← c87451, ← c87452] at e_1424_6
  have e_1424_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_7
  simp only [← c87453, ← c87454, k2, ← c87455] at e_1424_7
  have hr := e_1424_7
  simp only [e_1424_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13489)) := by
  have c87456 := (publicBatchWrapper89_copies2733 a h).1
  have c87457 := (publicBatchWrapper89_copies2733 a h).2.1
  have c87458 := (publicBatchWrapper89_copies2733 a h).2.2.1
  have c87459 := (publicBatchWrapper89_copies2733 a h).2.2.2.1
  have c87460 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.1
  have c87461 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_8
  simp only [← c87456, ← c87457, ← c87458] at e_1424_8
  have e_1424_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_9
  simp only [← c87459, ← c87460, k2, ← c87461] at e_1424_9
  have hr := e_1424_9
  simp only [e_1424_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13490)) := by
  have c87462 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.1
  have c87463 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.1
  have c87464 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.1
  have c87465 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.1
  have c87466 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87467 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_10
  simp only [← c87462, ← c87463, ← c87464] at e_1424_10
  have e_1424_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_11
  simp only [← c87465, ← c87466, k2, ← c87467] at e_1424_11
  have hr := e_1424_11
  simp only [e_1424_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13491)) := by
  have c87468 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87469 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87470 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87471 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87472 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87473 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_12
  simp only [← c87468, ← c87469, ← c87470] at e_1424_12
  have e_1424_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_13
  simp only [← c87471, ← c87472, k2, ← c87473] at e_1424_13
  have hr := e_1424_13
  simp only [e_1424_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13492)) := by
  have c87474 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87475 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87476 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87477 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87478 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87479 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_14
  simp only [← c87474, ← c87475, ← c87476] at e_1424_14
  have e_1424_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_15
  simp only [← c87477, ← c87478, k2, ← c87479] at e_1424_15
  have hr := e_1424_15
  simp only [e_1424_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13493)) := by
  have c87480 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87481 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87482 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87483 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87484 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87485 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_16
  simp only [← c87480, ← c87481, ← c87482] at e_1424_16
  have e_1424_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_17
  simp only [← c87483, ← c87484, k2, ← c87485] at e_1424_17
  have hr := e_1424_17
  simp only [e_1424_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1424 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13494)) := by
  have c87486 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87487 := (publicBatchWrapper89_copies2733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87488 := (publicBatchWrapper89_copies2734 a h).1
  have c87489 := (publicBatchWrapper89_copies2734 a h).2.1
  have c87490 := (publicBatchWrapper89_copies2734 a h).2.2.1
  have c87491 := (publicBatchWrapper89_copies2734 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1424_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_18
  simp only [← c87486, ← c87487, ← c87488] at e_1424_18
  have e_1424_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1424 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1424_19
  simp only [← c87489, ← c87490, k2, ← c87491] at e_1424_19
  have hr := e_1424_19
  simp only [e_1424_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13495)) := by
  have c87492 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.1
  have c87493 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.1
  have c87494 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.1
  have c87495 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.1
  have c87496 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.1
  have c87497 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_0
  simp only [← c87492, ← c87493, ← c87494] at e_1425_0
  have e_1425_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_1
  simp only [← c87495, ← c87496, k2, ← c87497] at e_1425_1
  have hr := e_1425_1
  simp only [e_1425_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13496)) := by
  have c87498 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87499 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87500 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87501 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87502 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87503 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_2
  simp only [← c87498, ← c87499, ← c87500] at e_1425_2
  have e_1425_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_3
  simp only [← c87501, ← c87502, k2, ← c87503] at e_1425_3
  have hr := e_1425_3
  simp only [e_1425_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13497)) := by
  have c87504 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87505 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87506 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87507 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87508 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87509 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_4
  simp only [← c87504, ← c87505, ← c87506] at e_1425_4
  have e_1425_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_5
  simp only [← c87507, ← c87508, k2, ← c87509] at e_1425_5
  have hr := e_1425_5
  simp only [e_1425_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13498)) := by
  have c87510 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87511 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87512 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87513 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87514 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87515 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_6
  simp only [← c87510, ← c87511, ← c87512] at e_1425_6
  have e_1425_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_7
  simp only [← c87513, ← c87514, k2, ← c87515] at e_1425_7
  have hr := e_1425_7
  simp only [e_1425_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13499)) := by
  have c87516 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87517 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87518 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87519 := (publicBatchWrapper89_copies2734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87520 := (publicBatchWrapper89_copies2735 a h).1
  have c87521 := (publicBatchWrapper89_copies2735 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_8
  simp only [← c87516, ← c87517, ← c87518] at e_1425_8
  have e_1425_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_9
  simp only [← c87519, ← c87520, k2, ← c87521] at e_1425_9
  have hr := e_1425_9
  simp only [e_1425_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13500)) := by
  have c87522 := (publicBatchWrapper89_copies2735 a h).2.2.1
  have c87523 := (publicBatchWrapper89_copies2735 a h).2.2.2.1
  have c87524 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.1
  have c87525 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.1
  have c87526 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.1
  have c87527 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_10
  simp only [← c87522, ← c87523, ← c87524] at e_1425_10
  have e_1425_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_11
  simp only [← c87525, ← c87526, k2, ← c87527] at e_1425_11
  have hr := e_1425_11
  simp only [e_1425_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13501)) := by
  have c87528 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.1
  have c87529 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.1
  have c87530 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87531 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87532 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87533 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_12
  simp only [← c87528, ← c87529, ← c87530] at e_1425_12
  have e_1425_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_13
  simp only [← c87531, ← c87532, k2, ← c87533] at e_1425_13
  have hr := e_1425_13
  simp only [e_1425_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13502)) := by
  have c87534 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87535 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87536 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87537 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87538 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87539 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_14
  simp only [← c87534, ← c87535, ← c87536] at e_1425_14
  have e_1425_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_15
  simp only [← c87537, ← c87538, k2, ← c87539] at e_1425_15
  have hr := e_1425_15
  simp only [e_1425_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13503)) := by
  have c87540 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87541 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87542 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87543 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87544 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87545 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_16
  simp only [← c87540, ← c87541, ← c87542] at e_1425_16
  have e_1425_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_17
  simp only [← c87543, ← c87544, k2, ← c87545] at e_1425_17
  have hr := e_1425_17
  simp only [e_1425_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1425 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13504)) := by
  have c87546 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87547 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87548 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87549 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87550 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87551 := (publicBatchWrapper89_copies2735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1425_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_18
  simp only [← c87546, ← c87547, ← c87548] at e_1425_18
  have e_1425_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1425 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1425_19
  simp only [← c87549, ← c87550, k2, ← c87551] at e_1425_19
  have hr := e_1425_19
  simp only [e_1425_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13505)) := by
  have c87552 := (publicBatchWrapper89_copies2736 a h).1
  have c87553 := (publicBatchWrapper89_copies2736 a h).2.1
  have c87554 := (publicBatchWrapper89_copies2736 a h).2.2.1
  have c87555 := (publicBatchWrapper89_copies2736 a h).2.2.2.1
  have c87556 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.1
  have c87557 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_0
  simp only [← c87552, ← c87553, ← c87554] at e_1426_0
  have e_1426_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_1
  simp only [← c87555, ← c87556, k2, ← c87557] at e_1426_1
  have hr := e_1426_1
  simp only [e_1426_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13506)) := by
  have c87558 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.1
  have c87559 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.1
  have c87560 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.1
  have c87561 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.1
  have c87562 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87563 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_2
  simp only [← c87558, ← c87559, ← c87560] at e_1426_2
  have e_1426_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_3
  simp only [← c87561, ← c87562, k2, ← c87563] at e_1426_3
  have hr := e_1426_3
  simp only [e_1426_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13507)) := by
  have c87564 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87565 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87566 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87567 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87568 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87569 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_4
  simp only [← c87564, ← c87565, ← c87566] at e_1426_4
  have e_1426_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_5
  simp only [← c87567, ← c87568, k2, ← c87569] at e_1426_5
  have hr := e_1426_5
  simp only [e_1426_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13508)) := by
  have c87570 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87571 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87572 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87573 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87574 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87575 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_6
  simp only [← c87570, ← c87571, ← c87572] at e_1426_6
  have e_1426_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_7
  simp only [← c87573, ← c87574, k2, ← c87575] at e_1426_7
  have hr := e_1426_7
  simp only [e_1426_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13509)) := by
  have c87576 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87577 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87578 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87579 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87580 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87581 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_8
  simp only [← c87576, ← c87577, ← c87578] at e_1426_8
  have e_1426_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_9
  simp only [← c87579, ← c87580, k2, ← c87581] at e_1426_9
  have hr := e_1426_9
  simp only [e_1426_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13510)) := by
  have c87582 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87583 := (publicBatchWrapper89_copies2736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87584 := (publicBatchWrapper89_copies2737 a h).1
  have c87585 := (publicBatchWrapper89_copies2737 a h).2.1
  have c87586 := (publicBatchWrapper89_copies2737 a h).2.2.1
  have c87587 := (publicBatchWrapper89_copies2737 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_10
  simp only [← c87582, ← c87583, ← c87584] at e_1426_10
  have e_1426_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_11
  simp only [← c87585, ← c87586, k2, ← c87587] at e_1426_11
  have hr := e_1426_11
  simp only [e_1426_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13511)) := by
  have c87588 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.1
  have c87589 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.1
  have c87590 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.1
  have c87591 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.1
  have c87592 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.1
  have c87593 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_12
  simp only [← c87588, ← c87589, ← c87590] at e_1426_12
  have e_1426_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_13
  simp only [← c87591, ← c87592, k2, ← c87593] at e_1426_13
  have hr := e_1426_13
  simp only [e_1426_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13512)) := by
  have c87594 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87595 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87596 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87597 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87598 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87599 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_14
  simp only [← c87594, ← c87595, ← c87596] at e_1426_14
  have e_1426_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_15
  simp only [← c87597, ← c87598, k2, ← c87599] at e_1426_15
  have hr := e_1426_15
  simp only [e_1426_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13513)) := by
  have c87600 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87601 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87602 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87603 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87604 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87605 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_16
  simp only [← c87600, ← c87601, ← c87602] at e_1426_16
  have e_1426_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_17
  simp only [← c87603, ← c87604, k2, ← c87605] at e_1426_17
  have hr := e_1426_17
  simp only [e_1426_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1426 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13514)) := by
  have c87606 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87607 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87608 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87609 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87610 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87611 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1426_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_18
  simp only [← c87606, ← c87607, ← c87608] at e_1426_18
  have e_1426_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1426 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1426_19
  simp only [← c87609, ← c87610, k2, ← c87611] at e_1426_19
  have hr := e_1426_19
  simp only [e_1426_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13515)) := by
  have c87612 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87613 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87614 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87615 := (publicBatchWrapper89_copies2737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87616 := (publicBatchWrapper89_copies2738 a h).1
  have c87617 := (publicBatchWrapper89_copies2738 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_0
  simp only [← c87612, ← c87613, ← c87614] at e_1427_0
  have e_1427_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_1
  simp only [← c87615, ← c87616, k2, ← c87617] at e_1427_1
  have hr := e_1427_1
  simp only [e_1427_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13516)) := by
  have c87618 := (publicBatchWrapper89_copies2738 a h).2.2.1
  have c87619 := (publicBatchWrapper89_copies2738 a h).2.2.2.1
  have c87620 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.1
  have c87621 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.1
  have c87622 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.1
  have c87623 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_2
  simp only [← c87618, ← c87619, ← c87620] at e_1427_2
  have e_1427_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_3
  simp only [← c87621, ← c87622, k2, ← c87623] at e_1427_3
  have hr := e_1427_3
  simp only [e_1427_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13517)) := by
  have c87624 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.1
  have c87625 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.1
  have c87626 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87627 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87628 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87629 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_4
  simp only [← c87624, ← c87625, ← c87626] at e_1427_4
  have e_1427_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_5
  simp only [← c87627, ← c87628, k2, ← c87629] at e_1427_5
  have hr := e_1427_5
  simp only [e_1427_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13518)) := by
  have c87630 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87631 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87632 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87633 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87634 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87635 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_6
  simp only [← c87630, ← c87631, ← c87632] at e_1427_6
  have e_1427_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_7
  simp only [← c87633, ← c87634, k2, ← c87635] at e_1427_7
  have hr := e_1427_7
  simp only [e_1427_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13519)) := by
  have c87636 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87637 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87638 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87639 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87640 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87641 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_8
  simp only [← c87636, ← c87637, ← c87638] at e_1427_8
  have e_1427_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_9
  simp only [← c87639, ← c87640, k2, ← c87641] at e_1427_9
  have hr := e_1427_9
  simp only [e_1427_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13520)) := by
  have c87642 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87643 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87644 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87645 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87646 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87647 := (publicBatchWrapper89_copies2738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_10
  simp only [← c87642, ← c87643, ← c87644] at e_1427_10
  have e_1427_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_11
  simp only [← c87645, ← c87646, k2, ← c87647] at e_1427_11
  have hr := e_1427_11
  simp only [e_1427_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13521)) := by
  have c87648 := (publicBatchWrapper89_copies2739 a h).1
  have c87649 := (publicBatchWrapper89_copies2739 a h).2.1
  have c87650 := (publicBatchWrapper89_copies2739 a h).2.2.1
  have c87651 := (publicBatchWrapper89_copies2739 a h).2.2.2.1
  have c87652 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.1
  have c87653 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_12
  simp only [← c87648, ← c87649, ← c87650] at e_1427_12
  have e_1427_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_13
  simp only [← c87651, ← c87652, k2, ← c87653] at e_1427_13
  have hr := e_1427_13
  simp only [e_1427_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13522)) := by
  have c87654 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.1
  have c87655 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.1
  have c87656 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.1
  have c87657 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.1
  have c87658 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87659 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_14
  simp only [← c87654, ← c87655, ← c87656] at e_1427_14
  have e_1427_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_15
  simp only [← c87657, ← c87658, k2, ← c87659] at e_1427_15
  have hr := e_1427_15
  simp only [e_1427_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13523)) := by
  have c87660 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87661 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87662 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87663 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87664 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87665 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_16
  simp only [← c87660, ← c87661, ← c87662] at e_1427_16
  have e_1427_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_17
  simp only [← c87663, ← c87664, k2, ← c87665] at e_1427_17
  have hr := e_1427_17
  simp only [e_1427_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1427 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13524)) := by
  have c87666 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87667 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87668 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87669 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87670 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87671 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1427_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_18
  simp only [← c87666, ← c87667, ← c87668] at e_1427_18
  have e_1427_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1427 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1427_19
  simp only [← c87669, ← c87670, k2, ← c87671] at e_1427_19
  have hr := e_1427_19
  simp only [e_1427_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13525)) := by
  have c87672 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87673 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87674 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87675 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87676 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87677 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_0
  simp only [← c87672, ← c87673, ← c87674] at e_1428_0
  have e_1428_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_1
  simp only [← c87675, ← c87676, k2, ← c87677] at e_1428_1
  have hr := e_1428_1
  simp only [e_1428_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13526)) := by
  have c87678 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87679 := (publicBatchWrapper89_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87680 := (publicBatchWrapper89_copies2740 a h).1
  have c87681 := (publicBatchWrapper89_copies2740 a h).2.1
  have c87682 := (publicBatchWrapper89_copies2740 a h).2.2.1
  have c87683 := (publicBatchWrapper89_copies2740 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_2
  simp only [← c87678, ← c87679, ← c87680] at e_1428_2
  have e_1428_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_3
  simp only [← c87681, ← c87682, k2, ← c87683] at e_1428_3
  have hr := e_1428_3
  simp only [e_1428_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13527)) := by
  have c87684 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.1
  have c87685 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.1
  have c87686 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.1
  have c87687 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.1
  have c87688 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.1
  have c87689 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_4
  simp only [← c87684, ← c87685, ← c87686] at e_1428_4
  have e_1428_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_5
  simp only [← c87687, ← c87688, k2, ← c87689] at e_1428_5
  have hr := e_1428_5
  simp only [e_1428_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13528)) := by
  have c87690 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87691 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87692 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87693 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87694 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87695 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_6
  simp only [← c87690, ← c87691, ← c87692] at e_1428_6
  have e_1428_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_7
  simp only [← c87693, ← c87694, k2, ← c87695] at e_1428_7
  have hr := e_1428_7
  simp only [e_1428_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13529)) := by
  have c87696 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87697 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87698 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87699 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87700 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87701 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_8
  simp only [← c87696, ← c87697, ← c87698] at e_1428_8
  have e_1428_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_9
  simp only [← c87699, ← c87700, k2, ← c87701] at e_1428_9
  have hr := e_1428_9
  simp only [e_1428_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13530)) := by
  have c87702 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87703 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87704 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87705 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87706 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87707 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_10
  simp only [← c87702, ← c87703, ← c87704] at e_1428_10
  have e_1428_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_11
  simp only [← c87705, ← c87706, k2, ← c87707] at e_1428_11
  have hr := e_1428_11
  simp only [e_1428_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13531)) := by
  have c87708 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87709 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87710 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87711 := (publicBatchWrapper89_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87712 := (publicBatchWrapper89_copies2741 a h).1
  have c87713 := (publicBatchWrapper89_copies2741 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_12
  simp only [← c87708, ← c87709, ← c87710] at e_1428_12
  have e_1428_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_13
  simp only [← c87711, ← c87712, k2, ← c87713] at e_1428_13
  have hr := e_1428_13
  simp only [e_1428_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13532)) := by
  have c87714 := (publicBatchWrapper89_copies2741 a h).2.2.1
  have c87715 := (publicBatchWrapper89_copies2741 a h).2.2.2.1
  have c87716 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.1
  have c87717 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.1
  have c87718 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.1
  have c87719 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_14
  simp only [← c87714, ← c87715, ← c87716] at e_1428_14
  have e_1428_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_15
  simp only [← c87717, ← c87718, k2, ← c87719] at e_1428_15
  have hr := e_1428_15
  simp only [e_1428_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13533)) := by
  have c87720 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.1
  have c87721 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.1
  have c87722 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87723 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87724 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87725 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_16
  simp only [← c87720, ← c87721, ← c87722] at e_1428_16
  have e_1428_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_17
  simp only [← c87723, ← c87724, k2, ← c87725] at e_1428_17
  have hr := e_1428_17
  simp only [e_1428_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1428 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13534)) := by
  have c87726 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87727 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87728 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87729 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87730 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87731 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1428_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_18
  simp only [← c87726, ← c87727, ← c87728] at e_1428_18
  have e_1428_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1428 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1428_19
  simp only [← c87729, ← c87730, k2, ← c87731] at e_1428_19
  have hr := e_1428_19
  simp only [e_1428_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13535)) := by
  have c87732 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87733 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87734 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87735 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87736 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87737 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_0
  simp only [← c87732, ← c87733, ← c87734] at e_1429_0
  have e_1429_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_1
  simp only [← c87735, ← c87736, k2, ← c87737] at e_1429_1
  have hr := e_1429_1
  simp only [e_1429_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13536)) := by
  have c87738 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87739 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87740 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87741 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87742 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87743 := (publicBatchWrapper89_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_2
  simp only [← c87738, ← c87739, ← c87740] at e_1429_2
  have e_1429_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_3
  simp only [← c87741, ← c87742, k2, ← c87743] at e_1429_3
  have hr := e_1429_3
  simp only [e_1429_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13537)) := by
  have c87744 := (publicBatchWrapper89_copies2742 a h).1
  have c87745 := (publicBatchWrapper89_copies2742 a h).2.1
  have c87746 := (publicBatchWrapper89_copies2742 a h).2.2.1
  have c87747 := (publicBatchWrapper89_copies2742 a h).2.2.2.1
  have c87748 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.1
  have c87749 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_4
  simp only [← c87744, ← c87745, ← c87746] at e_1429_4
  have e_1429_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_5
  simp only [← c87747, ← c87748, k2, ← c87749] at e_1429_5
  have hr := e_1429_5
  simp only [e_1429_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13538)) := by
  have c87750 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.1
  have c87751 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.1
  have c87752 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.1
  have c87753 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.1
  have c87754 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87755 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_6
  simp only [← c87750, ← c87751, ← c87752] at e_1429_6
  have e_1429_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_7
  simp only [← c87753, ← c87754, k2, ← c87755] at e_1429_7
  have hr := e_1429_7
  simp only [e_1429_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13539)) := by
  have c87756 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87757 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87758 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87759 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87760 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87761 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_8
  simp only [← c87756, ← c87757, ← c87758] at e_1429_8
  have e_1429_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_9
  simp only [← c87759, ← c87760, k2, ← c87761] at e_1429_9
  have hr := e_1429_9
  simp only [e_1429_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13540)) := by
  have c87762 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87763 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87764 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87765 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87766 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87767 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_10
  simp only [← c87762, ← c87763, ← c87764] at e_1429_10
  have e_1429_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_11
  simp only [← c87765, ← c87766, k2, ← c87767] at e_1429_11
  have hr := e_1429_11
  simp only [e_1429_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13541)) := by
  have c87768 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87769 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87770 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87771 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87772 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87773 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_12
  simp only [← c87768, ← c87769, ← c87770] at e_1429_12
  have e_1429_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_13
  simp only [← c87771, ← c87772, k2, ← c87773] at e_1429_13
  have hr := e_1429_13
  simp only [e_1429_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13542)) := by
  have c87774 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87775 := (publicBatchWrapper89_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87776 := (publicBatchWrapper89_copies2743 a h).1
  have c87777 := (publicBatchWrapper89_copies2743 a h).2.1
  have c87778 := (publicBatchWrapper89_copies2743 a h).2.2.1
  have c87779 := (publicBatchWrapper89_copies2743 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_14
  simp only [← c87774, ← c87775, ← c87776] at e_1429_14
  have e_1429_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_15
  simp only [← c87777, ← c87778, k2, ← c87779] at e_1429_15
  have hr := e_1429_15
  simp only [e_1429_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13543)) := by
  have c87780 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.1
  have c87781 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.1
  have c87782 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.1
  have c87783 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.1
  have c87784 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.1
  have c87785 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_16
  simp only [← c87780, ← c87781, ← c87782] at e_1429_16
  have e_1429_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_17
  simp only [← c87783, ← c87784, k2, ← c87785] at e_1429_17
  have hr := e_1429_17
  simp only [e_1429_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1429 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13544)) := by
  have c87786 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87787 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87788 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87789 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87790 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87791 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1429_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_18
  simp only [← c87786, ← c87787, ← c87788] at e_1429_18
  have e_1429_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1429 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_19
  simp only [← c87789, ← c87790, k2, ← c87791] at e_1429_19
  have hr := e_1429_19
  simp only [e_1429_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13545)) := by
  have c87792 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87793 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87794 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87795 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87796 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87797 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_0
  simp only [← c87792, ← c87793, ← c87794] at e_1430_0
  have e_1430_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_1
  simp only [← c87795, ← c87796, k2, ← c87797] at e_1430_1
  have hr := e_1430_1
  simp only [e_1430_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13546)) := by
  have c87798 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87799 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87800 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87801 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87802 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87803 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_2
  simp only [← c87798, ← c87799, ← c87800] at e_1430_2
  have e_1430_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_3
  simp only [← c87801, ← c87802, k2, ← c87803] at e_1430_3
  have hr := e_1430_3
  simp only [e_1430_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13547)) := by
  have c87804 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87805 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87806 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87807 := (publicBatchWrapper89_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87808 := (publicBatchWrapper89_copies2744 a h).1
  have c87809 := (publicBatchWrapper89_copies2744 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_4
  simp only [← c87804, ← c87805, ← c87806] at e_1430_4
  have e_1430_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_5
  simp only [← c87807, ← c87808, k2, ← c87809] at e_1430_5
  have hr := e_1430_5
  simp only [e_1430_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13548)) := by
  have c87810 := (publicBatchWrapper89_copies2744 a h).2.2.1
  have c87811 := (publicBatchWrapper89_copies2744 a h).2.2.2.1
  have c87812 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.1
  have c87813 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.1
  have c87814 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.1
  have c87815 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_6
  simp only [← c87810, ← c87811, ← c87812] at e_1430_6
  have e_1430_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_7
  simp only [← c87813, ← c87814, k2, ← c87815] at e_1430_7
  have hr := e_1430_7
  simp only [e_1430_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13549)) := by
  have c87816 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.1
  have c87817 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.1
  have c87818 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87819 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87820 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87821 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_8
  simp only [← c87816, ← c87817, ← c87818] at e_1430_8
  have e_1430_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_9
  simp only [← c87819, ← c87820, k2, ← c87821] at e_1430_9
  have hr := e_1430_9
  simp only [e_1430_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13550)) := by
  have c87822 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87823 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87824 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87825 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87826 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87827 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_10
  simp only [← c87822, ← c87823, ← c87824] at e_1430_10
  have e_1430_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_11
  simp only [← c87825, ← c87826, k2, ← c87827] at e_1430_11
  have hr := e_1430_11
  simp only [e_1430_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13551)) := by
  have c87828 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87829 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87830 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87831 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87832 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87833 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_12
  simp only [← c87828, ← c87829, ← c87830] at e_1430_12
  have e_1430_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_13
  simp only [← c87831, ← c87832, k2, ← c87833] at e_1430_13
  have hr := e_1430_13
  simp only [e_1430_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13552)) := by
  have c87834 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87835 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87836 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87837 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87838 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87839 := (publicBatchWrapper89_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_14
  simp only [← c87834, ← c87835, ← c87836] at e_1430_14
  have e_1430_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_15
  simp only [← c87837, ← c87838, k2, ← c87839] at e_1430_15
  have hr := e_1430_15
  simp only [e_1430_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13553)) := by
  have c87840 := (publicBatchWrapper89_copies2745 a h).1
  have c87841 := (publicBatchWrapper89_copies2745 a h).2.1
  have c87842 := (publicBatchWrapper89_copies2745 a h).2.2.1
  have c87843 := (publicBatchWrapper89_copies2745 a h).2.2.2.1
  have c87844 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.1
  have c87845 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_16
  simp only [← c87840, ← c87841, ← c87842] at e_1430_16
  have e_1430_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_17
  simp only [← c87843, ← c87844, k2, ← c87845] at e_1430_17
  have hr := e_1430_17
  simp only [e_1430_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1430 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13554)) := by
  have c87846 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.1
  have c87847 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.1
  have c87848 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.1
  have c87849 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.1
  have c87850 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87851 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1430_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_18
  simp only [← c87846, ← c87847, ← c87848] at e_1430_18
  have e_1430_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1430 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_19
  simp only [← c87849, ← c87850, k2, ← c87851] at e_1430_19
  have hr := e_1430_19
  simp only [e_1430_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13555)) := by
  have c87852 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87853 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87854 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87855 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87856 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87857 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_0
  simp only [← c87852, ← c87853, ← c87854] at e_1431_0
  have e_1431_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_1
  simp only [← c87855, ← c87856, k2, ← c87857] at e_1431_1
  have hr := e_1431_1
  simp only [e_1431_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13556)) := by
  have c87858 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87859 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87860 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87861 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87862 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87863 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_2
  simp only [← c87858, ← c87859, ← c87860] at e_1431_2
  have e_1431_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_3
  simp only [← c87861, ← c87862, k2, ← c87863] at e_1431_3
  have hr := e_1431_3
  simp only [e_1431_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13557)) := by
  have c87864 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87865 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87866 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87867 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87868 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87869 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_4
  simp only [← c87864, ← c87865, ← c87866] at e_1431_4
  have e_1431_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_5
  simp only [← c87867, ← c87868, k2, ← c87869] at e_1431_5
  have hr := e_1431_5
  simp only [e_1431_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13558)) := by
  have c87870 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87871 := (publicBatchWrapper89_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87872 := (publicBatchWrapper89_copies2746 a h).1
  have c87873 := (publicBatchWrapper89_copies2746 a h).2.1
  have c87874 := (publicBatchWrapper89_copies2746 a h).2.2.1
  have c87875 := (publicBatchWrapper89_copies2746 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_6
  simp only [← c87870, ← c87871, ← c87872] at e_1431_6
  have e_1431_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_7
  simp only [← c87873, ← c87874, k2, ← c87875] at e_1431_7
  have hr := e_1431_7
  simp only [e_1431_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13559)) := by
  have c87876 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.1
  have c87877 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.1
  have c87878 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.1
  have c87879 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.1
  have c87880 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.1
  have c87881 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_8
  simp only [← c87876, ← c87877, ← c87878] at e_1431_8
  have e_1431_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_9
  simp only [← c87879, ← c87880, k2, ← c87881] at e_1431_9
  have hr := e_1431_9
  simp only [e_1431_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13560)) := by
  have c87882 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87883 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87884 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87885 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87886 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87887 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_10
  simp only [← c87882, ← c87883, ← c87884] at e_1431_10
  have e_1431_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_11
  simp only [← c87885, ← c87886, k2, ← c87887] at e_1431_11
  have hr := e_1431_11
  simp only [e_1431_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13561)) := by
  have c87888 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87889 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87890 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87891 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87892 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87893 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_12
  simp only [← c87888, ← c87889, ← c87890] at e_1431_12
  have e_1431_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_13
  simp only [← c87891, ← c87892, k2, ← c87893] at e_1431_13
  have hr := e_1431_13
  simp only [e_1431_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13562)) := by
  have c87894 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87895 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87896 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87897 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87898 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87899 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_14
  simp only [← c87894, ← c87895, ← c87896] at e_1431_14
  have e_1431_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_15
  simp only [← c87897, ← c87898, k2, ← c87899] at e_1431_15
  have hr := e_1431_15
  simp only [e_1431_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13563)) := by
  have c87900 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87901 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87902 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87903 := (publicBatchWrapper89_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87904 := (publicBatchWrapper89_copies2747 a h).1
  have c87905 := (publicBatchWrapper89_copies2747 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_16
  simp only [← c87900, ← c87901, ← c87902] at e_1431_16
  have e_1431_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_17
  simp only [← c87903, ← c87904, k2, ← c87905] at e_1431_17
  have hr := e_1431_17
  simp only [e_1431_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1431 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13564)) := by
  have c87906 := (publicBatchWrapper89_copies2747 a h).2.2.1
  have c87907 := (publicBatchWrapper89_copies2747 a h).2.2.2.1
  have c87908 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.1
  have c87909 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.1
  have c87910 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.1
  have c87911 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1431_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_18
  simp only [← c87906, ← c87907, ← c87908] at e_1431_18
  have e_1431_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1431 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_19
  simp only [← c87909, ← c87910, k2, ← c87911] at e_1431_19
  have hr := e_1431_19
  simp only [e_1431_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13565)) := by
  have c87912 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.1
  have c87913 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.1
  have c87914 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87915 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87916 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87917 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_0
  simp only [← c87912, ← c87913, ← c87914] at e_1432_0
  have e_1432_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_1
  simp only [← c87915, ← c87916, k2, ← c87917] at e_1432_1
  have hr := e_1432_1
  simp only [e_1432_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13566)) := by
  have c87918 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87919 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87920 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87921 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87922 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87923 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_2
  simp only [← c87918, ← c87919, ← c87920] at e_1432_2
  have e_1432_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_3
  simp only [← c87921, ← c87922, k2, ← c87923] at e_1432_3
  have hr := e_1432_3
  simp only [e_1432_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13567)) := by
  have c87924 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87925 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87926 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87927 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87928 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87929 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_4
  simp only [← c87924, ← c87925, ← c87926] at e_1432_4
  have e_1432_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_5
  simp only [← c87927, ← c87928, k2, ← c87929] at e_1432_5
  have hr := e_1432_5
  simp only [e_1432_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13568)) := by
  have c87930 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87931 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87932 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87933 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87934 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87935 := (publicBatchWrapper89_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_6
  simp only [← c87930, ← c87931, ← c87932] at e_1432_6
  have e_1432_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_7
  simp only [← c87933, ← c87934, k2, ← c87935] at e_1432_7
  have hr := e_1432_7
  simp only [e_1432_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13569)) := by
  have c87936 := (publicBatchWrapper89_copies2748 a h).1
  have c87937 := (publicBatchWrapper89_copies2748 a h).2.1
  have c87938 := (publicBatchWrapper89_copies2748 a h).2.2.1
  have c87939 := (publicBatchWrapper89_copies2748 a h).2.2.2.1
  have c87940 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.1
  have c87941 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_8
  simp only [← c87936, ← c87937, ← c87938] at e_1432_8
  have e_1432_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_9
  simp only [← c87939, ← c87940, k2, ← c87941] at e_1432_9
  have hr := e_1432_9
  simp only [e_1432_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13610)) := by
  have c87942 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.1
  have c87943 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.1
  have c87944 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.1
  have c87945 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.1
  have c87946 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87947 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_10
  simp only [← c87942, ← c87943, ← c87944] at e_1432_10
  have e_1432_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_11
  simp only [← c87945, ← c87946, k2, ← c87947] at e_1432_11
  have hr := e_1432_11
  simp only [e_1432_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13611)) := by
  have c87948 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87949 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87950 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87951 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87952 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87953 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_12
  simp only [← c87948, ← c87949, ← c87950] at e_1432_12
  have e_1432_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_13
  simp only [← c87951, ← c87952, k2, ← c87953] at e_1432_13
  have hr := e_1432_13
  simp only [e_1432_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13612)) := by
  have c87954 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87955 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87956 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87957 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87958 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87959 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_14
  simp only [← c87954, ← c87955, ← c87956] at e_1432_14
  have e_1432_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_15
  simp only [← c87957, ← c87958, k2, ← c87959] at e_1432_15
  have hr := e_1432_15
  simp only [e_1432_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13613)) := by
  have c87960 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87961 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87962 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87963 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87964 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87965 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_16
  simp only [← c87960, ← c87961, ← c87962] at e_1432_16
  have e_1432_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_17
  simp only [← c87963, ← c87964, k2, ← c87965] at e_1432_17
  have hr := e_1432_17
  simp only [e_1432_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1432 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13614)) := by
  have c87966 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87967 := (publicBatchWrapper89_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87968 := (publicBatchWrapper89_copies2749 a h).1
  have c87969 := (publicBatchWrapper89_copies2749 a h).2.1
  have c87970 := (publicBatchWrapper89_copies2749 a h).2.2.1
  have c87971 := (publicBatchWrapper89_copies2749 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1432_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_18
  simp only [← c87966, ← c87967, ← c87968] at e_1432_18
  have e_1432_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1432 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_19
  simp only [← c87969, ← c87970, k2, ← c87971] at e_1432_19
  have hr := e_1432_19
  simp only [e_1432_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13615)) := by
  have c87972 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.1
  have c87973 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.1
  have c87974 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.1
  have c87975 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.1
  have c87976 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.1
  have c87977 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_0
  simp only [← c87972, ← c87973, ← c87974] at e_1433_0
  have e_1433_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_1
  simp only [← c87975, ← c87976, k2, ← c87977] at e_1433_1
  have hr := e_1433_1
  simp only [e_1433_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13616)) := by
  have c87978 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87979 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87980 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87981 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87982 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87983 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_2
  simp only [← c87978, ← c87979, ← c87980] at e_1433_2
  have e_1433_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_3
  simp only [← c87981, ← c87982, k2, ← c87983] at e_1433_3
  have hr := e_1433_3
  simp only [e_1433_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13617)) := by
  have c87984 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87985 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87986 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87987 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87988 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87989 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_4
  simp only [← c87984, ← c87985, ← c87986] at e_1433_4
  have e_1433_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_5
  simp only [← c87987, ← c87988, k2, ← c87989] at e_1433_5
  have hr := e_1433_5
  simp only [e_1433_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13618)) := by
  have c87990 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87991 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87992 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87993 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87994 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87995 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_6
  simp only [← c87990, ← c87991, ← c87992] at e_1433_6
  have e_1433_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_7
  simp only [← c87993, ← c87994, k2, ← c87995] at e_1433_7
  have hr := e_1433_7
  simp only [e_1433_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13619)) := by
  have c87996 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87997 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87998 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87999 := (publicBatchWrapper89_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88000 := (publicBatchWrapper89_copies2750 a h).1
  have c88001 := (publicBatchWrapper89_copies2750 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_8
  simp only [← c87996, ← c87997, ← c87998] at e_1433_8
  have e_1433_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_9
  simp only [← c87999, ← c88000, k2, ← c88001] at e_1433_9
  have hr := e_1433_9
  simp only [e_1433_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13620)) := by
  have c88002 := (publicBatchWrapper89_copies2750 a h).2.2.1
  have c88003 := (publicBatchWrapper89_copies2750 a h).2.2.2.1
  have c88004 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.1
  have c88005 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.1
  have c88006 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.1
  have c88007 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_10
  simp only [← c88002, ← c88003, ← c88004] at e_1433_10
  have e_1433_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_11
  simp only [← c88005, ← c88006, k2, ← c88007] at e_1433_11
  have hr := e_1433_11
  simp only [e_1433_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13621)) := by
  have c88008 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.1
  have c88009 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.1
  have c88010 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88011 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88012 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88013 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_12
  simp only [← c88008, ← c88009, ← c88010] at e_1433_12
  have e_1433_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_13
  simp only [← c88011, ← c88012, k2, ← c88013] at e_1433_13
  have hr := e_1433_13
  simp only [e_1433_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13622)) := by
  have c88014 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88015 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88016 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88017 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88018 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88019 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_14
  simp only [← c88014, ← c88015, ← c88016] at e_1433_14
  have e_1433_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_15
  simp only [← c88017, ← c88018, k2, ← c88019] at e_1433_15
  have hr := e_1433_15
  simp only [e_1433_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13623)) := by
  have c88020 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88021 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88022 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88023 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88024 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88025 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_16
  simp only [← c88020, ← c88021, ← c88022] at e_1433_16
  have e_1433_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_17
  simp only [← c88023, ← c88024, k2, ← c88025] at e_1433_17
  have hr := e_1433_17
  simp only [e_1433_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1433 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13624)) := by
  have c88026 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88027 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88028 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88029 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88030 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88031 := (publicBatchWrapper89_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1433_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_18
  simp only [← c88026, ← c88027, ← c88028] at e_1433_18
  have e_1433_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1433 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_19
  simp only [← c88029, ← c88030, k2, ← c88031] at e_1433_19
  have hr := e_1433_19
  simp only [e_1433_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13625)) := by
  have c88032 := (publicBatchWrapper89_copies2751 a h).1
  have c88033 := (publicBatchWrapper89_copies2751 a h).2.1
  have c88034 := (publicBatchWrapper89_copies2751 a h).2.2.1
  have c88035 := (publicBatchWrapper89_copies2751 a h).2.2.2.1
  have c88036 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.1
  have c88037 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_0
  simp only [← c88032, ← c88033, ← c88034] at e_1434_0
  have e_1434_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_1
  simp only [← c88035, ← c88036, k2, ← c88037] at e_1434_1
  have hr := e_1434_1
  simp only [e_1434_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13626)) := by
  have c88038 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.1
  have c88039 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.1
  have c88040 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.1
  have c88041 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.1
  have c88042 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88043 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_2
  simp only [← c88038, ← c88039, ← c88040] at e_1434_2
  have e_1434_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_3
  simp only [← c88041, ← c88042, k2, ← c88043] at e_1434_3
  have hr := e_1434_3
  simp only [e_1434_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13627)) := by
  have c88044 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88045 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88046 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88047 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88048 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88049 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_4
  simp only [← c88044, ← c88045, ← c88046] at e_1434_4
  have e_1434_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_5
  simp only [← c88047, ← c88048, k2, ← c88049] at e_1434_5
  have hr := e_1434_5
  simp only [e_1434_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13628)) := by
  have c88050 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88051 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88052 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88053 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88054 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88055 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_6
  simp only [← c88050, ← c88051, ← c88052] at e_1434_6
  have e_1434_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_7
  simp only [← c88053, ← c88054, k2, ← c88055] at e_1434_7
  have hr := e_1434_7
  simp only [e_1434_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13629)) := by
  have c88056 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88057 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88058 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88059 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88060 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88061 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_8
  simp only [← c88056, ← c88057, ← c88058] at e_1434_8
  have e_1434_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_9
  simp only [← c88059, ← c88060, k2, ← c88061] at e_1434_9
  have hr := e_1434_9
  simp only [e_1434_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13630)) := by
  have c88062 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88063 := (publicBatchWrapper89_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88064 := (publicBatchWrapper89_copies2752 a h).1
  have c88065 := (publicBatchWrapper89_copies2752 a h).2.1
  have c88066 := (publicBatchWrapper89_copies2752 a h).2.2.1
  have c88067 := (publicBatchWrapper89_copies2752 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_10
  simp only [← c88062, ← c88063, ← c88064] at e_1434_10
  have e_1434_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_11
  simp only [← c88065, ← c88066, k2, ← c88067] at e_1434_11
  have hr := e_1434_11
  simp only [e_1434_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13631)) := by
  have c88068 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.1
  have c88069 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.1
  have c88070 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.1
  have c88071 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.1
  have c88072 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.1
  have c88073 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_12
  simp only [← c88068, ← c88069, ← c88070] at e_1434_12
  have e_1434_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_13
  simp only [← c88071, ← c88072, k2, ← c88073] at e_1434_13
  have hr := e_1434_13
  simp only [e_1434_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13632)) := by
  have c88074 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88075 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88076 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88077 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88078 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88079 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_14
  simp only [← c88074, ← c88075, ← c88076] at e_1434_14
  have e_1434_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_15
  simp only [← c88077, ← c88078, k2, ← c88079] at e_1434_15
  have hr := e_1434_15
  simp only [e_1434_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13633)) := by
  have c88080 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88081 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88082 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88083 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88084 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88085 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_16
  simp only [← c88080, ← c88081, ← c88082] at e_1434_16
  have e_1434_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_17
  simp only [← c88083, ← c88084, k2, ← c88085] at e_1434_17
  have hr := e_1434_17
  simp only [e_1434_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1434 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13634)) := by
  have c88086 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88087 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88088 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88089 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88090 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88091 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1434_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_18
  simp only [← c88086, ← c88087, ← c88088] at e_1434_18
  have e_1434_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1434 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_19
  simp only [← c88089, ← c88090, k2, ← c88091] at e_1434_19
  have hr := e_1434_19
  simp only [e_1434_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13635)) := by
  have c88092 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88093 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88094 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88095 := (publicBatchWrapper89_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88096 := (publicBatchWrapper89_copies2753 a h).1
  have c88097 := (publicBatchWrapper89_copies2753 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_0
  simp only [← c88092, ← c88093, ← c88094] at e_1435_0
  have e_1435_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_1
  simp only [← c88095, ← c88096, k2, ← c88097] at e_1435_1
  have hr := e_1435_1
  simp only [e_1435_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13636)) := by
  have c88098 := (publicBatchWrapper89_copies2753 a h).2.2.1
  have c88099 := (publicBatchWrapper89_copies2753 a h).2.2.2.1
  have c88100 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.1
  have c88101 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.1
  have c88102 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.1
  have c88103 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_2
  simp only [← c88098, ← c88099, ← c88100] at e_1435_2
  have e_1435_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_3
  simp only [← c88101, ← c88102, k2, ← c88103] at e_1435_3
  have hr := e_1435_3
  simp only [e_1435_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13637)) := by
  have c88104 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.1
  have c88105 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.1
  have c88106 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88107 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88108 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88109 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_4
  simp only [← c88104, ← c88105, ← c88106] at e_1435_4
  have e_1435_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_5
  simp only [← c88107, ← c88108, k2, ← c88109] at e_1435_5
  have hr := e_1435_5
  simp only [e_1435_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13638)) := by
  have c88110 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88111 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88112 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88113 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88114 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88115 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_6
  simp only [← c88110, ← c88111, ← c88112] at e_1435_6
  have e_1435_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_7
  simp only [← c88113, ← c88114, k2, ← c88115] at e_1435_7
  have hr := e_1435_7
  simp only [e_1435_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13639)) := by
  have c88116 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88117 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88118 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88119 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88120 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88121 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_8
  simp only [← c88116, ← c88117, ← c88118] at e_1435_8
  have e_1435_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_9
  simp only [← c88119, ← c88120, k2, ← c88121] at e_1435_9
  have hr := e_1435_9
  simp only [e_1435_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13640)) := by
  have c88122 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88123 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88124 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88125 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88126 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88127 := (publicBatchWrapper89_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_10
  simp only [← c88122, ← c88123, ← c88124] at e_1435_10
  have e_1435_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_11
  simp only [← c88125, ← c88126, k2, ← c88127] at e_1435_11
  have hr := e_1435_11
  simp only [e_1435_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13641)) := by
  have c88128 := (publicBatchWrapper89_copies2754 a h).1
  have c88129 := (publicBatchWrapper89_copies2754 a h).2.1
  have c88130 := (publicBatchWrapper89_copies2754 a h).2.2.1
  have c88131 := (publicBatchWrapper89_copies2754 a h).2.2.2.1
  have c88132 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.1
  have c88133 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_12
  simp only [← c88128, ← c88129, ← c88130] at e_1435_12
  have e_1435_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_13
  simp only [← c88131, ← c88132, k2, ← c88133] at e_1435_13
  have hr := e_1435_13
  simp only [e_1435_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13642)) := by
  have c88134 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.1
  have c88135 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.1
  have c88136 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.1
  have c88137 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.1
  have c88138 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88139 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_14
  simp only [← c88134, ← c88135, ← c88136] at e_1435_14
  have e_1435_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_15
  simp only [← c88137, ← c88138, k2, ← c88139] at e_1435_15
  have hr := e_1435_15
  simp only [e_1435_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13643)) := by
  have c88140 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88141 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88142 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88143 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88144 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88145 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_16
  simp only [← c88140, ← c88141, ← c88142] at e_1435_16
  have e_1435_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_17
  simp only [← c88143, ← c88144, k2, ← c88145] at e_1435_17
  have hr := e_1435_17
  simp only [e_1435_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1435 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13644)) := by
  have c88146 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88147 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88148 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88149 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88150 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88151 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1435_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_18
  simp only [← c88146, ← c88147, ← c88148] at e_1435_18
  have e_1435_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1435 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_19
  simp only [← c88149, ← c88150, k2, ← c88151] at e_1435_19
  have hr := e_1435_19
  simp only [e_1435_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13645)) := by
  have c88152 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88153 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88154 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88155 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88156 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88157 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_0
  simp only [← c88152, ← c88153, ← c88154] at e_1436_0
  have e_1436_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_1
  simp only [← c88155, ← c88156, k2, ← c88157] at e_1436_1
  have hr := e_1436_1
  simp only [e_1436_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13646)) := by
  have c88158 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88159 := (publicBatchWrapper89_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88160 := (publicBatchWrapper89_copies2755 a h).1
  have c88161 := (publicBatchWrapper89_copies2755 a h).2.1
  have c88162 := (publicBatchWrapper89_copies2755 a h).2.2.1
  have c88163 := (publicBatchWrapper89_copies2755 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_2
  simp only [← c88158, ← c88159, ← c88160] at e_1436_2
  have e_1436_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_3
  simp only [← c88161, ← c88162, k2, ← c88163] at e_1436_3
  have hr := e_1436_3
  simp only [e_1436_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13647)) := by
  have c88164 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.1
  have c88165 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.1
  have c88166 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.1
  have c88167 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.1
  have c88168 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.1
  have c88169 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_4
  simp only [← c88164, ← c88165, ← c88166] at e_1436_4
  have e_1436_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_5
  simp only [← c88167, ← c88168, k2, ← c88169] at e_1436_5
  have hr := e_1436_5
  simp only [e_1436_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13648)) := by
  have c88170 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88171 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88172 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88173 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88174 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88175 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_6
  simp only [← c88170, ← c88171, ← c88172] at e_1436_6
  have e_1436_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_7
  simp only [← c88173, ← c88174, k2, ← c88175] at e_1436_7
  have hr := e_1436_7
  simp only [e_1436_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13649)) := by
  have c88176 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88177 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88178 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88179 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88180 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88181 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_8
  simp only [← c88176, ← c88177, ← c88178] at e_1436_8
  have e_1436_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_9
  simp only [← c88179, ← c88180, k2, ← c88181] at e_1436_9
  have hr := e_1436_9
  simp only [e_1436_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13650)) := by
  have c88182 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88183 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88184 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88185 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88186 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88187 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_10
  simp only [← c88182, ← c88183, ← c88184] at e_1436_10
  have e_1436_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_11
  simp only [← c88185, ← c88186, k2, ← c88187] at e_1436_11
  have hr := e_1436_11
  simp only [e_1436_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13651)) := by
  have c88188 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88189 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88190 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88191 := (publicBatchWrapper89_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88192 := (publicBatchWrapper89_copies2756 a h).1
  have c88193 := (publicBatchWrapper89_copies2756 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_12
  simp only [← c88188, ← c88189, ← c88190] at e_1436_12
  have e_1436_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_13
  simp only [← c88191, ← c88192, k2, ← c88193] at e_1436_13
  have hr := e_1436_13
  simp only [e_1436_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13652)) := by
  have c88194 := (publicBatchWrapper89_copies2756 a h).2.2.1
  have c88195 := (publicBatchWrapper89_copies2756 a h).2.2.2.1
  have c88196 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.1
  have c88197 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.1
  have c88198 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.1
  have c88199 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_14
  simp only [← c88194, ← c88195, ← c88196] at e_1436_14
  have e_1436_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_15
  simp only [← c88197, ← c88198, k2, ← c88199] at e_1436_15
  have hr := e_1436_15
  simp only [e_1436_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13653)) := by
  have c88200 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.1
  have c88201 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.1
  have c88202 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88203 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88204 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88205 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_16
  simp only [← c88200, ← c88201, ← c88202] at e_1436_16
  have e_1436_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_17
  simp only [← c88203, ← c88204, k2, ← c88205] at e_1436_17
  have hr := e_1436_17
  simp only [e_1436_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1436 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13654)) := by
  have c88206 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88207 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88208 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88209 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88210 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88211 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1436_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_18
  simp only [← c88206, ← c88207, ← c88208] at e_1436_18
  have e_1436_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1436 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_19
  simp only [← c88209, ← c88210, k2, ← c88211] at e_1436_19
  have hr := e_1436_19
  simp only [e_1436_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13655)) := by
  have c88212 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88213 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88214 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88215 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88216 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88217 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_0
  simp only [← c88212, ← c88213, ← c88214] at e_1437_0
  have e_1437_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_1
  simp only [← c88215, ← c88216, k2, ← c88217] at e_1437_1
  have hr := e_1437_1
  simp only [e_1437_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13656)) := by
  have c88218 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88219 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88220 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88221 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88222 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88223 := (publicBatchWrapper89_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_2
  simp only [← c88218, ← c88219, ← c88220] at e_1437_2
  have e_1437_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_3
  simp only [← c88221, ← c88222, k2, ← c88223] at e_1437_3
  have hr := e_1437_3
  simp only [e_1437_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13657)) := by
  have c88224 := (publicBatchWrapper89_copies2757 a h).1
  have c88225 := (publicBatchWrapper89_copies2757 a h).2.1
  have c88226 := (publicBatchWrapper89_copies2757 a h).2.2.1
  have c88227 := (publicBatchWrapper89_copies2757 a h).2.2.2.1
  have c88228 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.1
  have c88229 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_4
  simp only [← c88224, ← c88225, ← c88226] at e_1437_4
  have e_1437_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_5
  simp only [← c88227, ← c88228, k2, ← c88229] at e_1437_5
  have hr := e_1437_5
  simp only [e_1437_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13658)) := by
  have c88230 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.1
  have c88231 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.1
  have c88232 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.1
  have c88233 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.1
  have c88234 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88235 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_6
  simp only [← c88230, ← c88231, ← c88232] at e_1437_6
  have e_1437_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_7
  simp only [← c88233, ← c88234, k2, ← c88235] at e_1437_7
  have hr := e_1437_7
  simp only [e_1437_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13659)) := by
  have c88236 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88237 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88238 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88239 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88240 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88241 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_8
  simp only [← c88236, ← c88237, ← c88238] at e_1437_8
  have e_1437_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_9
  simp only [← c88239, ← c88240, k2, ← c88241] at e_1437_9
  have hr := e_1437_9
  simp only [e_1437_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13660)) := by
  have c88242 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88243 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88244 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88245 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88246 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88247 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_10
  simp only [← c88242, ← c88243, ← c88244] at e_1437_10
  have e_1437_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_11
  simp only [← c88245, ← c88246, k2, ← c88247] at e_1437_11
  have hr := e_1437_11
  simp only [e_1437_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13661)) := by
  have c88248 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88249 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88250 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88251 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88252 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88253 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_12
  simp only [← c88248, ← c88249, ← c88250] at e_1437_12
  have e_1437_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_13
  simp only [← c88251, ← c88252, k2, ← c88253] at e_1437_13
  have hr := e_1437_13
  simp only [e_1437_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13662)) := by
  have c88254 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88255 := (publicBatchWrapper89_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88256 := (publicBatchWrapper89_copies2758 a h).1
  have c88257 := (publicBatchWrapper89_copies2758 a h).2.1
  have c88258 := (publicBatchWrapper89_copies2758 a h).2.2.1
  have c88259 := (publicBatchWrapper89_copies2758 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_14
  simp only [← c88254, ← c88255, ← c88256] at e_1437_14
  have e_1437_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_15
  simp only [← c88257, ← c88258, k2, ← c88259] at e_1437_15
  have hr := e_1437_15
  simp only [e_1437_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13663)) := by
  have c88260 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.1
  have c88261 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.1
  have c88262 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.1
  have c88263 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.1
  have c88264 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.1
  have c88265 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_16
  simp only [← c88260, ← c88261, ← c88262] at e_1437_16
  have e_1437_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_17
  simp only [← c88263, ← c88264, k2, ← c88265] at e_1437_17
  have hr := e_1437_17
  simp only [e_1437_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1437 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13664)) := by
  have c88266 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88267 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88268 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88269 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88270 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88271 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1437_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_18
  simp only [← c88266, ← c88267, ← c88268] at e_1437_18
  have e_1437_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1437 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_19
  simp only [← c88269, ← c88270, k2, ← c88271] at e_1437_19
  have hr := e_1437_19
  simp only [e_1437_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13665)) := by
  have c88272 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88273 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88274 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88275 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88276 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88277 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_0
  simp only [← c88272, ← c88273, ← c88274] at e_1438_0
  have e_1438_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_1
  simp only [← c88275, ← c88276, k2, ← c88277] at e_1438_1
  have hr := e_1438_1
  simp only [e_1438_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13666)) := by
  have c88278 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88279 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88280 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88281 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88282 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88283 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_2
  simp only [← c88278, ← c88279, ← c88280] at e_1438_2
  have e_1438_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_3
  simp only [← c88281, ← c88282, k2, ← c88283] at e_1438_3
  have hr := e_1438_3
  simp only [e_1438_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13667)) := by
  have c88284 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88285 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88286 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88287 := (publicBatchWrapper89_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88288 := (publicBatchWrapper89_copies2759 a h).1
  have c88289 := (publicBatchWrapper89_copies2759 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_4
  simp only [← c88284, ← c88285, ← c88286] at e_1438_4
  have e_1438_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_5
  simp only [← c88287, ← c88288, k2, ← c88289] at e_1438_5
  have hr := e_1438_5
  simp only [e_1438_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13668)) := by
  have c88290 := (publicBatchWrapper89_copies2759 a h).2.2.1
  have c88291 := (publicBatchWrapper89_copies2759 a h).2.2.2.1
  have c88292 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.1
  have c88293 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.1
  have c88294 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.1
  have c88295 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_6
  simp only [← c88290, ← c88291, ← c88292] at e_1438_6
  have e_1438_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_7
  simp only [← c88293, ← c88294, k2, ← c88295] at e_1438_7
  have hr := e_1438_7
  simp only [e_1438_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13669)) := by
  have c88296 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.1
  have c88297 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.1
  have c88298 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88299 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88300 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88301 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_8
  simp only [← c88296, ← c88297, ← c88298] at e_1438_8
  have e_1438_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_9
  simp only [← c88299, ← c88300, k2, ← c88301] at e_1438_9
  have hr := e_1438_9
  simp only [e_1438_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13670)) := by
  have c88302 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88303 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88304 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88305 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88306 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88307 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_10
  simp only [← c88302, ← c88303, ← c88304] at e_1438_10
  have e_1438_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_11
  simp only [← c88305, ← c88306, k2, ← c88307] at e_1438_11
  have hr := e_1438_11
  simp only [e_1438_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13671)) := by
  have c88308 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88309 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88310 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88311 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88312 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88313 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_12
  simp only [← c88308, ← c88309, ← c88310] at e_1438_12
  have e_1438_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_13
  simp only [← c88311, ← c88312, k2, ← c88313] at e_1438_13
  have hr := e_1438_13
  simp only [e_1438_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13672)) := by
  have c88314 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88315 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88316 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88317 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88318 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88319 := (publicBatchWrapper89_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_14
  simp only [← c88314, ← c88315, ← c88316] at e_1438_14
  have e_1438_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_15
  simp only [← c88317, ← c88318, k2, ← c88319] at e_1438_15
  have hr := e_1438_15
  simp only [e_1438_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13673)) := by
  have c88320 := (publicBatchWrapper89_copies2760 a h).1
  have c88321 := (publicBatchWrapper89_copies2760 a h).2.1
  have c88322 := (publicBatchWrapper89_copies2760 a h).2.2.1
  have c88323 := (publicBatchWrapper89_copies2760 a h).2.2.2.1
  have c88324 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.1
  have c88325 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_16
  simp only [← c88320, ← c88321, ← c88322] at e_1438_16
  have e_1438_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_17
  simp only [← c88323, ← c88324, k2, ← c88325] at e_1438_17
  have hr := e_1438_17
  simp only [e_1438_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1438 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13674)) := by
  have c88326 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.1
  have c88327 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.1
  have c88328 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.1
  have c88329 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.1
  have c88330 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88331 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1438_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_18
  simp only [← c88326, ← c88327, ← c88328] at e_1438_18
  have e_1438_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1438 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_19
  simp only [← c88329, ← c88330, k2, ← c88331] at e_1438_19
  have hr := e_1438_19
  simp only [e_1438_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13675)) := by
  have c88332 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88333 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88334 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88335 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88336 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88337 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_0
  simp only [← c88332, ← c88333, ← c88334] at e_1439_0
  have e_1439_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_1
  simp only [← c88335, ← c88336, k2, ← c88337] at e_1439_1
  have hr := e_1439_1
  simp only [e_1439_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13676)) := by
  have c88338 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88339 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88340 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88341 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88342 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88343 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_2
  simp only [← c88338, ← c88339, ← c88340] at e_1439_2
  have e_1439_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_3
  simp only [← c88341, ← c88342, k2, ← c88343] at e_1439_3
  have hr := e_1439_3
  simp only [e_1439_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13677)) := by
  have c88344 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88345 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88346 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88347 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88348 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88349 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_4
  simp only [← c88344, ← c88345, ← c88346] at e_1439_4
  have e_1439_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_5
  simp only [← c88347, ← c88348, k2, ← c88349] at e_1439_5
  have hr := e_1439_5
  simp only [e_1439_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13678)) := by
  have c88350 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88351 := (publicBatchWrapper89_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88352 := (publicBatchWrapper89_copies2761 a h).1
  have c88353 := (publicBatchWrapper89_copies2761 a h).2.1
  have c88354 := (publicBatchWrapper89_copies2761 a h).2.2.1
  have c88355 := (publicBatchWrapper89_copies2761 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_6
  simp only [← c88350, ← c88351, ← c88352] at e_1439_6
  have e_1439_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_7
  simp only [← c88353, ← c88354, k2, ← c88355] at e_1439_7
  have hr := e_1439_7
  simp only [e_1439_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13679)) := by
  have c88356 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.1
  have c88357 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.1
  have c88358 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.1
  have c88359 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.1
  have c88360 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.1
  have c88361 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_8
  simp only [← c88356, ← c88357, ← c88358] at e_1439_8
  have e_1439_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_9
  simp only [← c88359, ← c88360, k2, ← c88361] at e_1439_9
  have hr := e_1439_9
  simp only [e_1439_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13680)) := by
  have c88362 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88363 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88364 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88365 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88366 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88367 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_10
  simp only [← c88362, ← c88363, ← c88364] at e_1439_10
  have e_1439_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_11
  simp only [← c88365, ← c88366, k2, ← c88367] at e_1439_11
  have hr := e_1439_11
  simp only [e_1439_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13681)) := by
  have c88368 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88369 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88370 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88371 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88372 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88373 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_12
  simp only [← c88368, ← c88369, ← c88370] at e_1439_12
  have e_1439_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_13
  simp only [← c88371, ← c88372, k2, ← c88373] at e_1439_13
  have hr := e_1439_13
  simp only [e_1439_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13682)) := by
  have c88374 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88375 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88376 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88377 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88378 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88379 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_14
  simp only [← c88374, ← c88375, ← c88376] at e_1439_14
  have e_1439_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_15
  simp only [← c88377, ← c88378, k2, ← c88379] at e_1439_15
  have hr := e_1439_15
  simp only [e_1439_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13683)) := by
  have c88380 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88381 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88382 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88383 := (publicBatchWrapper89_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88384 := (publicBatchWrapper89_copies2762 a h).1
  have c88385 := (publicBatchWrapper89_copies2762 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_16
  simp only [← c88380, ← c88381, ← c88382] at e_1439_16
  have e_1439_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_17
  simp only [← c88383, ← c88384, k2, ← c88385] at e_1439_17
  have hr := e_1439_17
  simp only [e_1439_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1439 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13684)) := by
  have c88386 := (publicBatchWrapper89_copies2762 a h).2.2.1
  have c88387 := (publicBatchWrapper89_copies2762 a h).2.2.2.1
  have c88388 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.1
  have c88389 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.1
  have c88390 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.1
  have c88391 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1439_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_18
  simp only [← c88386, ← c88387, ← c88388] at e_1439_18
  have e_1439_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1439 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_19
  simp only [← c88389, ← c88390, k2, ← c88391] at e_1439_19
  have hr := e_1439_19
  simp only [e_1439_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13685)) := by
  have c88392 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.1
  have c88393 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.1
  have c88394 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88395 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88396 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88397 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_0
  simp only [← c88392, ← c88393, ← c88394] at e_1440_0
  have e_1440_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_1
  simp only [← c88395, ← c88396, k2, ← c88397] at e_1440_1
  have hr := e_1440_1
  simp only [e_1440_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13686)) := by
  have c88398 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88399 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88400 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88401 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88402 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88403 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_2
  simp only [← c88398, ← c88399, ← c88400] at e_1440_2
  have e_1440_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_3
  simp only [← c88401, ← c88402, k2, ← c88403] at e_1440_3
  have hr := e_1440_3
  simp only [e_1440_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13687)) := by
  have c88404 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88405 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88406 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88407 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88408 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88409 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_4
  simp only [← c88404, ← c88405, ← c88406] at e_1440_4
  have e_1440_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_5
  simp only [← c88407, ← c88408, k2, ← c88409] at e_1440_5
  have hr := e_1440_5
  simp only [e_1440_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13688)) := by
  have c88410 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88411 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88412 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88413 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88414 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88415 := (publicBatchWrapper89_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_6
  simp only [← c88410, ← c88411, ← c88412] at e_1440_6
  have e_1440_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_7
  simp only [← c88413, ← c88414, k2, ← c88415] at e_1440_7
  have hr := e_1440_7
  simp only [e_1440_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13689)) := by
  have c88416 := (publicBatchWrapper89_copies2763 a h).1
  have c88417 := (publicBatchWrapper89_copies2763 a h).2.1
  have c88418 := (publicBatchWrapper89_copies2763 a h).2.2.1
  have c88419 := (publicBatchWrapper89_copies2763 a h).2.2.2.1
  have c88420 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.1
  have c88421 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_8
  simp only [← c88416, ← c88417, ← c88418] at e_1440_8
  have e_1440_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_9
  simp only [← c88419, ← c88420, k2, ← c88421] at e_1440_9
  have hr := e_1440_9
  simp only [e_1440_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13690)) := by
  have c88422 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.1
  have c88423 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.1
  have c88424 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.1
  have c88425 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.1
  have c88426 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88427 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_10
  simp only [← c88422, ← c88423, ← c88424] at e_1440_10
  have e_1440_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_11
  simp only [← c88425, ← c88426, k2, ← c88427] at e_1440_11
  have hr := e_1440_11
  simp only [e_1440_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13691)) := by
  have c88428 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88429 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88430 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88431 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88432 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88433 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_12
  simp only [← c88428, ← c88429, ← c88430] at e_1440_12
  have e_1440_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_13
  simp only [← c88431, ← c88432, k2, ← c88433] at e_1440_13
  have hr := e_1440_13
  simp only [e_1440_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13692)) := by
  have c88434 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88435 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88436 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88437 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88438 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88439 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_14
  simp only [← c88434, ← c88435, ← c88436] at e_1440_14
  have e_1440_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_15
  simp only [← c88437, ← c88438, k2, ← c88439] at e_1440_15
  have hr := e_1440_15
  simp only [e_1440_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13693)) := by
  have c88440 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88441 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88442 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88443 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88444 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88445 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_16
  simp only [← c88440, ← c88441, ← c88442] at e_1440_16
  have e_1440_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_17
  simp only [← c88443, ← c88444, k2, ← c88445] at e_1440_17
  have hr := e_1440_17
  simp only [e_1440_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1440 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13694)) := by
  have c88446 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88447 := (publicBatchWrapper89_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88448 := (publicBatchWrapper89_copies2764 a h).1
  have c88449 := (publicBatchWrapper89_copies2764 a h).2.1
  have c88450 := (publicBatchWrapper89_copies2764 a h).2.2.1
  have c88451 := (publicBatchWrapper89_copies2764 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1440_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_18
  simp only [← c88446, ← c88447, ← c88448] at e_1440_18
  have e_1440_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1440 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_19
  simp only [← c88449, ← c88450, k2, ← c88451] at e_1440_19
  have hr := e_1440_19
  simp only [e_1440_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13695)) := by
  have c88452 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.1
  have c88453 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.1
  have c88454 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.1
  have c88455 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.1
  have c88456 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.1
  have c88457 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_0
  simp only [← c88452, ← c88453, ← c88454] at e_1441_0
  have e_1441_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_1
  simp only [← c88455, ← c88456, k2, ← c88457] at e_1441_1
  have hr := e_1441_1
  simp only [e_1441_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13696)) := by
  have c88458 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88459 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88460 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88461 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88462 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88463 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_2
  simp only [← c88458, ← c88459, ← c88460] at e_1441_2
  have e_1441_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_3
  simp only [← c88461, ← c88462, k2, ← c88463] at e_1441_3
  have hr := e_1441_3
  simp only [e_1441_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13697)) := by
  have c88464 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88465 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88466 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88467 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88468 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88469 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_4
  simp only [← c88464, ← c88465, ← c88466] at e_1441_4
  have e_1441_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_5
  simp only [← c88467, ← c88468, k2, ← c88469] at e_1441_5
  have hr := e_1441_5
  simp only [e_1441_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13698)) := by
  have c88470 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88471 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88472 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88473 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88474 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88475 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_6
  simp only [← c88470, ← c88471, ← c88472] at e_1441_6
  have e_1441_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_7
  simp only [← c88473, ← c88474, k2, ← c88475] at e_1441_7
  have hr := e_1441_7
  simp only [e_1441_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13699)) := by
  have c88476 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88477 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88478 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88479 := (publicBatchWrapper89_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88480 := (publicBatchWrapper89_copies2765 a h).1
  have c88481 := (publicBatchWrapper89_copies2765 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_8
  simp only [← c88476, ← c88477, ← c88478] at e_1441_8
  have e_1441_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_9
  simp only [← c88479, ← c88480, k2, ← c88481] at e_1441_9
  have hr := e_1441_9
  simp only [e_1441_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13700)) := by
  have c88482 := (publicBatchWrapper89_copies2765 a h).2.2.1
  have c88483 := (publicBatchWrapper89_copies2765 a h).2.2.2.1
  have c88484 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.1
  have c88485 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.1
  have c88486 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.1
  have c88487 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_10
  simp only [← c88482, ← c88483, ← c88484] at e_1441_10
  have e_1441_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_11
  simp only [← c88485, ← c88486, k2, ← c88487] at e_1441_11
  have hr := e_1441_11
  simp only [e_1441_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13701)) := by
  have c88488 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.1
  have c88489 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.1
  have c88490 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88491 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88492 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88493 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_12
  simp only [← c88488, ← c88489, ← c88490] at e_1441_12
  have e_1441_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_13
  simp only [← c88491, ← c88492, k2, ← c88493] at e_1441_13
  have hr := e_1441_13
  simp only [e_1441_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13702)) := by
  have c88494 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88495 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88496 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88497 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88498 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88499 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_14
  simp only [← c88494, ← c88495, ← c88496] at e_1441_14
  have e_1441_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_15
  simp only [← c88497, ← c88498, k2, ← c88499] at e_1441_15
  have hr := e_1441_15
  simp only [e_1441_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13703)) := by
  have c88500 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88501 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88502 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88503 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88504 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88505 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_16
  simp only [← c88500, ← c88501, ← c88502] at e_1441_16
  have e_1441_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_17
  simp only [← c88503, ← c88504, k2, ← c88505] at e_1441_17
  have hr := e_1441_17
  simp only [e_1441_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1441 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13704)) := by
  have c88506 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88507 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88508 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88509 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88510 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88511 := (publicBatchWrapper89_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1441_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_18
  simp only [← c88506, ← c88507, ← c88508] at e_1441_18
  have e_1441_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1441 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_19
  simp only [← c88509, ← c88510, k2, ← c88511] at e_1441_19
  have hr := e_1441_19
  simp only [e_1441_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13705)) := by
  have c88512 := (publicBatchWrapper89_copies2766 a h).1
  have c88513 := (publicBatchWrapper89_copies2766 a h).2.1
  have c88514 := (publicBatchWrapper89_copies2766 a h).2.2.1
  have c88515 := (publicBatchWrapper89_copies2766 a h).2.2.2.1
  have c88516 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.1
  have c88517 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_0
  simp only [← c88512, ← c88513, ← c88514] at e_1442_0
  have e_1442_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_1
  simp only [← c88515, ← c88516, k2, ← c88517] at e_1442_1
  have hr := e_1442_1
  simp only [e_1442_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13706)) := by
  have c88518 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.1
  have c88519 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.1
  have c88520 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.1
  have c88521 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.1
  have c88522 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88523 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_2
  simp only [← c88518, ← c88519, ← c88520] at e_1442_2
  have e_1442_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_3
  simp only [← c88521, ← c88522, k2, ← c88523] at e_1442_3
  have hr := e_1442_3
  simp only [e_1442_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13707)) := by
  have c88524 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88525 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88526 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88527 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88528 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88529 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_4
  simp only [← c88524, ← c88525, ← c88526] at e_1442_4
  have e_1442_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_5
  simp only [← c88527, ← c88528, k2, ← c88529] at e_1442_5
  have hr := e_1442_5
  simp only [e_1442_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13708)) := by
  have c88530 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88531 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88532 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88533 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88534 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88535 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_6
  simp only [← c88530, ← c88531, ← c88532] at e_1442_6
  have e_1442_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_7
  simp only [← c88533, ← c88534, k2, ← c88535] at e_1442_7
  have hr := e_1442_7
  simp only [e_1442_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13709)) := by
  have c88536 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88537 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88538 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88539 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88540 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88541 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_8
  simp only [← c88536, ← c88537, ← c88538] at e_1442_8
  have e_1442_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_9
  simp only [← c88539, ← c88540, k2, ← c88541] at e_1442_9
  have hr := e_1442_9
  simp only [e_1442_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13710)) := by
  have c88542 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88543 := (publicBatchWrapper89_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88544 := (publicBatchWrapper89_copies2767 a h).1
  have c88545 := (publicBatchWrapper89_copies2767 a h).2.1
  have c88546 := (publicBatchWrapper89_copies2767 a h).2.2.1
  have c88547 := (publicBatchWrapper89_copies2767 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_10
  simp only [← c88542, ← c88543, ← c88544] at e_1442_10
  have e_1442_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_11
  simp only [← c88545, ← c88546, k2, ← c88547] at e_1442_11
  have hr := e_1442_11
  simp only [e_1442_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13711)) := by
  have c88548 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.1
  have c88549 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.1
  have c88550 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.1
  have c88551 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.1
  have c88552 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.1
  have c88553 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_12
  simp only [← c88548, ← c88549, ← c88550] at e_1442_12
  have e_1442_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_13
  simp only [← c88551, ← c88552, k2, ← c88553] at e_1442_13
  have hr := e_1442_13
  simp only [e_1442_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13712)) := by
  have c88554 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88555 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88556 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88557 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88558 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88559 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_14
  simp only [← c88554, ← c88555, ← c88556] at e_1442_14
  have e_1442_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_15
  simp only [← c88557, ← c88558, k2, ← c88559] at e_1442_15
  have hr := e_1442_15
  simp only [e_1442_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13713)) := by
  have c88560 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88561 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88562 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88563 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88564 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88565 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_16
  simp only [← c88560, ← c88561, ← c88562] at e_1442_16
  have e_1442_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_17
  simp only [← c88563, ← c88564, k2, ← c88565] at e_1442_17
  have hr := e_1442_17
  simp only [e_1442_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1442 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13714)) := by
  have c88566 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88567 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88568 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88569 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88570 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88571 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1442_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_18
  simp only [← c88566, ← c88567, ← c88568] at e_1442_18
  have e_1442_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1442 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_19
  simp only [← c88569, ← c88570, k2, ← c88571] at e_1442_19
  have hr := e_1442_19
  simp only [e_1442_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13715)) := by
  have c88572 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88573 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88574 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88575 := (publicBatchWrapper89_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88576 := (publicBatchWrapper89_copies2768 a h).1
  have c88577 := (publicBatchWrapper89_copies2768 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_0
  simp only [← c88572, ← c88573, ← c88574] at e_1443_0
  have e_1443_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_1
  simp only [← c88575, ← c88576, k2, ← c88577] at e_1443_1
  have hr := e_1443_1
  simp only [e_1443_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13716)) := by
  have c88578 := (publicBatchWrapper89_copies2768 a h).2.2.1
  have c88579 := (publicBatchWrapper89_copies2768 a h).2.2.2.1
  have c88580 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.1
  have c88581 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.1
  have c88582 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.1
  have c88583 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_2
  simp only [← c88578, ← c88579, ← c88580] at e_1443_2
  have e_1443_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_3
  simp only [← c88581, ← c88582, k2, ← c88583] at e_1443_3
  have hr := e_1443_3
  simp only [e_1443_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13717)) := by
  have c88584 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.1
  have c88585 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.1
  have c88586 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88587 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88588 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88589 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_4
  simp only [← c88584, ← c88585, ← c88586] at e_1443_4
  have e_1443_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_5
  simp only [← c88587, ← c88588, k2, ← c88589] at e_1443_5
  have hr := e_1443_5
  simp only [e_1443_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13718)) := by
  have c88590 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88591 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88592 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88593 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88594 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88595 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_6
  simp only [← c88590, ← c88591, ← c88592] at e_1443_6
  have e_1443_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_7
  simp only [← c88593, ← c88594, k2, ← c88595] at e_1443_7
  have hr := e_1443_7
  simp only [e_1443_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13719)) := by
  have c88596 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88597 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88598 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88599 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88600 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88601 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_8
  simp only [← c88596, ← c88597, ← c88598] at e_1443_8
  have e_1443_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_9
  simp only [← c88599, ← c88600, k2, ← c88601] at e_1443_9
  have hr := e_1443_9
  simp only [e_1443_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13720)) := by
  have c88602 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88603 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88604 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88605 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88606 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88607 := (publicBatchWrapper89_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_10
  simp only [← c88602, ← c88603, ← c88604] at e_1443_10
  have e_1443_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_11
  simp only [← c88605, ← c88606, k2, ← c88607] at e_1443_11
  have hr := e_1443_11
  simp only [e_1443_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13721)) := by
  have c88608 := (publicBatchWrapper89_copies2769 a h).1
  have c88609 := (publicBatchWrapper89_copies2769 a h).2.1
  have c88610 := (publicBatchWrapper89_copies2769 a h).2.2.1
  have c88611 := (publicBatchWrapper89_copies2769 a h).2.2.2.1
  have c88612 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.1
  have c88613 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_12
  simp only [← c88608, ← c88609, ← c88610] at e_1443_12
  have e_1443_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_13
  simp only [← c88611, ← c88612, k2, ← c88613] at e_1443_13
  have hr := e_1443_13
  simp only [e_1443_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13722)) := by
  have c88614 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.1
  have c88615 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.1
  have c88616 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.1
  have c88617 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.1
  have c88618 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88619 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_14
  simp only [← c88614, ← c88615, ← c88616] at e_1443_14
  have e_1443_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_15
  simp only [← c88617, ← c88618, k2, ← c88619] at e_1443_15
  have hr := e_1443_15
  simp only [e_1443_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13723)) := by
  have c88620 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88621 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88622 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88623 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88624 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88625 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_16
  simp only [← c88620, ← c88621, ← c88622] at e_1443_16
  have e_1443_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_17
  simp only [← c88623, ← c88624, k2, ← c88625] at e_1443_17
  have hr := e_1443_17
  simp only [e_1443_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1443 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13724)) := by
  have c88626 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88627 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88628 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88629 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88630 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88631 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1443_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_18
  simp only [← c88626, ← c88627, ← c88628] at e_1443_18
  have e_1443_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1443 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_19
  simp only [← c88629, ← c88630, k2, ← c88631] at e_1443_19
  have hr := e_1443_19
  simp only [e_1443_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13725)) := by
  have c88632 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88633 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88634 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88635 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88636 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88637 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_0
  simp only [← c88632, ← c88633, ← c88634] at e_1444_0
  have e_1444_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_1
  simp only [← c88635, ← c88636, k2, ← c88637] at e_1444_1
  have hr := e_1444_1
  simp only [e_1444_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13726)) := by
  have c88638 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88639 := (publicBatchWrapper89_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88640 := (publicBatchWrapper89_copies2770 a h).1
  have c88641 := (publicBatchWrapper89_copies2770 a h).2.1
  have c88642 := (publicBatchWrapper89_copies2770 a h).2.2.1
  have c88643 := (publicBatchWrapper89_copies2770 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_2
  simp only [← c88638, ← c88639, ← c88640] at e_1444_2
  have e_1444_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_3
  simp only [← c88641, ← c88642, k2, ← c88643] at e_1444_3
  have hr := e_1444_3
  simp only [e_1444_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13727)) := by
  have c88644 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.1
  have c88645 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.1
  have c88646 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.1
  have c88647 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.1
  have c88648 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.1
  have c88649 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_4
  simp only [← c88644, ← c88645, ← c88646] at e_1444_4
  have e_1444_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_5
  simp only [← c88647, ← c88648, k2, ← c88649] at e_1444_5
  have hr := e_1444_5
  simp only [e_1444_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13728)) := by
  have c88650 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88651 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88652 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88653 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88654 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88655 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_6
  simp only [← c88650, ← c88651, ← c88652] at e_1444_6
  have e_1444_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_7
  simp only [← c88653, ← c88654, k2, ← c88655] at e_1444_7
  have hr := e_1444_7
  simp only [e_1444_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13729)) := by
  have c88656 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88657 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88658 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88659 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88660 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88661 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_8
  simp only [← c88656, ← c88657, ← c88658] at e_1444_8
  have e_1444_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_9
  simp only [← c88659, ← c88660, k2, ← c88661] at e_1444_9
  have hr := e_1444_9
  simp only [e_1444_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13730)) := by
  have c88662 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88663 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88664 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88665 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88666 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88667 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_10
  simp only [← c88662, ← c88663, ← c88664] at e_1444_10
  have e_1444_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_11
  simp only [← c88665, ← c88666, k2, ← c88667] at e_1444_11
  have hr := e_1444_11
  simp only [e_1444_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13731)) := by
  have c88668 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88669 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88670 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88671 := (publicBatchWrapper89_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88672 := (publicBatchWrapper89_copies2771 a h).1
  have c88673 := (publicBatchWrapper89_copies2771 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_12
  simp only [← c88668, ← c88669, ← c88670] at e_1444_12
  have e_1444_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_13
  simp only [← c88671, ← c88672, k2, ← c88673] at e_1444_13
  have hr := e_1444_13
  simp only [e_1444_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13732)) := by
  have c88674 := (publicBatchWrapper89_copies2771 a h).2.2.1
  have c88675 := (publicBatchWrapper89_copies2771 a h).2.2.2.1
  have c88676 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.1
  have c88677 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.1
  have c88678 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.1
  have c88679 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_14
  simp only [← c88674, ← c88675, ← c88676] at e_1444_14
  have e_1444_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_15
  simp only [← c88677, ← c88678, k2, ← c88679] at e_1444_15
  have hr := e_1444_15
  simp only [e_1444_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13733)) := by
  have c88680 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.1
  have c88681 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.1
  have c88682 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88683 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88684 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88685 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_16
  simp only [← c88680, ← c88681, ← c88682] at e_1444_16
  have e_1444_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_17
  simp only [← c88683, ← c88684, k2, ← c88685] at e_1444_17
  have hr := e_1444_17
  simp only [e_1444_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1444 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13734)) := by
  have c88686 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88687 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88688 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88689 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88690 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88691 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1444_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_18
  simp only [← c88686, ← c88687, ← c88688] at e_1444_18
  have e_1444_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1444 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_19
  simp only [← c88689, ← c88690, k2, ← c88691] at e_1444_19
  have hr := e_1444_19
  simp only [e_1444_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13735)) := by
  have c88692 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88693 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88694 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88695 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88696 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88697 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_0
  simp only [← c88692, ← c88693, ← c88694] at e_1445_0
  have e_1445_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_1
  simp only [← c88695, ← c88696, k2, ← c88697] at e_1445_1
  have hr := e_1445_1
  simp only [e_1445_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13736)) := by
  have c88698 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88699 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88700 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88701 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88702 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88703 := (publicBatchWrapper89_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_2
  simp only [← c88698, ← c88699, ← c88700] at e_1445_2
  have e_1445_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_3
  simp only [← c88701, ← c88702, k2, ← c88703] at e_1445_3
  have hr := e_1445_3
  simp only [e_1445_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13737)) := by
  have c88704 := (publicBatchWrapper89_copies2772 a h).1
  have c88705 := (publicBatchWrapper89_copies2772 a h).2.1
  have c88706 := (publicBatchWrapper89_copies2772 a h).2.2.1
  have c88707 := (publicBatchWrapper89_copies2772 a h).2.2.2.1
  have c88708 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.1
  have c88709 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_4
  simp only [← c88704, ← c88705, ← c88706] at e_1445_4
  have e_1445_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_5
  simp only [← c88707, ← c88708, k2, ← c88709] at e_1445_5
  have hr := e_1445_5
  simp only [e_1445_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13738)) := by
  have c88710 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.1
  have c88711 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.1
  have c88712 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.1
  have c88713 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.1
  have c88714 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88715 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_6
  simp only [← c88710, ← c88711, ← c88712] at e_1445_6
  have e_1445_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_7
  simp only [← c88713, ← c88714, k2, ← c88715] at e_1445_7
  have hr := e_1445_7
  simp only [e_1445_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13739)) := by
  have c88716 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88717 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88718 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88719 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88720 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88721 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_8
  simp only [← c88716, ← c88717, ← c88718] at e_1445_8
  have e_1445_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_9
  simp only [← c88719, ← c88720, k2, ← c88721] at e_1445_9
  have hr := e_1445_9
  simp only [e_1445_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13740)) := by
  have c88722 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88723 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88724 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88725 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88726 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88727 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_10
  simp only [← c88722, ← c88723, ← c88724] at e_1445_10
  have e_1445_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_11
  simp only [← c88725, ← c88726, k2, ← c88727] at e_1445_11
  have hr := e_1445_11
  simp only [e_1445_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13741)) := by
  have c88728 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88729 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88730 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88731 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88732 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88733 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_12
  simp only [← c88728, ← c88729, ← c88730] at e_1445_12
  have e_1445_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_13
  simp only [← c88731, ← c88732, k2, ← c88733] at e_1445_13
  have hr := e_1445_13
  simp only [e_1445_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13742)) := by
  have c88734 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88735 := (publicBatchWrapper89_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88736 := (publicBatchWrapper89_copies2773 a h).1
  have c88737 := (publicBatchWrapper89_copies2773 a h).2.1
  have c88738 := (publicBatchWrapper89_copies2773 a h).2.2.1
  have c88739 := (publicBatchWrapper89_copies2773 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_14
  simp only [← c88734, ← c88735, ← c88736] at e_1445_14
  have e_1445_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_15
  simp only [← c88737, ← c88738, k2, ← c88739] at e_1445_15
  have hr := e_1445_15
  simp only [e_1445_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13743)) := by
  have c88740 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.1
  have c88741 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.1
  have c88742 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.1
  have c88743 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.1
  have c88744 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.1
  have c88745 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_16
  simp only [← c88740, ← c88741, ← c88742] at e_1445_16
  have e_1445_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_17
  simp only [← c88743, ← c88744, k2, ← c88745] at e_1445_17
  have hr := e_1445_17
  simp only [e_1445_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1445 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13744)) := by
  have c88746 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88747 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88748 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88749 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88750 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88751 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1445_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_18
  simp only [← c88746, ← c88747, ← c88748] at e_1445_18
  have e_1445_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1445 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_19
  simp only [← c88749, ← c88750, k2, ← c88751] at e_1445_19
  have hr := e_1445_19
  simp only [e_1445_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13745)) := by
  have c88752 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88753 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88754 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88755 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88756 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88757 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_0
  simp only [← c88752, ← c88753, ← c88754] at e_1446_0
  have e_1446_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_1
  simp only [← c88755, ← c88756, k2, ← c88757] at e_1446_1
  have hr := e_1446_1
  simp only [e_1446_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13746)) := by
  have c88758 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88759 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88760 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88761 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88762 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88763 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_2
  simp only [← c88758, ← c88759, ← c88760] at e_1446_2
  have e_1446_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_3
  simp only [← c88761, ← c88762, k2, ← c88763] at e_1446_3
  have hr := e_1446_3
  simp only [e_1446_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13747)) := by
  have c88764 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88765 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88766 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88767 := (publicBatchWrapper89_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88768 := (publicBatchWrapper89_copies2774 a h).1
  have c88769 := (publicBatchWrapper89_copies2774 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_4
  simp only [← c88764, ← c88765, ← c88766] at e_1446_4
  have e_1446_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_5
  simp only [← c88767, ← c88768, k2, ← c88769] at e_1446_5
  have hr := e_1446_5
  simp only [e_1446_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13748)) := by
  have c88770 := (publicBatchWrapper89_copies2774 a h).2.2.1
  have c88771 := (publicBatchWrapper89_copies2774 a h).2.2.2.1
  have c88772 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.1
  have c88773 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.1
  have c88774 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.1
  have c88775 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_6
  simp only [← c88770, ← c88771, ← c88772] at e_1446_6
  have e_1446_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_7
  simp only [← c88773, ← c88774, k2, ← c88775] at e_1446_7
  have hr := e_1446_7
  simp only [e_1446_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13749)) := by
  have c88776 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.1
  have c88777 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.1
  have c88778 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88779 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88780 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88781 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_8
  simp only [← c88776, ← c88777, ← c88778] at e_1446_8
  have e_1446_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_9
  simp only [← c88779, ← c88780, k2, ← c88781] at e_1446_9
  have hr := e_1446_9
  simp only [e_1446_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13750)) := by
  have c88782 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88783 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88784 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88785 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88786 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88787 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_10
  simp only [← c88782, ← c88783, ← c88784] at e_1446_10
  have e_1446_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_11
  simp only [← c88785, ← c88786, k2, ← c88787] at e_1446_11
  have hr := e_1446_11
  simp only [e_1446_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13751)) := by
  have c88788 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88789 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88790 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88791 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88792 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88793 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_12
  simp only [← c88788, ← c88789, ← c88790] at e_1446_12
  have e_1446_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_13
  simp only [← c88791, ← c88792, k2, ← c88793] at e_1446_13
  have hr := e_1446_13
  simp only [e_1446_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13752)) := by
  have c88794 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88795 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88796 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88797 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88798 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88799 := (publicBatchWrapper89_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_14
  simp only [← c88794, ← c88795, ← c88796] at e_1446_14
  have e_1446_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_15
  simp only [← c88797, ← c88798, k2, ← c88799] at e_1446_15
  have hr := e_1446_15
  simp only [e_1446_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13753)) := by
  have c88800 := (publicBatchWrapper89_copies2775 a h).1
  have c88801 := (publicBatchWrapper89_copies2775 a h).2.1
  have c88802 := (publicBatchWrapper89_copies2775 a h).2.2.1
  have c88803 := (publicBatchWrapper89_copies2775 a h).2.2.2.1
  have c88804 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.1
  have c88805 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_16
  simp only [← c88800, ← c88801, ← c88802] at e_1446_16
  have e_1446_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_17
  simp only [← c88803, ← c88804, k2, ← c88805] at e_1446_17
  have hr := e_1446_17
  simp only [e_1446_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1446 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13754)) := by
  have c88806 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.1
  have c88807 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.1
  have c88808 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.1
  have c88809 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.1
  have c88810 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88811 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1446_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_18
  simp only [← c88806, ← c88807, ← c88808] at e_1446_18
  have e_1446_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1446 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_19
  simp only [← c88809, ← c88810, k2, ← c88811] at e_1446_19
  have hr := e_1446_19
  simp only [e_1446_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13755)) := by
  have c88812 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88813 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88814 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88815 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88816 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88817 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_0
  simp only [← c88812, ← c88813, ← c88814] at e_1447_0
  have e_1447_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_1
  simp only [← c88815, ← c88816, k2, ← c88817] at e_1447_1
  have hr := e_1447_1
  simp only [e_1447_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13756)) := by
  have c88818 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88819 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88820 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88821 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88822 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88823 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_2
  simp only [← c88818, ← c88819, ← c88820] at e_1447_2
  have e_1447_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_3
  simp only [← c88821, ← c88822, k2, ← c88823] at e_1447_3
  have hr := e_1447_3
  simp only [e_1447_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13757)) := by
  have c88824 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88825 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88826 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88827 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88828 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88829 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_4
  simp only [← c88824, ← c88825, ← c88826] at e_1447_4
  have e_1447_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_5
  simp only [← c88827, ← c88828, k2, ← c88829] at e_1447_5
  have hr := e_1447_5
  simp only [e_1447_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13758)) := by
  have c88830 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88831 := (publicBatchWrapper89_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88832 := (publicBatchWrapper89_copies2776 a h).1
  have c88833 := (publicBatchWrapper89_copies2776 a h).2.1
  have c88834 := (publicBatchWrapper89_copies2776 a h).2.2.1
  have c88835 := (publicBatchWrapper89_copies2776 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_6
  simp only [← c88830, ← c88831, ← c88832] at e_1447_6
  have e_1447_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_7
  simp only [← c88833, ← c88834, k2, ← c88835] at e_1447_7
  have hr := e_1447_7
  simp only [e_1447_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13759)) := by
  have c88836 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.1
  have c88837 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.1
  have c88838 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.1
  have c88839 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.1
  have c88840 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.1
  have c88841 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_8
  simp only [← c88836, ← c88837, ← c88838] at e_1447_8
  have e_1447_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_9
  simp only [← c88839, ← c88840, k2, ← c88841] at e_1447_9
  have hr := e_1447_9
  simp only [e_1447_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13760)) := by
  have c88842 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88843 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88844 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88845 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88846 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88847 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_10
  simp only [← c88842, ← c88843, ← c88844] at e_1447_10
  have e_1447_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_11
  simp only [← c88845, ← c88846, k2, ← c88847] at e_1447_11
  have hr := e_1447_11
  simp only [e_1447_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13761)) := by
  have c88848 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88849 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88850 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88851 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88852 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88853 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_12
  simp only [← c88848, ← c88849, ← c88850] at e_1447_12
  have e_1447_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_13
  simp only [← c88851, ← c88852, k2, ← c88853] at e_1447_13
  have hr := e_1447_13
  simp only [e_1447_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13762)) := by
  have c88854 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88855 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88856 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88857 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88858 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88859 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_14
  simp only [← c88854, ← c88855, ← c88856] at e_1447_14
  have e_1447_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_15
  simp only [← c88857, ← c88858, k2, ← c88859] at e_1447_15
  have hr := e_1447_15
  simp only [e_1447_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13763)) := by
  have c88860 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88861 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88862 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88863 := (publicBatchWrapper89_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88864 := (publicBatchWrapper89_copies2777 a h).1
  have c88865 := (publicBatchWrapper89_copies2777 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_16
  simp only [← c88860, ← c88861, ← c88862] at e_1447_16
  have e_1447_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_17
  simp only [← c88863, ← c88864, k2, ← c88865] at e_1447_17
  have hr := e_1447_17
  simp only [e_1447_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1447 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13764)) := by
  have c88866 := (publicBatchWrapper89_copies2777 a h).2.2.1
  have c88867 := (publicBatchWrapper89_copies2777 a h).2.2.2.1
  have c88868 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.1
  have c88869 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.1
  have c88870 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.1
  have c88871 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1447_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_18
  simp only [← c88866, ← c88867, ← c88868] at e_1447_18
  have e_1447_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1447 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_19
  simp only [← c88869, ← c88870, k2, ← c88871] at e_1447_19
  have hr := e_1447_19
  simp only [e_1447_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13765)) := by
  have c88872 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.1
  have c88873 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.1
  have c88874 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88875 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88876 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88877 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_0
  simp only [← c88872, ← c88873, ← c88874] at e_1448_0
  have e_1448_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_1
  simp only [← c88875, ← c88876, k2, ← c88877] at e_1448_1
  have hr := e_1448_1
  simp only [e_1448_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13766)) := by
  have c88878 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88879 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88880 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88881 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88882 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88883 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_2
  simp only [← c88878, ← c88879, ← c88880] at e_1448_2
  have e_1448_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_3
  simp only [← c88881, ← c88882, k2, ← c88883] at e_1448_3
  have hr := e_1448_3
  simp only [e_1448_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13767)) := by
  have c88884 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88885 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88886 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88887 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88888 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88889 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_4
  simp only [← c88884, ← c88885, ← c88886] at e_1448_4
  have e_1448_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_5
  simp only [← c88887, ← c88888, k2, ← c88889] at e_1448_5
  have hr := e_1448_5
  simp only [e_1448_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13768)) := by
  have c88890 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88891 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88892 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88893 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88894 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88895 := (publicBatchWrapper89_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_6
  simp only [← c88890, ← c88891, ← c88892] at e_1448_6
  have e_1448_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_7
  simp only [← c88893, ← c88894, k2, ← c88895] at e_1448_7
  have hr := e_1448_7
  simp only [e_1448_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13769)) := by
  have c88896 := (publicBatchWrapper89_copies2778 a h).1
  have c88897 := (publicBatchWrapper89_copies2778 a h).2.1
  have c88898 := (publicBatchWrapper89_copies2778 a h).2.2.1
  have c88899 := (publicBatchWrapper89_copies2778 a h).2.2.2.1
  have c88900 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.1
  have c88901 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_8
  simp only [← c88896, ← c88897, ← c88898] at e_1448_8
  have e_1448_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_9
  simp only [← c88899, ← c88900, k2, ← c88901] at e_1448_9
  have hr := e_1448_9
  simp only [e_1448_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13810)) := by
  have c88902 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.1
  have c88903 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.1
  have c88904 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.1
  have c88905 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.1
  have c88906 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88907 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_10
  simp only [← c88902, ← c88903, ← c88904] at e_1448_10
  have e_1448_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_11
  simp only [← c88905, ← c88906, k2, ← c88907] at e_1448_11
  have hr := e_1448_11
  simp only [e_1448_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13811)) := by
  have c88908 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88909 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88910 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88911 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88912 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88913 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_12
  simp only [← c88908, ← c88909, ← c88910] at e_1448_12
  have e_1448_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_13
  simp only [← c88911, ← c88912, k2, ← c88913] at e_1448_13
  have hr := e_1448_13
  simp only [e_1448_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13812)) := by
  have c88914 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88915 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88916 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88917 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88918 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88919 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_14
  simp only [← c88914, ← c88915, ← c88916] at e_1448_14
  have e_1448_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_15
  simp only [← c88917, ← c88918, k2, ← c88919] at e_1448_15
  have hr := e_1448_15
  simp only [e_1448_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13813)) := by
  have c88920 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88921 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88922 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88923 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88924 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88925 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_16
  simp only [← c88920, ← c88921, ← c88922] at e_1448_16
  have e_1448_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_17
  simp only [← c88923, ← c88924, k2, ← c88925] at e_1448_17
  have hr := e_1448_17
  simp only [e_1448_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1448 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13814)) := by
  have c88926 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88927 := (publicBatchWrapper89_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88928 := (publicBatchWrapper89_copies2779 a h).1
  have c88929 := (publicBatchWrapper89_copies2779 a h).2.1
  have c88930 := (publicBatchWrapper89_copies2779 a h).2.2.1
  have c88931 := (publicBatchWrapper89_copies2779 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1448_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_18
  simp only [← c88926, ← c88927, ← c88928] at e_1448_18
  have e_1448_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1448 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_19
  simp only [← c88929, ← c88930, k2, ← c88931] at e_1448_19
  have hr := e_1448_19
  simp only [e_1448_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13815)) := by
  have c88932 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.1
  have c88933 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.1
  have c88934 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.1
  have c88935 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.1
  have c88936 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.1
  have c88937 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_0
  simp only [← c88932, ← c88933, ← c88934] at e_1449_0
  have e_1449_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_1
  simp only [← c88935, ← c88936, k2, ← c88937] at e_1449_1
  have hr := e_1449_1
  simp only [e_1449_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13816)) := by
  have c88938 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88939 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88940 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88941 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88942 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88943 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_2
  simp only [← c88938, ← c88939, ← c88940] at e_1449_2
  have e_1449_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_3
  simp only [← c88941, ← c88942, k2, ← c88943] at e_1449_3
  have hr := e_1449_3
  simp only [e_1449_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13817)) := by
  have c88944 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88945 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88946 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88947 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88948 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88949 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_4
  simp only [← c88944, ← c88945, ← c88946] at e_1449_4
  have e_1449_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_5
  simp only [← c88947, ← c88948, k2, ← c88949] at e_1449_5
  have hr := e_1449_5
  simp only [e_1449_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13818)) := by
  have c88950 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88951 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88952 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88953 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88954 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88955 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_6
  simp only [← c88950, ← c88951, ← c88952] at e_1449_6
  have e_1449_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_7
  simp only [← c88953, ← c88954, k2, ← c88955] at e_1449_7
  have hr := e_1449_7
  simp only [e_1449_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13819)) := by
  have c88956 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88957 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88958 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88959 := (publicBatchWrapper89_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88960 := (publicBatchWrapper89_copies2780 a h).1
  have c88961 := (publicBatchWrapper89_copies2780 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_8
  simp only [← c88956, ← c88957, ← c88958] at e_1449_8
  have e_1449_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_9
  simp only [← c88959, ← c88960, k2, ← c88961] at e_1449_9
  have hr := e_1449_9
  simp only [e_1449_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13820)) := by
  have c88962 := (publicBatchWrapper89_copies2780 a h).2.2.1
  have c88963 := (publicBatchWrapper89_copies2780 a h).2.2.2.1
  have c88964 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.1
  have c88965 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.1
  have c88966 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.1
  have c88967 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_10
  simp only [← c88962, ← c88963, ← c88964] at e_1449_10
  have e_1449_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_11
  simp only [← c88965, ← c88966, k2, ← c88967] at e_1449_11
  have hr := e_1449_11
  simp only [e_1449_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13821)) := by
  have c88968 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.1
  have c88969 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.1
  have c88970 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88971 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88972 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88973 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_12
  simp only [← c88968, ← c88969, ← c88970] at e_1449_12
  have e_1449_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_13
  simp only [← c88971, ← c88972, k2, ← c88973] at e_1449_13
  have hr := e_1449_13
  simp only [e_1449_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13822)) := by
  have c88974 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88975 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88976 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88977 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88978 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88979 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_14
  simp only [← c88974, ← c88975, ← c88976] at e_1449_14
  have e_1449_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_15
  simp only [← c88977, ← c88978, k2, ← c88979] at e_1449_15
  have hr := e_1449_15
  simp only [e_1449_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13823)) := by
  have c88980 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88981 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88982 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88983 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88984 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88985 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_16
  simp only [← c88980, ← c88981, ← c88982] at e_1449_16
  have e_1449_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_17
  simp only [← c88983, ← c88984, k2, ← c88985] at e_1449_17
  have hr := e_1449_17
  simp only [e_1449_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1449 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13824)) := by
  have c88986 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88987 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88988 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88989 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88990 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88991 := (publicBatchWrapper89_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1449_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_18
  simp only [← c88986, ← c88987, ← c88988] at e_1449_18
  have e_1449_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1449 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_19
  simp only [← c88989, ← c88990, k2, ← c88991] at e_1449_19
  have hr := e_1449_19
  simp only [e_1449_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13825)) := by
  have c88992 := (publicBatchWrapper89_copies2781 a h).1
  have c88993 := (publicBatchWrapper89_copies2781 a h).2.1
  have c88994 := (publicBatchWrapper89_copies2781 a h).2.2.1
  have c88995 := (publicBatchWrapper89_copies2781 a h).2.2.2.1
  have c88996 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.1
  have c88997 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_0
  simp only [← c88992, ← c88993, ← c88994] at e_1450_0
  have e_1450_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_1
  simp only [← c88995, ← c88996, k2, ← c88997] at e_1450_1
  have hr := e_1450_1
  simp only [e_1450_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13826)) := by
  have c88998 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.1
  have c88999 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.1
  have c89000 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.1
  have c89001 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.1
  have c89002 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89003 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_2
  simp only [← c88998, ← c88999, ← c89000] at e_1450_2
  have e_1450_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_3
  simp only [← c89001, ← c89002, k2, ← c89003] at e_1450_3
  have hr := e_1450_3
  simp only [e_1450_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13827)) := by
  have c89004 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89005 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89006 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89007 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89008 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89009 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_4
  simp only [← c89004, ← c89005, ← c89006] at e_1450_4
  have e_1450_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_5
  simp only [← c89007, ← c89008, k2, ← c89009] at e_1450_5
  have hr := e_1450_5
  simp only [e_1450_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13828)) := by
  have c89010 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89011 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89012 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89013 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89014 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89015 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_6
  simp only [← c89010, ← c89011, ← c89012] at e_1450_6
  have e_1450_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_7
  simp only [← c89013, ← c89014, k2, ← c89015] at e_1450_7
  have hr := e_1450_7
  simp only [e_1450_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13829)) := by
  have c89016 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89017 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89018 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89019 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89020 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89021 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_8
  simp only [← c89016, ← c89017, ← c89018] at e_1450_8
  have e_1450_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_9
  simp only [← c89019, ← c89020, k2, ← c89021] at e_1450_9
  have hr := e_1450_9
  simp only [e_1450_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13830)) := by
  have c89022 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89023 := (publicBatchWrapper89_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89024 := (publicBatchWrapper89_copies2782 a h).1
  have c89025 := (publicBatchWrapper89_copies2782 a h).2.1
  have c89026 := (publicBatchWrapper89_copies2782 a h).2.2.1
  have c89027 := (publicBatchWrapper89_copies2782 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_10
  simp only [← c89022, ← c89023, ← c89024] at e_1450_10
  have e_1450_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_11
  simp only [← c89025, ← c89026, k2, ← c89027] at e_1450_11
  have hr := e_1450_11
  simp only [e_1450_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13831)) := by
  have c89028 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.1
  have c89029 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.1
  have c89030 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.1
  have c89031 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.1
  have c89032 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.1
  have c89033 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_12
  simp only [← c89028, ← c89029, ← c89030] at e_1450_12
  have e_1450_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_13
  simp only [← c89031, ← c89032, k2, ← c89033] at e_1450_13
  have hr := e_1450_13
  simp only [e_1450_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13832)) := by
  have c89034 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89035 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89036 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89037 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89038 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89039 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_14
  simp only [← c89034, ← c89035, ← c89036] at e_1450_14
  have e_1450_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_15
  simp only [← c89037, ← c89038, k2, ← c89039] at e_1450_15
  have hr := e_1450_15
  simp only [e_1450_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13833)) := by
  have c89040 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89041 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89042 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89043 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89044 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89045 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_16
  simp only [← c89040, ← c89041, ← c89042] at e_1450_16
  have e_1450_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_17
  simp only [← c89043, ← c89044, k2, ← c89045] at e_1450_17
  have hr := e_1450_17
  simp only [e_1450_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1450 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13834)) := by
  have c89046 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89047 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89048 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89049 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89050 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89051 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1450_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_18
  simp only [← c89046, ← c89047, ← c89048] at e_1450_18
  have e_1450_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1450 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_19
  simp only [← c89049, ← c89050, k2, ← c89051] at e_1450_19
  have hr := e_1450_19
  simp only [e_1450_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13835)) := by
  have c89052 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89053 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89054 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89055 := (publicBatchWrapper89_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89056 := (publicBatchWrapper89_copies2783 a h).1
  have c89057 := (publicBatchWrapper89_copies2783 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_0
  simp only [← c89052, ← c89053, ← c89054] at e_1451_0
  have e_1451_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_1
  simp only [← c89055, ← c89056, k2, ← c89057] at e_1451_1
  have hr := e_1451_1
  simp only [e_1451_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13836)) := by
  have c89058 := (publicBatchWrapper89_copies2783 a h).2.2.1
  have c89059 := (publicBatchWrapper89_copies2783 a h).2.2.2.1
  have c89060 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.1
  have c89061 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.1
  have c89062 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.1
  have c89063 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_2
  simp only [← c89058, ← c89059, ← c89060] at e_1451_2
  have e_1451_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_3
  simp only [← c89061, ← c89062, k2, ← c89063] at e_1451_3
  have hr := e_1451_3
  simp only [e_1451_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13837)) := by
  have c89064 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.1
  have c89065 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.1
  have c89066 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89067 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89068 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89069 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_4
  simp only [← c89064, ← c89065, ← c89066] at e_1451_4
  have e_1451_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_5
  simp only [← c89067, ← c89068, k2, ← c89069] at e_1451_5
  have hr := e_1451_5
  simp only [e_1451_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13838)) := by
  have c89070 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89071 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89072 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89073 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89074 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89075 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_6
  simp only [← c89070, ← c89071, ← c89072] at e_1451_6
  have e_1451_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_7
  simp only [← c89073, ← c89074, k2, ← c89075] at e_1451_7
  have hr := e_1451_7
  simp only [e_1451_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13839)) := by
  have c89076 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89077 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89078 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89079 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89080 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89081 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_8
  simp only [← c89076, ← c89077, ← c89078] at e_1451_8
  have e_1451_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_9
  simp only [← c89079, ← c89080, k2, ← c89081] at e_1451_9
  have hr := e_1451_9
  simp only [e_1451_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13840)) := by
  have c89082 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89083 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89084 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89085 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89086 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89087 := (publicBatchWrapper89_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_10
  simp only [← c89082, ← c89083, ← c89084] at e_1451_10
  have e_1451_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_11
  simp only [← c89085, ← c89086, k2, ← c89087] at e_1451_11
  have hr := e_1451_11
  simp only [e_1451_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13841)) := by
  have c89088 := (publicBatchWrapper89_copies2784 a h).1
  have c89089 := (publicBatchWrapper89_copies2784 a h).2.1
  have c89090 := (publicBatchWrapper89_copies2784 a h).2.2.1
  have c89091 := (publicBatchWrapper89_copies2784 a h).2.2.2.1
  have c89092 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.1
  have c89093 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_12
  simp only [← c89088, ← c89089, ← c89090] at e_1451_12
  have e_1451_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_13
  simp only [← c89091, ← c89092, k2, ← c89093] at e_1451_13
  have hr := e_1451_13
  simp only [e_1451_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13842)) := by
  have c89094 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.1
  have c89095 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.1
  have c89096 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.1
  have c89097 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.1
  have c89098 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89099 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_14
  simp only [← c89094, ← c89095, ← c89096] at e_1451_14
  have e_1451_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_15
  simp only [← c89097, ← c89098, k2, ← c89099] at e_1451_15
  have hr := e_1451_15
  simp only [e_1451_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13843)) := by
  have c89100 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89101 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89102 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89103 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89104 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89105 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_16
  simp only [← c89100, ← c89101, ← c89102] at e_1451_16
  have e_1451_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_17
  simp only [← c89103, ← c89104, k2, ← c89105] at e_1451_17
  have hr := e_1451_17
  simp only [e_1451_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1451 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13844)) := by
  have c89106 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89107 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89108 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89109 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89110 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89111 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1451_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_18
  simp only [← c89106, ← c89107, ← c89108] at e_1451_18
  have e_1451_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1451 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_19
  simp only [← c89109, ← c89110, k2, ← c89111] at e_1451_19
  have hr := e_1451_19
  simp only [e_1451_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13845)) := by
  have c89112 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89113 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89114 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89115 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89116 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89117 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_0
  simp only [← c89112, ← c89113, ← c89114] at e_1452_0
  have e_1452_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_1
  simp only [← c89115, ← c89116, k2, ← c89117] at e_1452_1
  have hr := e_1452_1
  simp only [e_1452_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13846)) := by
  have c89118 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89119 := (publicBatchWrapper89_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89120 := (publicBatchWrapper89_copies2785 a h).1
  have c89121 := (publicBatchWrapper89_copies2785 a h).2.1
  have c89122 := (publicBatchWrapper89_copies2785 a h).2.2.1
  have c89123 := (publicBatchWrapper89_copies2785 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_2
  simp only [← c89118, ← c89119, ← c89120] at e_1452_2
  have e_1452_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_3
  simp only [← c89121, ← c89122, k2, ← c89123] at e_1452_3
  have hr := e_1452_3
  simp only [e_1452_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13847)) := by
  have c89124 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.1
  have c89125 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.1
  have c89126 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.1
  have c89127 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.1
  have c89128 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.1
  have c89129 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_4
  simp only [← c89124, ← c89125, ← c89126] at e_1452_4
  have e_1452_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_5
  simp only [← c89127, ← c89128, k2, ← c89129] at e_1452_5
  have hr := e_1452_5
  simp only [e_1452_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13848)) := by
  have c89130 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89131 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89132 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89133 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89134 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89135 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_6
  simp only [← c89130, ← c89131, ← c89132] at e_1452_6
  have e_1452_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_7
  simp only [← c89133, ← c89134, k2, ← c89135] at e_1452_7
  have hr := e_1452_7
  simp only [e_1452_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13849)) := by
  have c89136 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89137 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89138 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89139 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89140 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89141 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_8
  simp only [← c89136, ← c89137, ← c89138] at e_1452_8
  have e_1452_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_9
  simp only [← c89139, ← c89140, k2, ← c89141] at e_1452_9
  have hr := e_1452_9
  simp only [e_1452_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13850)) := by
  have c89142 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89143 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89144 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89145 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89146 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89147 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_10
  simp only [← c89142, ← c89143, ← c89144] at e_1452_10
  have e_1452_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_11
  simp only [← c89145, ← c89146, k2, ← c89147] at e_1452_11
  have hr := e_1452_11
  simp only [e_1452_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13851)) := by
  have c89148 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89149 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89150 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89151 := (publicBatchWrapper89_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89152 := (publicBatchWrapper89_copies2786 a h).1
  have c89153 := (publicBatchWrapper89_copies2786 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_12
  simp only [← c89148, ← c89149, ← c89150] at e_1452_12
  have e_1452_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_13
  simp only [← c89151, ← c89152, k2, ← c89153] at e_1452_13
  have hr := e_1452_13
  simp only [e_1452_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13852)) := by
  have c89154 := (publicBatchWrapper89_copies2786 a h).2.2.1
  have c89155 := (publicBatchWrapper89_copies2786 a h).2.2.2.1
  have c89156 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.1
  have c89157 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.1
  have c89158 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.1
  have c89159 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_14
  simp only [← c89154, ← c89155, ← c89156] at e_1452_14
  have e_1452_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_15
  simp only [← c89157, ← c89158, k2, ← c89159] at e_1452_15
  have hr := e_1452_15
  simp only [e_1452_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13853)) := by
  have c89160 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.1
  have c89161 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.1
  have c89162 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89163 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89164 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89165 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_16
  simp only [← c89160, ← c89161, ← c89162] at e_1452_16
  have e_1452_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_17
  simp only [← c89163, ← c89164, k2, ← c89165] at e_1452_17
  have hr := e_1452_17
  simp only [e_1452_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1452 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13854)) := by
  have c89166 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89167 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89168 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89169 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89170 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89171 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1452_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_18
  simp only [← c89166, ← c89167, ← c89168] at e_1452_18
  have e_1452_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1452 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_19
  simp only [← c89169, ← c89170, k2, ← c89171] at e_1452_19
  have hr := e_1452_19
  simp only [e_1452_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13855)) := by
  have c89172 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89173 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89174 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89175 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89176 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89177 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_0
  simp only [← c89172, ← c89173, ← c89174] at e_1453_0
  have e_1453_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_1
  simp only [← c89175, ← c89176, k2, ← c89177] at e_1453_1
  have hr := e_1453_1
  simp only [e_1453_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13856)) := by
  have c89178 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89179 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89180 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89181 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89182 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89183 := (publicBatchWrapper89_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_2
  simp only [← c89178, ← c89179, ← c89180] at e_1453_2
  have e_1453_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_3
  simp only [← c89181, ← c89182, k2, ← c89183] at e_1453_3
  have hr := e_1453_3
  simp only [e_1453_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13857)) := by
  have c89184 := (publicBatchWrapper89_copies2787 a h).1
  have c89185 := (publicBatchWrapper89_copies2787 a h).2.1
  have c89186 := (publicBatchWrapper89_copies2787 a h).2.2.1
  have c89187 := (publicBatchWrapper89_copies2787 a h).2.2.2.1
  have c89188 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.1
  have c89189 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_4
  simp only [← c89184, ← c89185, ← c89186] at e_1453_4
  have e_1453_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_5
  simp only [← c89187, ← c89188, k2, ← c89189] at e_1453_5
  have hr := e_1453_5
  simp only [e_1453_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13858)) := by
  have c89190 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.1
  have c89191 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.1
  have c89192 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.1
  have c89193 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.1
  have c89194 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89195 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_6
  simp only [← c89190, ← c89191, ← c89192] at e_1453_6
  have e_1453_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_7
  simp only [← c89193, ← c89194, k2, ← c89195] at e_1453_7
  have hr := e_1453_7
  simp only [e_1453_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13859)) := by
  have c89196 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89197 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89198 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89199 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89200 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89201 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_8
  simp only [← c89196, ← c89197, ← c89198] at e_1453_8
  have e_1453_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_9
  simp only [← c89199, ← c89200, k2, ← c89201] at e_1453_9
  have hr := e_1453_9
  simp only [e_1453_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13860)) := by
  have c89202 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89203 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89204 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89205 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89206 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89207 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_10
  simp only [← c89202, ← c89203, ← c89204] at e_1453_10
  have e_1453_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_11
  simp only [← c89205, ← c89206, k2, ← c89207] at e_1453_11
  have hr := e_1453_11
  simp only [e_1453_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13861)) := by
  have c89208 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89209 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89210 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89211 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89212 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89213 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_12
  simp only [← c89208, ← c89209, ← c89210] at e_1453_12
  have e_1453_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_13
  simp only [← c89211, ← c89212, k2, ← c89213] at e_1453_13
  have hr := e_1453_13
  simp only [e_1453_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13862)) := by
  have c89214 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89215 := (publicBatchWrapper89_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89216 := (publicBatchWrapper89_copies2788 a h).1
  have c89217 := (publicBatchWrapper89_copies2788 a h).2.1
  have c89218 := (publicBatchWrapper89_copies2788 a h).2.2.1
  have c89219 := (publicBatchWrapper89_copies2788 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_14
  simp only [← c89214, ← c89215, ← c89216] at e_1453_14
  have e_1453_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_15
  simp only [← c89217, ← c89218, k2, ← c89219] at e_1453_15
  have hr := e_1453_15
  simp only [e_1453_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13863)) := by
  have c89220 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.1
  have c89221 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.1
  have c89222 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.1
  have c89223 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.1
  have c89224 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.1
  have c89225 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_16
  simp only [← c89220, ← c89221, ← c89222] at e_1453_16
  have e_1453_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_17
  simp only [← c89223, ← c89224, k2, ← c89225] at e_1453_17
  have hr := e_1453_17
  simp only [e_1453_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1453 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13864)) := by
  have c89226 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89227 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89228 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89229 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89230 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89231 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1453_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_18
  simp only [← c89226, ← c89227, ← c89228] at e_1453_18
  have e_1453_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1453 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_19
  simp only [← c89229, ← c89230, k2, ← c89231] at e_1453_19
  have hr := e_1453_19
  simp only [e_1453_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13865)) := by
  have c89232 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89233 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89234 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89235 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89236 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89237 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_0
  simp only [← c89232, ← c89233, ← c89234] at e_1454_0
  have e_1454_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_1
  simp only [← c89235, ← c89236, k2, ← c89237] at e_1454_1
  have hr := e_1454_1
  simp only [e_1454_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13866)) := by
  have c89238 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89239 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89240 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89241 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89242 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89243 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_2
  simp only [← c89238, ← c89239, ← c89240] at e_1454_2
  have e_1454_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_3
  simp only [← c89241, ← c89242, k2, ← c89243] at e_1454_3
  have hr := e_1454_3
  simp only [e_1454_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13867)) := by
  have c89244 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89245 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89246 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89247 := (publicBatchWrapper89_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89248 := (publicBatchWrapper89_copies2789 a h).1
  have c89249 := (publicBatchWrapper89_copies2789 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_4
  simp only [← c89244, ← c89245, ← c89246] at e_1454_4
  have e_1454_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_5
  simp only [← c89247, ← c89248, k2, ← c89249] at e_1454_5
  have hr := e_1454_5
  simp only [e_1454_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13868)) := by
  have c89250 := (publicBatchWrapper89_copies2789 a h).2.2.1
  have c89251 := (publicBatchWrapper89_copies2789 a h).2.2.2.1
  have c89252 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.1
  have c89253 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.1
  have c89254 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.1
  have c89255 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_6
  simp only [← c89250, ← c89251, ← c89252] at e_1454_6
  have e_1454_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_7
  simp only [← c89253, ← c89254, k2, ← c89255] at e_1454_7
  have hr := e_1454_7
  simp only [e_1454_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13869)) := by
  have c89256 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.1
  have c89257 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.1
  have c89258 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89259 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89260 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89261 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_8
  simp only [← c89256, ← c89257, ← c89258] at e_1454_8
  have e_1454_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_9
  simp only [← c89259, ← c89260, k2, ← c89261] at e_1454_9
  have hr := e_1454_9
  simp only [e_1454_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13870)) := by
  have c89262 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89263 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89264 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89265 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89266 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89267 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_10
  simp only [← c89262, ← c89263, ← c89264] at e_1454_10
  have e_1454_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_11
  simp only [← c89265, ← c89266, k2, ← c89267] at e_1454_11
  have hr := e_1454_11
  simp only [e_1454_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13871)) := by
  have c89268 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89269 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89270 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89271 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89272 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89273 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_12
  simp only [← c89268, ← c89269, ← c89270] at e_1454_12
  have e_1454_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_13
  simp only [← c89271, ← c89272, k2, ← c89273] at e_1454_13
  have hr := e_1454_13
  simp only [e_1454_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13872)) := by
  have c89274 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89275 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89276 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89277 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89278 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89279 := (publicBatchWrapper89_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_14
  simp only [← c89274, ← c89275, ← c89276] at e_1454_14
  have e_1454_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_15
  simp only [← c89277, ← c89278, k2, ← c89279] at e_1454_15
  have hr := e_1454_15
  simp only [e_1454_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13873)) := by
  have c89280 := (publicBatchWrapper89_copies2790 a h).1
  have c89281 := (publicBatchWrapper89_copies2790 a h).2.1
  have c89282 := (publicBatchWrapper89_copies2790 a h).2.2.1
  have c89283 := (publicBatchWrapper89_copies2790 a h).2.2.2.1
  have c89284 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.1
  have c89285 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_16
  simp only [← c89280, ← c89281, ← c89282] at e_1454_16
  have e_1454_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_17
  simp only [← c89283, ← c89284, k2, ← c89285] at e_1454_17
  have hr := e_1454_17
  simp only [e_1454_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1454 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13874)) := by
  have c89286 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.1
  have c89287 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.1
  have c89288 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.1
  have c89289 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.1
  have c89290 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89291 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1454_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_18
  simp only [← c89286, ← c89287, ← c89288] at e_1454_18
  have e_1454_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1454 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_19
  simp only [← c89289, ← c89290, k2, ← c89291] at e_1454_19
  have hr := e_1454_19
  simp only [e_1454_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13875)) := by
  have c89292 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89293 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89294 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89295 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89296 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89297 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_0
  simp only [← c89292, ← c89293, ← c89294] at e_1455_0
  have e_1455_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_1
  simp only [← c89295, ← c89296, k2, ← c89297] at e_1455_1
  have hr := e_1455_1
  simp only [e_1455_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13876)) := by
  have c89298 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89299 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89300 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89301 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89302 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89303 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_2
  simp only [← c89298, ← c89299, ← c89300] at e_1455_2
  have e_1455_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_3
  simp only [← c89301, ← c89302, k2, ← c89303] at e_1455_3
  have hr := e_1455_3
  simp only [e_1455_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13877)) := by
  have c89304 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89305 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89306 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89307 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89308 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89309 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_4
  simp only [← c89304, ← c89305, ← c89306] at e_1455_4
  have e_1455_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_5
  simp only [← c89307, ← c89308, k2, ← c89309] at e_1455_5
  have hr := e_1455_5
  simp only [e_1455_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13878)) := by
  have c89310 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89311 := (publicBatchWrapper89_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89312 := (publicBatchWrapper89_copies2791 a h).1
  have c89313 := (publicBatchWrapper89_copies2791 a h).2.1
  have c89314 := (publicBatchWrapper89_copies2791 a h).2.2.1
  have c89315 := (publicBatchWrapper89_copies2791 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_6
  simp only [← c89310, ← c89311, ← c89312] at e_1455_6
  have e_1455_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_7
  simp only [← c89313, ← c89314, k2, ← c89315] at e_1455_7
  have hr := e_1455_7
  simp only [e_1455_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13879)) := by
  have c89316 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.1
  have c89317 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.1
  have c89318 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.1
  have c89319 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.1
  have c89320 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.1
  have c89321 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_8
  simp only [← c89316, ← c89317, ← c89318] at e_1455_8
  have e_1455_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_9
  simp only [← c89319, ← c89320, k2, ← c89321] at e_1455_9
  have hr := e_1455_9
  simp only [e_1455_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13880)) := by
  have c89322 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89323 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89324 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89325 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89326 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89327 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_10
  simp only [← c89322, ← c89323, ← c89324] at e_1455_10
  have e_1455_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_11
  simp only [← c89325, ← c89326, k2, ← c89327] at e_1455_11
  have hr := e_1455_11
  simp only [e_1455_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13881)) := by
  have c89328 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89329 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89330 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89331 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89332 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89333 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_12
  simp only [← c89328, ← c89329, ← c89330] at e_1455_12
  have e_1455_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_13
  simp only [← c89331, ← c89332, k2, ← c89333] at e_1455_13
  have hr := e_1455_13
  simp only [e_1455_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13882)) := by
  have c89334 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89335 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89336 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89337 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89338 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89339 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_14
  simp only [← c89334, ← c89335, ← c89336] at e_1455_14
  have e_1455_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_15
  simp only [← c89337, ← c89338, k2, ← c89339] at e_1455_15
  have hr := e_1455_15
  simp only [e_1455_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13883)) := by
  have c89340 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89341 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89342 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89343 := (publicBatchWrapper89_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89344 := (publicBatchWrapper89_copies2792 a h).1
  have c89345 := (publicBatchWrapper89_copies2792 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_16
  simp only [← c89340, ← c89341, ← c89342] at e_1455_16
  have e_1455_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_17
  simp only [← c89343, ← c89344, k2, ← c89345] at e_1455_17
  have hr := e_1455_17
  simp only [e_1455_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1455 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13884)) := by
  have c89346 := (publicBatchWrapper89_copies2792 a h).2.2.1
  have c89347 := (publicBatchWrapper89_copies2792 a h).2.2.2.1
  have c89348 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.1
  have c89349 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.1
  have c89350 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.1
  have c89351 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1455_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_18
  simp only [← c89346, ← c89347, ← c89348] at e_1455_18
  have e_1455_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1455 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_19
  simp only [← c89349, ← c89350, k2, ← c89351] at e_1455_19
  have hr := e_1455_19
  simp only [e_1455_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13885)) := by
  have c89352 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.1
  have c89353 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.1
  have c89354 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89355 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89356 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89357 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_0
  simp only [← c89352, ← c89353, ← c89354] at e_1456_0
  have e_1456_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_1
  simp only [← c89355, ← c89356, k2, ← c89357] at e_1456_1
  have hr := e_1456_1
  simp only [e_1456_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13886)) := by
  have c89358 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89359 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89360 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89361 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89362 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89363 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_2
  simp only [← c89358, ← c89359, ← c89360] at e_1456_2
  have e_1456_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_3
  simp only [← c89361, ← c89362, k2, ← c89363] at e_1456_3
  have hr := e_1456_3
  simp only [e_1456_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13887)) := by
  have c89364 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89365 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89366 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89367 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89368 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89369 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_4
  simp only [← c89364, ← c89365, ← c89366] at e_1456_4
  have e_1456_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_5
  simp only [← c89367, ← c89368, k2, ← c89369] at e_1456_5
  have hr := e_1456_5
  simp only [e_1456_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13888)) := by
  have c89370 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89371 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89372 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89373 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89374 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89375 := (publicBatchWrapper89_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_6
  simp only [← c89370, ← c89371, ← c89372] at e_1456_6
  have e_1456_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_7
  simp only [← c89373, ← c89374, k2, ← c89375] at e_1456_7
  have hr := e_1456_7
  simp only [e_1456_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13889)) := by
  have c89376 := (publicBatchWrapper89_copies2793 a h).1
  have c89377 := (publicBatchWrapper89_copies2793 a h).2.1
  have c89378 := (publicBatchWrapper89_copies2793 a h).2.2.1
  have c89379 := (publicBatchWrapper89_copies2793 a h).2.2.2.1
  have c89380 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.1
  have c89381 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_8
  simp only [← c89376, ← c89377, ← c89378] at e_1456_8
  have e_1456_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_9
  simp only [← c89379, ← c89380, k2, ← c89381] at e_1456_9
  have hr := e_1456_9
  simp only [e_1456_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13890)) := by
  have c89382 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.1
  have c89383 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.1
  have c89384 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.1
  have c89385 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.1
  have c89386 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89387 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_10
  simp only [← c89382, ← c89383, ← c89384] at e_1456_10
  have e_1456_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_11
  simp only [← c89385, ← c89386, k2, ← c89387] at e_1456_11
  have hr := e_1456_11
  simp only [e_1456_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13891)) := by
  have c89388 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89389 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89390 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89391 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89392 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89393 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_12
  simp only [← c89388, ← c89389, ← c89390] at e_1456_12
  have e_1456_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_13
  simp only [← c89391, ← c89392, k2, ← c89393] at e_1456_13
  have hr := e_1456_13
  simp only [e_1456_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13892)) := by
  have c89394 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89395 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89396 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89397 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89398 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89399 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_14
  simp only [← c89394, ← c89395, ← c89396] at e_1456_14
  have e_1456_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_15
  simp only [← c89397, ← c89398, k2, ← c89399] at e_1456_15
  have hr := e_1456_15
  simp only [e_1456_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13893)) := by
  have c89400 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89401 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89402 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89403 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89404 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89405 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_16
  simp only [← c89400, ← c89401, ← c89402] at e_1456_16
  have e_1456_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_17
  simp only [← c89403, ← c89404, k2, ← c89405] at e_1456_17
  have hr := e_1456_17
  simp only [e_1456_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1456 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13894)) := by
  have c89406 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89407 := (publicBatchWrapper89_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89408 := (publicBatchWrapper89_copies2794 a h).1
  have c89409 := (publicBatchWrapper89_copies2794 a h).2.1
  have c89410 := (publicBatchWrapper89_copies2794 a h).2.2.1
  have c89411 := (publicBatchWrapper89_copies2794 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1456_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_18
  simp only [← c89406, ← c89407, ← c89408] at e_1456_18
  have e_1456_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1456 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_19
  simp only [← c89409, ← c89410, k2, ← c89411] at e_1456_19
  have hr := e_1456_19
  simp only [e_1456_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13895)) := by
  have c89412 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.1
  have c89413 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.1
  have c89414 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.1
  have c89415 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.1
  have c89416 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.1
  have c89417 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_0
  simp only [← c89412, ← c89413, ← c89414] at e_1457_0
  have e_1457_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_1
  simp only [← c89415, ← c89416, k2, ← c89417] at e_1457_1
  have hr := e_1457_1
  simp only [e_1457_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13896)) := by
  have c89418 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89419 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89420 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89421 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89422 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89423 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_2
  simp only [← c89418, ← c89419, ← c89420] at e_1457_2
  have e_1457_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_3
  simp only [← c89421, ← c89422, k2, ← c89423] at e_1457_3
  have hr := e_1457_3
  simp only [e_1457_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13897)) := by
  have c89424 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89425 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89426 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89427 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89428 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89429 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_4
  simp only [← c89424, ← c89425, ← c89426] at e_1457_4
  have e_1457_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_5
  simp only [← c89427, ← c89428, k2, ← c89429] at e_1457_5
  have hr := e_1457_5
  simp only [e_1457_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13898)) := by
  have c89430 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89431 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89432 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89433 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89434 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89435 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_6
  simp only [← c89430, ← c89431, ← c89432] at e_1457_6
  have e_1457_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_7
  simp only [← c89433, ← c89434, k2, ← c89435] at e_1457_7
  have hr := e_1457_7
  simp only [e_1457_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13899)) := by
  have c89436 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89437 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89438 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89439 := (publicBatchWrapper89_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89440 := (publicBatchWrapper89_copies2795 a h).1
  have c89441 := (publicBatchWrapper89_copies2795 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_8
  simp only [← c89436, ← c89437, ← c89438] at e_1457_8
  have e_1457_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_9
  simp only [← c89439, ← c89440, k2, ← c89441] at e_1457_9
  have hr := e_1457_9
  simp only [e_1457_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13900)) := by
  have c89442 := (publicBatchWrapper89_copies2795 a h).2.2.1
  have c89443 := (publicBatchWrapper89_copies2795 a h).2.2.2.1
  have c89444 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.1
  have c89445 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.1
  have c89446 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.1
  have c89447 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_10
  simp only [← c89442, ← c89443, ← c89444] at e_1457_10
  have e_1457_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_11
  simp only [← c89445, ← c89446, k2, ← c89447] at e_1457_11
  have hr := e_1457_11
  simp only [e_1457_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13901)) := by
  have c89448 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.1
  have c89449 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.1
  have c89450 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89451 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89452 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89453 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_12
  simp only [← c89448, ← c89449, ← c89450] at e_1457_12
  have e_1457_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_13
  simp only [← c89451, ← c89452, k2, ← c89453] at e_1457_13
  have hr := e_1457_13
  simp only [e_1457_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13902)) := by
  have c89454 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89455 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89456 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89457 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89458 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89459 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_14
  simp only [← c89454, ← c89455, ← c89456] at e_1457_14
  have e_1457_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_15
  simp only [← c89457, ← c89458, k2, ← c89459] at e_1457_15
  have hr := e_1457_15
  simp only [e_1457_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13903)) := by
  have c89460 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89461 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89462 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89463 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89464 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89465 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_16
  simp only [← c89460, ← c89461, ← c89462] at e_1457_16
  have e_1457_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_17
  simp only [← c89463, ← c89464, k2, ← c89465] at e_1457_17
  have hr := e_1457_17
  simp only [e_1457_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1457 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13904)) := by
  have c89466 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89467 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89468 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89469 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89470 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89471 := (publicBatchWrapper89_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1457_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_18
  simp only [← c89466, ← c89467, ← c89468] at e_1457_18
  have e_1457_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1457 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_19
  simp only [← c89469, ← c89470, k2, ← c89471] at e_1457_19
  have hr := e_1457_19
  simp only [e_1457_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13905)) := by
  have c89472 := (publicBatchWrapper89_copies2796 a h).1
  have c89473 := (publicBatchWrapper89_copies2796 a h).2.1
  have c89474 := (publicBatchWrapper89_copies2796 a h).2.2.1
  have c89475 := (publicBatchWrapper89_copies2796 a h).2.2.2.1
  have c89476 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.1
  have c89477 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_0
  simp only [← c89472, ← c89473, ← c89474] at e_1458_0
  have e_1458_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_1
  simp only [← c89475, ← c89476, k2, ← c89477] at e_1458_1
  have hr := e_1458_1
  simp only [e_1458_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13906)) := by
  have c89478 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.1
  have c89479 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.1
  have c89480 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.1
  have c89481 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.1
  have c89482 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89483 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_2
  simp only [← c89478, ← c89479, ← c89480] at e_1458_2
  have e_1458_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_3
  simp only [← c89481, ← c89482, k2, ← c89483] at e_1458_3
  have hr := e_1458_3
  simp only [e_1458_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13907)) := by
  have c89484 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89485 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89486 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89487 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89488 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89489 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_4
  simp only [← c89484, ← c89485, ← c89486] at e_1458_4
  have e_1458_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_5
  simp only [← c89487, ← c89488, k2, ← c89489] at e_1458_5
  have hr := e_1458_5
  simp only [e_1458_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13908)) := by
  have c89490 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89491 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89492 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89493 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89494 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89495 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_6
  simp only [← c89490, ← c89491, ← c89492] at e_1458_6
  have e_1458_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_7
  simp only [← c89493, ← c89494, k2, ← c89495] at e_1458_7
  have hr := e_1458_7
  simp only [e_1458_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13909)) := by
  have c89496 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89497 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89498 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89499 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89500 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89501 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_8
  simp only [← c89496, ← c89497, ← c89498] at e_1458_8
  have e_1458_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_9
  simp only [← c89499, ← c89500, k2, ← c89501] at e_1458_9
  have hr := e_1458_9
  simp only [e_1458_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13910)) := by
  have c89502 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89503 := (publicBatchWrapper89_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89504 := (publicBatchWrapper89_copies2797 a h).1
  have c89505 := (publicBatchWrapper89_copies2797 a h).2.1
  have c89506 := (publicBatchWrapper89_copies2797 a h).2.2.1
  have c89507 := (publicBatchWrapper89_copies2797 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_10
  simp only [← c89502, ← c89503, ← c89504] at e_1458_10
  have e_1458_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_11
  simp only [← c89505, ← c89506, k2, ← c89507] at e_1458_11
  have hr := e_1458_11
  simp only [e_1458_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13911)) := by
  have c89508 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.1
  have c89509 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.1
  have c89510 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.1
  have c89511 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.1
  have c89512 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.1
  have c89513 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_12
  simp only [← c89508, ← c89509, ← c89510] at e_1458_12
  have e_1458_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_13
  simp only [← c89511, ← c89512, k2, ← c89513] at e_1458_13
  have hr := e_1458_13
  simp only [e_1458_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13912)) := by
  have c89514 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89515 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89516 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89517 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89518 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89519 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_14
  simp only [← c89514, ← c89515, ← c89516] at e_1458_14
  have e_1458_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_15
  simp only [← c89517, ← c89518, k2, ← c89519] at e_1458_15
  have hr := e_1458_15
  simp only [e_1458_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13913)) := by
  have c89520 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89521 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89522 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89523 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89524 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89525 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_16
  simp only [← c89520, ← c89521, ← c89522] at e_1458_16
  have e_1458_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_17
  simp only [← c89523, ← c89524, k2, ← c89525] at e_1458_17
  have hr := e_1458_17
  simp only [e_1458_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1458 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13914)) := by
  have c89526 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89527 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89528 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89529 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89530 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89531 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1458_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_18
  simp only [← c89526, ← c89527, ← c89528] at e_1458_18
  have e_1458_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1458 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_19
  simp only [← c89529, ← c89530, k2, ← c89531] at e_1458_19
  have hr := e_1458_19
  simp only [e_1458_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13915)) := by
  have c89532 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89533 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89534 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89535 := (publicBatchWrapper89_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89536 := (publicBatchWrapper89_copies2798 a h).1
  have c89537 := (publicBatchWrapper89_copies2798 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_0
  simp only [← c89532, ← c89533, ← c89534] at e_1459_0
  have e_1459_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_1
  simp only [← c89535, ← c89536, k2, ← c89537] at e_1459_1
  have hr := e_1459_1
  simp only [e_1459_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13916)) := by
  have c89538 := (publicBatchWrapper89_copies2798 a h).2.2.1
  have c89539 := (publicBatchWrapper89_copies2798 a h).2.2.2.1
  have c89540 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.1
  have c89541 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.1
  have c89542 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.1
  have c89543 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_2
  simp only [← c89538, ← c89539, ← c89540] at e_1459_2
  have e_1459_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_3
  simp only [← c89541, ← c89542, k2, ← c89543] at e_1459_3
  have hr := e_1459_3
  simp only [e_1459_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13917)) := by
  have c89544 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.1
  have c89545 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.1
  have c89546 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89547 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89548 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89549 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_4
  simp only [← c89544, ← c89545, ← c89546] at e_1459_4
  have e_1459_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_5
  simp only [← c89547, ← c89548, k2, ← c89549] at e_1459_5
  have hr := e_1459_5
  simp only [e_1459_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13918)) := by
  have c89550 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89551 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89552 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89553 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89554 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89555 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_6
  simp only [← c89550, ← c89551, ← c89552] at e_1459_6
  have e_1459_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_7
  simp only [← c89553, ← c89554, k2, ← c89555] at e_1459_7
  have hr := e_1459_7
  simp only [e_1459_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13919)) := by
  have c89556 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89557 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89558 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89559 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89560 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89561 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_8
  simp only [← c89556, ← c89557, ← c89558] at e_1459_8
  have e_1459_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_9
  simp only [← c89559, ← c89560, k2, ← c89561] at e_1459_9
  have hr := e_1459_9
  simp only [e_1459_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13920)) := by
  have c89562 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89563 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89564 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89565 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89566 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89567 := (publicBatchWrapper89_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_10
  simp only [← c89562, ← c89563, ← c89564] at e_1459_10
  have e_1459_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_11
  simp only [← c89565, ← c89566, k2, ← c89567] at e_1459_11
  have hr := e_1459_11
  simp only [e_1459_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13921)) := by
  have c89568 := (publicBatchWrapper89_copies2799 a h).1
  have c89569 := (publicBatchWrapper89_copies2799 a h).2.1
  have c89570 := (publicBatchWrapper89_copies2799 a h).2.2.1
  have c89571 := (publicBatchWrapper89_copies2799 a h).2.2.2.1
  have c89572 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.1
  have c89573 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_12
  simp only [← c89568, ← c89569, ← c89570] at e_1459_12
  have e_1459_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_13
  simp only [← c89571, ← c89572, k2, ← c89573] at e_1459_13
  have hr := e_1459_13
  simp only [e_1459_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13922)) := by
  have c89574 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.1
  have c89575 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.1
  have c89576 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.1
  have c89577 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.1
  have c89578 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89579 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_14
  simp only [← c89574, ← c89575, ← c89576] at e_1459_14
  have e_1459_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_15
  simp only [← c89577, ← c89578, k2, ← c89579] at e_1459_15
  have hr := e_1459_15
  simp only [e_1459_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13923)) := by
  have c89580 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89581 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89582 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89583 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89584 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89585 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_16
  simp only [← c89580, ← c89581, ← c89582] at e_1459_16
  have e_1459_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_17
  simp only [← c89583, ← c89584, k2, ← c89585] at e_1459_17
  have hr := e_1459_17
  simp only [e_1459_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1459 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13924)) := by
  have c89586 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89587 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89588 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89589 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89590 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89591 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1459_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_18
  simp only [← c89586, ← c89587, ← c89588] at e_1459_18
  have e_1459_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1459 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_19
  simp only [← c89589, ← c89590, k2, ← c89591] at e_1459_19
  have hr := e_1459_19
  simp only [e_1459_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13925)) := by
  have c89592 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89593 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89594 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89595 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89596 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89597 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_0
  simp only [← c89592, ← c89593, ← c89594] at e_1460_0
  have e_1460_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_1
  simp only [← c89595, ← c89596, k2, ← c89597] at e_1460_1
  have hr := e_1460_1
  simp only [e_1460_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13926)) := by
  have c89598 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89599 := (publicBatchWrapper89_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89600 := (publicBatchWrapper89_copies2800 a h).1
  have c89601 := (publicBatchWrapper89_copies2800 a h).2.1
  have c89602 := (publicBatchWrapper89_copies2800 a h).2.2.1
  have c89603 := (publicBatchWrapper89_copies2800 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_2
  simp only [← c89598, ← c89599, ← c89600] at e_1460_2
  have e_1460_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_3
  simp only [← c89601, ← c89602, k2, ← c89603] at e_1460_3
  have hr := e_1460_3
  simp only [e_1460_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13927)) := by
  have c89604 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.1
  have c89605 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.1
  have c89606 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.1
  have c89607 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.1
  have c89608 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.1
  have c89609 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_4
  simp only [← c89604, ← c89605, ← c89606] at e_1460_4
  have e_1460_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_5
  simp only [← c89607, ← c89608, k2, ← c89609] at e_1460_5
  have hr := e_1460_5
  simp only [e_1460_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13928)) := by
  have c89610 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89611 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89612 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89613 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89614 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89615 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_6
  simp only [← c89610, ← c89611, ← c89612] at e_1460_6
  have e_1460_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_7
  simp only [← c89613, ← c89614, k2, ← c89615] at e_1460_7
  have hr := e_1460_7
  simp only [e_1460_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13929)) := by
  have c89616 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89617 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89618 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89619 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89620 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89621 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_8
  simp only [← c89616, ← c89617, ← c89618] at e_1460_8
  have e_1460_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_9
  simp only [← c89619, ← c89620, k2, ← c89621] at e_1460_9
  have hr := e_1460_9
  simp only [e_1460_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13930)) := by
  have c89622 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89623 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89624 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89625 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89626 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89627 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_10
  simp only [← c89622, ← c89623, ← c89624] at e_1460_10
  have e_1460_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_11
  simp only [← c89625, ← c89626, k2, ← c89627] at e_1460_11
  have hr := e_1460_11
  simp only [e_1460_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13931)) := by
  have c89628 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89629 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89630 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89631 := (publicBatchWrapper89_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89632 := (publicBatchWrapper89_copies2801 a h).1
  have c89633 := (publicBatchWrapper89_copies2801 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_12
  simp only [← c89628, ← c89629, ← c89630] at e_1460_12
  have e_1460_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_13
  simp only [← c89631, ← c89632, k2, ← c89633] at e_1460_13
  have hr := e_1460_13
  simp only [e_1460_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13932)) := by
  have c89634 := (publicBatchWrapper89_copies2801 a h).2.2.1
  have c89635 := (publicBatchWrapper89_copies2801 a h).2.2.2.1
  have c89636 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.1
  have c89637 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.1
  have c89638 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.1
  have c89639 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_14
  simp only [← c89634, ← c89635, ← c89636] at e_1460_14
  have e_1460_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_15
  simp only [← c89637, ← c89638, k2, ← c89639] at e_1460_15
  have hr := e_1460_15
  simp only [e_1460_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13933)) := by
  have c89640 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.1
  have c89641 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.1
  have c89642 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89643 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89644 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89645 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_16
  simp only [← c89640, ← c89641, ← c89642] at e_1460_16
  have e_1460_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_17
  simp only [← c89643, ← c89644, k2, ← c89645] at e_1460_17
  have hr := e_1460_17
  simp only [e_1460_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1460 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13934)) := by
  have c89646 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89647 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89648 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89649 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89650 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89651 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1460_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_18
  simp only [← c89646, ← c89647, ← c89648] at e_1460_18
  have e_1460_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1460 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_19
  simp only [← c89649, ← c89650, k2, ← c89651] at e_1460_19
  have hr := e_1460_19
  simp only [e_1460_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13935)) := by
  have c89652 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89653 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89654 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89655 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89656 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89657 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_0
  simp only [← c89652, ← c89653, ← c89654] at e_1461_0
  have e_1461_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_1
  simp only [← c89655, ← c89656, k2, ← c89657] at e_1461_1
  have hr := e_1461_1
  simp only [e_1461_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13936)) := by
  have c89658 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89659 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89660 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89661 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89662 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89663 := (publicBatchWrapper89_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_2
  simp only [← c89658, ← c89659, ← c89660] at e_1461_2
  have e_1461_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_3
  simp only [← c89661, ← c89662, k2, ← c89663] at e_1461_3
  have hr := e_1461_3
  simp only [e_1461_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13937)) := by
  have c89664 := (publicBatchWrapper89_copies2802 a h).1
  have c89665 := (publicBatchWrapper89_copies2802 a h).2.1
  have c89666 := (publicBatchWrapper89_copies2802 a h).2.2.1
  have c89667 := (publicBatchWrapper89_copies2802 a h).2.2.2.1
  have c89668 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.1
  have c89669 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_4
  simp only [← c89664, ← c89665, ← c89666] at e_1461_4
  have e_1461_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_5
  simp only [← c89667, ← c89668, k2, ← c89669] at e_1461_5
  have hr := e_1461_5
  simp only [e_1461_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13938)) := by
  have c89670 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.1
  have c89671 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.1
  have c89672 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.1
  have c89673 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.1
  have c89674 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89675 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_6
  simp only [← c89670, ← c89671, ← c89672] at e_1461_6
  have e_1461_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_7
  simp only [← c89673, ← c89674, k2, ← c89675] at e_1461_7
  have hr := e_1461_7
  simp only [e_1461_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13939)) := by
  have c89676 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89677 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89678 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89679 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89680 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89681 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_8
  simp only [← c89676, ← c89677, ← c89678] at e_1461_8
  have e_1461_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_9
  simp only [← c89679, ← c89680, k2, ← c89681] at e_1461_9
  have hr := e_1461_9
  simp only [e_1461_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13940)) := by
  have c89682 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89683 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89684 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89685 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89686 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89687 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_10
  simp only [← c89682, ← c89683, ← c89684] at e_1461_10
  have e_1461_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_11
  simp only [← c89685, ← c89686, k2, ← c89687] at e_1461_11
  have hr := e_1461_11
  simp only [e_1461_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13941)) := by
  have c89688 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89689 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89690 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89691 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89692 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89693 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_12
  simp only [← c89688, ← c89689, ← c89690] at e_1461_12
  have e_1461_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_13
  simp only [← c89691, ← c89692, k2, ← c89693] at e_1461_13
  have hr := e_1461_13
  simp only [e_1461_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13942)) := by
  have c89694 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89695 := (publicBatchWrapper89_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89696 := (publicBatchWrapper89_copies2803 a h).1
  have c89697 := (publicBatchWrapper89_copies2803 a h).2.1
  have c89698 := (publicBatchWrapper89_copies2803 a h).2.2.1
  have c89699 := (publicBatchWrapper89_copies2803 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_14
  simp only [← c89694, ← c89695, ← c89696] at e_1461_14
  have e_1461_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_15
  simp only [← c89697, ← c89698, k2, ← c89699] at e_1461_15
  have hr := e_1461_15
  simp only [e_1461_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13943)) := by
  have c89700 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.1
  have c89701 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.1
  have c89702 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.1
  have c89703 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.1
  have c89704 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.1
  have c89705 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_16
  simp only [← c89700, ← c89701, ← c89702] at e_1461_16
  have e_1461_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_17
  simp only [← c89703, ← c89704, k2, ← c89705] at e_1461_17
  have hr := e_1461_17
  simp only [e_1461_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1461 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13944)) := by
  have c89706 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89707 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89708 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89709 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89710 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89711 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1461_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_18
  simp only [← c89706, ← c89707, ← c89708] at e_1461_18
  have e_1461_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1461 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_19
  simp only [← c89709, ← c89710, k2, ← c89711] at e_1461_19
  have hr := e_1461_19
  simp only [e_1461_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13945)) := by
  have c89712 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89713 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89714 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89715 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89716 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89717 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_0
  simp only [← c89712, ← c89713, ← c89714] at e_1462_0
  have e_1462_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_1
  simp only [← c89715, ← c89716, k2, ← c89717] at e_1462_1
  have hr := e_1462_1
  simp only [e_1462_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13946)) := by
  have c89718 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89719 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89720 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89721 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89722 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89723 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_2
  simp only [← c89718, ← c89719, ← c89720] at e_1462_2
  have e_1462_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_3
  simp only [← c89721, ← c89722, k2, ← c89723] at e_1462_3
  have hr := e_1462_3
  simp only [e_1462_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13947)) := by
  have c89724 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89725 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89726 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89727 := (publicBatchWrapper89_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89728 := (publicBatchWrapper89_copies2804 a h).1
  have c89729 := (publicBatchWrapper89_copies2804 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_4
  simp only [← c89724, ← c89725, ← c89726] at e_1462_4
  have e_1462_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_5
  simp only [← c89727, ← c89728, k2, ← c89729] at e_1462_5
  have hr := e_1462_5
  simp only [e_1462_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13948)) := by
  have c89730 := (publicBatchWrapper89_copies2804 a h).2.2.1
  have c89731 := (publicBatchWrapper89_copies2804 a h).2.2.2.1
  have c89732 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.1
  have c89733 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.1
  have c89734 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.1
  have c89735 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_6
  simp only [← c89730, ← c89731, ← c89732] at e_1462_6
  have e_1462_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_7
  simp only [← c89733, ← c89734, k2, ← c89735] at e_1462_7
  have hr := e_1462_7
  simp only [e_1462_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13949)) := by
  have c89736 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.1
  have c89737 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.1
  have c89738 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89739 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89740 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89741 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_8
  simp only [← c89736, ← c89737, ← c89738] at e_1462_8
  have e_1462_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_9
  simp only [← c89739, ← c89740, k2, ← c89741] at e_1462_9
  have hr := e_1462_9
  simp only [e_1462_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13950)) := by
  have c89742 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89743 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89744 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89745 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89746 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89747 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_10
  simp only [← c89742, ← c89743, ← c89744] at e_1462_10
  have e_1462_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_11
  simp only [← c89745, ← c89746, k2, ← c89747] at e_1462_11
  have hr := e_1462_11
  simp only [e_1462_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13951)) := by
  have c89748 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89749 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89750 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89751 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89752 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89753 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_12
  simp only [← c89748, ← c89749, ← c89750] at e_1462_12
  have e_1462_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_13
  simp only [← c89751, ← c89752, k2, ← c89753] at e_1462_13
  have hr := e_1462_13
  simp only [e_1462_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13952)) := by
  have c89754 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89755 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89756 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89757 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89758 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89759 := (publicBatchWrapper89_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_14
  simp only [← c89754, ← c89755, ← c89756] at e_1462_14
  have e_1462_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_15
  simp only [← c89757, ← c89758, k2, ← c89759] at e_1462_15
  have hr := e_1462_15
  simp only [e_1462_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13953)) := by
  have c89760 := (publicBatchWrapper89_copies2805 a h).1
  have c89761 := (publicBatchWrapper89_copies2805 a h).2.1
  have c89762 := (publicBatchWrapper89_copies2805 a h).2.2.1
  have c89763 := (publicBatchWrapper89_copies2805 a h).2.2.2.1
  have c89764 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.1
  have c89765 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_16
  simp only [← c89760, ← c89761, ← c89762] at e_1462_16
  have e_1462_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_17
  simp only [← c89763, ← c89764, k2, ← c89765] at e_1462_17
  have hr := e_1462_17
  simp only [e_1462_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1462 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13954)) := by
  have c89766 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.1
  have c89767 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.1
  have c89768 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.1
  have c89769 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.1
  have c89770 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89771 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1462_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_18
  simp only [← c89766, ← c89767, ← c89768] at e_1462_18
  have e_1462_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1462 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_19
  simp only [← c89769, ← c89770, k2, ← c89771] at e_1462_19
  have hr := e_1462_19
  simp only [e_1462_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13955)) := by
  have c89772 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89773 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89774 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89775 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89776 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89777 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_0
  simp only [← c89772, ← c89773, ← c89774] at e_1463_0
  have e_1463_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_1
  simp only [← c89775, ← c89776, k2, ← c89777] at e_1463_1
  have hr := e_1463_1
  simp only [e_1463_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13956)) := by
  have c89778 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89779 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89780 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89781 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89782 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89783 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_2
  simp only [← c89778, ← c89779, ← c89780] at e_1463_2
  have e_1463_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_3
  simp only [← c89781, ← c89782, k2, ← c89783] at e_1463_3
  have hr := e_1463_3
  simp only [e_1463_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13957)) := by
  have c89784 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89785 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89786 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89787 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89788 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89789 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_4
  simp only [← c89784, ← c89785, ← c89786] at e_1463_4
  have e_1463_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_5
  simp only [← c89787, ← c89788, k2, ← c89789] at e_1463_5
  have hr := e_1463_5
  simp only [e_1463_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13958)) := by
  have c89790 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89791 := (publicBatchWrapper89_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89792 := (publicBatchWrapper89_copies2806 a h).1
  have c89793 := (publicBatchWrapper89_copies2806 a h).2.1
  have c89794 := (publicBatchWrapper89_copies2806 a h).2.2.1
  have c89795 := (publicBatchWrapper89_copies2806 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_6
  simp only [← c89790, ← c89791, ← c89792] at e_1463_6
  have e_1463_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_7
  simp only [← c89793, ← c89794, k2, ← c89795] at e_1463_7
  have hr := e_1463_7
  simp only [e_1463_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13959)) := by
  have c89796 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.1
  have c89797 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.1
  have c89798 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.1
  have c89799 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.1
  have c89800 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.1
  have c89801 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_8
  simp only [← c89796, ← c89797, ← c89798] at e_1463_8
  have e_1463_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_9
  simp only [← c89799, ← c89800, k2, ← c89801] at e_1463_9
  have hr := e_1463_9
  simp only [e_1463_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13960)) := by
  have c89802 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89803 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89804 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89805 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89806 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89807 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_10
  simp only [← c89802, ← c89803, ← c89804] at e_1463_10
  have e_1463_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_11
  simp only [← c89805, ← c89806, k2, ← c89807] at e_1463_11
  have hr := e_1463_11
  simp only [e_1463_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13961)) := by
  have c89808 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89809 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89810 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89811 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89812 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89813 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_12
  simp only [← c89808, ← c89809, ← c89810] at e_1463_12
  have e_1463_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_13
  simp only [← c89811, ← c89812, k2, ← c89813] at e_1463_13
  have hr := e_1463_13
  simp only [e_1463_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13962)) := by
  have c89814 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89815 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89816 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89817 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89818 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89819 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_14
  simp only [← c89814, ← c89815, ← c89816] at e_1463_14
  have e_1463_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_15
  simp only [← c89817, ← c89818, k2, ← c89819] at e_1463_15
  have hr := e_1463_15
  simp only [e_1463_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13963)) := by
  have c89820 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89821 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89822 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89823 := (publicBatchWrapper89_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89824 := (publicBatchWrapper89_copies2807 a h).1
  have c89825 := (publicBatchWrapper89_copies2807 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_16
  simp only [← c89820, ← c89821, ← c89822] at e_1463_16
  have e_1463_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_17
  simp only [← c89823, ← c89824, k2, ← c89825] at e_1463_17
  have hr := e_1463_17
  simp only [e_1463_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1463 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13964)) := by
  have c89826 := (publicBatchWrapper89_copies2807 a h).2.2.1
  have c89827 := (publicBatchWrapper89_copies2807 a h).2.2.2.1
  have c89828 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.1
  have c89829 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.1
  have c89830 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.1
  have c89831 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1463_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_18
  simp only [← c89826, ← c89827, ← c89828] at e_1463_18
  have e_1463_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1463 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_19
  simp only [← c89829, ← c89830, k2, ← c89831] at e_1463_19
  have hr := e_1463_19
  simp only [e_1463_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13965)) := by
  have c89832 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.1
  have c89833 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.1
  have c89834 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89835 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89836 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89837 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_0
  simp only [← c89832, ← c89833, ← c89834] at e_1464_0
  have e_1464_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_1
  simp only [← c89835, ← c89836, k2, ← c89837] at e_1464_1
  have hr := e_1464_1
  simp only [e_1464_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13966)) := by
  have c89838 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89839 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89840 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89841 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89842 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89843 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_2
  simp only [← c89838, ← c89839, ← c89840] at e_1464_2
  have e_1464_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_3
  simp only [← c89841, ← c89842, k2, ← c89843] at e_1464_3
  have hr := e_1464_3
  simp only [e_1464_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13967)) := by
  have c89844 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89845 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89846 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89847 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89848 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89849 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_4
  simp only [← c89844, ← c89845, ← c89846] at e_1464_4
  have e_1464_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_5
  simp only [← c89847, ← c89848, k2, ← c89849] at e_1464_5
  have hr := e_1464_5
  simp only [e_1464_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13968)) := by
  have c89850 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89851 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89852 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89853 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89854 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89855 := (publicBatchWrapper89_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_6
  simp only [← c89850, ← c89851, ← c89852] at e_1464_6
  have e_1464_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_7
  simp only [← c89853, ← c89854, k2, ← c89855] at e_1464_7
  have hr := e_1464_7
  simp only [e_1464_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13969)) := by
  have c89856 := (publicBatchWrapper89_copies2808 a h).1
  have c89857 := (publicBatchWrapper89_copies2808 a h).2.1
  have c89858 := (publicBatchWrapper89_copies2808 a h).2.2.1
  have c89859 := (publicBatchWrapper89_copies2808 a h).2.2.2.1
  have c89860 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.1
  have c89861 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_8
  simp only [← c89856, ← c89857, ← c89858] at e_1464_8
  have e_1464_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_9
  simp only [← c89859, ← c89860, k2, ← c89861] at e_1464_9
  have hr := e_1464_9
  simp only [e_1464_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14010)) := by
  have c89862 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.1
  have c89863 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.1
  have c89864 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.1
  have c89865 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.1
  have c89866 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89867 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_10
  simp only [← c89862, ← c89863, ← c89864] at e_1464_10
  have e_1464_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_11
  simp only [← c89865, ← c89866, k2, ← c89867] at e_1464_11
  have hr := e_1464_11
  simp only [e_1464_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14011)) := by
  have c89868 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89869 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89870 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89871 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89872 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89873 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_12
  simp only [← c89868, ← c89869, ← c89870] at e_1464_12
  have e_1464_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_13
  simp only [← c89871, ← c89872, k2, ← c89873] at e_1464_13
  have hr := e_1464_13
  simp only [e_1464_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14012)) := by
  have c89874 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89875 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89876 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89877 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89878 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89879 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_14
  simp only [← c89874, ← c89875, ← c89876] at e_1464_14
  have e_1464_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_15
  simp only [← c89877, ← c89878, k2, ← c89879] at e_1464_15
  have hr := e_1464_15
  simp only [e_1464_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14013)) := by
  have c89880 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89881 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89882 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89883 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89884 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89885 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_16
  simp only [← c89880, ← c89881, ← c89882] at e_1464_16
  have e_1464_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_17
  simp only [← c89883, ← c89884, k2, ← c89885] at e_1464_17
  have hr := e_1464_17
  simp only [e_1464_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1464 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14014)) := by
  have c89886 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89887 := (publicBatchWrapper89_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89888 := (publicBatchWrapper89_copies2809 a h).1
  have c89889 := (publicBatchWrapper89_copies2809 a h).2.1
  have c89890 := (publicBatchWrapper89_copies2809 a h).2.2.1
  have c89891 := (publicBatchWrapper89_copies2809 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1464_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_18
  simp only [← c89886, ← c89887, ← c89888] at e_1464_18
  have e_1464_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1464 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_19
  simp only [← c89889, ← c89890, k2, ← c89891] at e_1464_19
  have hr := e_1464_19
  simp only [e_1464_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14015)) := by
  have c89892 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.1
  have c89893 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.1
  have c89894 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.1
  have c89895 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.1
  have c89896 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.1
  have c89897 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_0
  simp only [← c89892, ← c89893, ← c89894] at e_1465_0
  have e_1465_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_1
  simp only [← c89895, ← c89896, k2, ← c89897] at e_1465_1
  have hr := e_1465_1
  simp only [e_1465_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14016)) := by
  have c89898 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89899 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89900 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89901 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89902 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89903 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_2
  simp only [← c89898, ← c89899, ← c89900] at e_1465_2
  have e_1465_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_3
  simp only [← c89901, ← c89902, k2, ← c89903] at e_1465_3
  have hr := e_1465_3
  simp only [e_1465_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14017)) := by
  have c89904 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89905 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89906 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89907 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89908 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89909 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_4
  simp only [← c89904, ← c89905, ← c89906] at e_1465_4
  have e_1465_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_5
  simp only [← c89907, ← c89908, k2, ← c89909] at e_1465_5
  have hr := e_1465_5
  simp only [e_1465_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14018)) := by
  have c89910 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89911 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89912 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89913 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89914 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89915 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_6
  simp only [← c89910, ← c89911, ← c89912] at e_1465_6
  have e_1465_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_7
  simp only [← c89913, ← c89914, k2, ← c89915] at e_1465_7
  have hr := e_1465_7
  simp only [e_1465_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14019)) := by
  have c89916 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89917 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89918 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89919 := (publicBatchWrapper89_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89920 := (publicBatchWrapper89_copies2810 a h).1
  have c89921 := (publicBatchWrapper89_copies2810 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_8
  simp only [← c89916, ← c89917, ← c89918] at e_1465_8
  have e_1465_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_9
  simp only [← c89919, ← c89920, k2, ← c89921] at e_1465_9
  have hr := e_1465_9
  simp only [e_1465_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14020)) := by
  have c89922 := (publicBatchWrapper89_copies2810 a h).2.2.1
  have c89923 := (publicBatchWrapper89_copies2810 a h).2.2.2.1
  have c89924 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.1
  have c89925 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.1
  have c89926 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.1
  have c89927 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_10
  simp only [← c89922, ← c89923, ← c89924] at e_1465_10
  have e_1465_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_11
  simp only [← c89925, ← c89926, k2, ← c89927] at e_1465_11
  have hr := e_1465_11
  simp only [e_1465_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14021)) := by
  have c89928 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.1
  have c89929 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.1
  have c89930 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89931 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89932 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89933 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_12
  simp only [← c89928, ← c89929, ← c89930] at e_1465_12
  have e_1465_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_13
  simp only [← c89931, ← c89932, k2, ← c89933] at e_1465_13
  have hr := e_1465_13
  simp only [e_1465_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14022)) := by
  have c89934 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89935 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89936 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89937 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89938 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89939 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_14
  simp only [← c89934, ← c89935, ← c89936] at e_1465_14
  have e_1465_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_15
  simp only [← c89937, ← c89938, k2, ← c89939] at e_1465_15
  have hr := e_1465_15
  simp only [e_1465_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14023)) := by
  have c89940 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89941 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89942 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89943 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89944 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89945 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_16
  simp only [← c89940, ← c89941, ← c89942] at e_1465_16
  have e_1465_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_17
  simp only [← c89943, ← c89944, k2, ← c89945] at e_1465_17
  have hr := e_1465_17
  simp only [e_1465_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1465 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14024)) := by
  have c89946 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89947 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89948 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89949 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89950 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89951 := (publicBatchWrapper89_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1465_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_18
  simp only [← c89946, ← c89947, ← c89948] at e_1465_18
  have e_1465_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1465 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_19
  simp only [← c89949, ← c89950, k2, ← c89951] at e_1465_19
  have hr := e_1465_19
  simp only [e_1465_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14025)) := by
  have c89952 := (publicBatchWrapper89_copies2811 a h).1
  have c89953 := (publicBatchWrapper89_copies2811 a h).2.1
  have c89954 := (publicBatchWrapper89_copies2811 a h).2.2.1
  have c89955 := (publicBatchWrapper89_copies2811 a h).2.2.2.1
  have c89956 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.1
  have c89957 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_0
  simp only [← c89952, ← c89953, ← c89954] at e_1466_0
  have e_1466_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_1
  simp only [← c89955, ← c89956, k2, ← c89957] at e_1466_1
  have hr := e_1466_1
  simp only [e_1466_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14026)) := by
  have c89958 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.1
  have c89959 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.1
  have c89960 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.1
  have c89961 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.1
  have c89962 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89963 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_2
  simp only [← c89958, ← c89959, ← c89960] at e_1466_2
  have e_1466_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_3
  simp only [← c89961, ← c89962, k2, ← c89963] at e_1466_3
  have hr := e_1466_3
  simp only [e_1466_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14027)) := by
  have c89964 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89965 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89966 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89967 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89968 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89969 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_4
  simp only [← c89964, ← c89965, ← c89966] at e_1466_4
  have e_1466_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_5
  simp only [← c89967, ← c89968, k2, ← c89969] at e_1466_5
  have hr := e_1466_5
  simp only [e_1466_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14028)) := by
  have c89970 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89971 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89972 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89973 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89974 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89975 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_6
  simp only [← c89970, ← c89971, ← c89972] at e_1466_6
  have e_1466_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_7
  simp only [← c89973, ← c89974, k2, ← c89975] at e_1466_7
  have hr := e_1466_7
  simp only [e_1466_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14029)) := by
  have c89976 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89977 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89978 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89979 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89980 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89981 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_8
  simp only [← c89976, ← c89977, ← c89978] at e_1466_8
  have e_1466_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_9
  simp only [← c89979, ← c89980, k2, ← c89981] at e_1466_9
  have hr := e_1466_9
  simp only [e_1466_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14030)) := by
  have c89982 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89983 := (publicBatchWrapper89_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89984 := (publicBatchWrapper89_copies2812 a h).1
  have c89985 := (publicBatchWrapper89_copies2812 a h).2.1
  have c89986 := (publicBatchWrapper89_copies2812 a h).2.2.1
  have c89987 := (publicBatchWrapper89_copies2812 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_10
  simp only [← c89982, ← c89983, ← c89984] at e_1466_10
  have e_1466_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_11
  simp only [← c89985, ← c89986, k2, ← c89987] at e_1466_11
  have hr := e_1466_11
  simp only [e_1466_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14031)) := by
  have c89988 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.1
  have c89989 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.1
  have c89990 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.1
  have c89991 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.1
  have c89992 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.1
  have c89993 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_12
  simp only [← c89988, ← c89989, ← c89990] at e_1466_12
  have e_1466_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_13
  simp only [← c89991, ← c89992, k2, ← c89993] at e_1466_13
  have hr := e_1466_13
  simp only [e_1466_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14032)) := by
  have c89994 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89995 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89996 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89997 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89998 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89999 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_14
  simp only [← c89994, ← c89995, ← c89996] at e_1466_14
  have e_1466_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_15
  simp only [← c89997, ← c89998, k2, ← c89999] at e_1466_15
  have hr := e_1466_15
  simp only [e_1466_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14033)) := by
  have c90000 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90001 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90002 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90003 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90004 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90005 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_16
  simp only [← c90000, ← c90001, ← c90002] at e_1466_16
  have e_1466_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_17
  simp only [← c90003, ← c90004, k2, ← c90005] at e_1466_17
  have hr := e_1466_17
  simp only [e_1466_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1466 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14034)) := by
  have c90006 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90007 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90008 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90009 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90010 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90011 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1466_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_18
  simp only [← c90006, ← c90007, ← c90008] at e_1466_18
  have e_1466_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1466 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_19
  simp only [← c90009, ← c90010, k2, ← c90011] at e_1466_19
  have hr := e_1466_19
  simp only [e_1466_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14035)) := by
  have c90012 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90013 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90014 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90015 := (publicBatchWrapper89_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90016 := (publicBatchWrapper89_copies2813 a h).1
  have c90017 := (publicBatchWrapper89_copies2813 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_0
  simp only [← c90012, ← c90013, ← c90014] at e_1467_0
  have e_1467_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_1
  simp only [← c90015, ← c90016, k2, ← c90017] at e_1467_1
  have hr := e_1467_1
  simp only [e_1467_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14036)) := by
  have c90018 := (publicBatchWrapper89_copies2813 a h).2.2.1
  have c90019 := (publicBatchWrapper89_copies2813 a h).2.2.2.1
  have c90020 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.1
  have c90021 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.1
  have c90022 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.1
  have c90023 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_2
  simp only [← c90018, ← c90019, ← c90020] at e_1467_2
  have e_1467_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_3
  simp only [← c90021, ← c90022, k2, ← c90023] at e_1467_3
  have hr := e_1467_3
  simp only [e_1467_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14037)) := by
  have c90024 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.1
  have c90025 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.1
  have c90026 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90027 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90028 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90029 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_4
  simp only [← c90024, ← c90025, ← c90026] at e_1467_4
  have e_1467_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_5
  simp only [← c90027, ← c90028, k2, ← c90029] at e_1467_5
  have hr := e_1467_5
  simp only [e_1467_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14038)) := by
  have c90030 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90031 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90032 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90033 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90034 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90035 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_6
  simp only [← c90030, ← c90031, ← c90032] at e_1467_6
  have e_1467_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_7
  simp only [← c90033, ← c90034, k2, ← c90035] at e_1467_7
  have hr := e_1467_7
  simp only [e_1467_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14039)) := by
  have c90036 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90037 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90038 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90039 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90040 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90041 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_8
  simp only [← c90036, ← c90037, ← c90038] at e_1467_8
  have e_1467_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_9
  simp only [← c90039, ← c90040, k2, ← c90041] at e_1467_9
  have hr := e_1467_9
  simp only [e_1467_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14040)) := by
  have c90042 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90043 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90044 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90045 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90046 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90047 := (publicBatchWrapper89_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_10
  simp only [← c90042, ← c90043, ← c90044] at e_1467_10
  have e_1467_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_11
  simp only [← c90045, ← c90046, k2, ← c90047] at e_1467_11
  have hr := e_1467_11
  simp only [e_1467_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14041)) := by
  have c90048 := (publicBatchWrapper89_copies2814 a h).1
  have c90049 := (publicBatchWrapper89_copies2814 a h).2.1
  have c90050 := (publicBatchWrapper89_copies2814 a h).2.2.1
  have c90051 := (publicBatchWrapper89_copies2814 a h).2.2.2.1
  have c90052 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.1
  have c90053 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_12
  simp only [← c90048, ← c90049, ← c90050] at e_1467_12
  have e_1467_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_13
  simp only [← c90051, ← c90052, k2, ← c90053] at e_1467_13
  have hr := e_1467_13
  simp only [e_1467_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14042)) := by
  have c90054 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.1
  have c90055 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.1
  have c90056 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.1
  have c90057 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.1
  have c90058 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90059 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_14
  simp only [← c90054, ← c90055, ← c90056] at e_1467_14
  have e_1467_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_15
  simp only [← c90057, ← c90058, k2, ← c90059] at e_1467_15
  have hr := e_1467_15
  simp only [e_1467_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14043)) := by
  have c90060 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90061 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90062 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90063 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90064 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90065 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_16
  simp only [← c90060, ← c90061, ← c90062] at e_1467_16
  have e_1467_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_17
  simp only [← c90063, ← c90064, k2, ← c90065] at e_1467_17
  have hr := e_1467_17
  simp only [e_1467_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1467 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14044)) := by
  have c90066 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90067 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90068 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90069 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90070 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90071 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1467_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_18
  simp only [← c90066, ← c90067, ← c90068] at e_1467_18
  have e_1467_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1467 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_19
  simp only [← c90069, ← c90070, k2, ← c90071] at e_1467_19
  have hr := e_1467_19
  simp only [e_1467_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14045)) := by
  have c90072 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90073 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90074 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90075 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90076 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90077 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_0
  simp only [← c90072, ← c90073, ← c90074] at e_1468_0
  have e_1468_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_1
  simp only [← c90075, ← c90076, k2, ← c90077] at e_1468_1
  have hr := e_1468_1
  simp only [e_1468_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14046)) := by
  have c90078 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90079 := (publicBatchWrapper89_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90080 := (publicBatchWrapper89_copies2815 a h).1
  have c90081 := (publicBatchWrapper89_copies2815 a h).2.1
  have c90082 := (publicBatchWrapper89_copies2815 a h).2.2.1
  have c90083 := (publicBatchWrapper89_copies2815 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_2
  simp only [← c90078, ← c90079, ← c90080] at e_1468_2
  have e_1468_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_3
  simp only [← c90081, ← c90082, k2, ← c90083] at e_1468_3
  have hr := e_1468_3
  simp only [e_1468_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14047)) := by
  have c90084 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.1
  have c90085 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.1
  have c90086 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.1
  have c90087 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.1
  have c90088 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.1
  have c90089 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_4
  simp only [← c90084, ← c90085, ← c90086] at e_1468_4
  have e_1468_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_5
  simp only [← c90087, ← c90088, k2, ← c90089] at e_1468_5
  have hr := e_1468_5
  simp only [e_1468_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14048)) := by
  have c90090 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90091 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90092 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90093 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90094 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90095 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_6
  simp only [← c90090, ← c90091, ← c90092] at e_1468_6
  have e_1468_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_7
  simp only [← c90093, ← c90094, k2, ← c90095] at e_1468_7
  have hr := e_1468_7
  simp only [e_1468_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14049)) := by
  have c90096 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90097 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90098 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90099 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90100 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90101 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_8
  simp only [← c90096, ← c90097, ← c90098] at e_1468_8
  have e_1468_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_9
  simp only [← c90099, ← c90100, k2, ← c90101] at e_1468_9
  have hr := e_1468_9
  simp only [e_1468_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14050)) := by
  have c90102 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90103 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90104 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90105 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90106 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90107 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_10
  simp only [← c90102, ← c90103, ← c90104] at e_1468_10
  have e_1468_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_11
  simp only [← c90105, ← c90106, k2, ← c90107] at e_1468_11
  have hr := e_1468_11
  simp only [e_1468_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14051)) := by
  have c90108 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90109 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90110 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90111 := (publicBatchWrapper89_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90112 := (publicBatchWrapper89_copies2816 a h).1
  have c90113 := (publicBatchWrapper89_copies2816 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_12
  simp only [← c90108, ← c90109, ← c90110] at e_1468_12
  have e_1468_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_13
  simp only [← c90111, ← c90112, k2, ← c90113] at e_1468_13
  have hr := e_1468_13
  simp only [e_1468_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14052)) := by
  have c90114 := (publicBatchWrapper89_copies2816 a h).2.2.1
  have c90115 := (publicBatchWrapper89_copies2816 a h).2.2.2.1
  have c90116 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.1
  have c90117 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.1
  have c90118 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.1
  have c90119 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_14
  simp only [← c90114, ← c90115, ← c90116] at e_1468_14
  have e_1468_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_15
  simp only [← c90117, ← c90118, k2, ← c90119] at e_1468_15
  have hr := e_1468_15
  simp only [e_1468_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14053)) := by
  have c90120 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.1
  have c90121 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.1
  have c90122 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90123 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90124 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90125 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_16
  simp only [← c90120, ← c90121, ← c90122] at e_1468_16
  have e_1468_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_17
  simp only [← c90123, ← c90124, k2, ← c90125] at e_1468_17
  have hr := e_1468_17
  simp only [e_1468_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1468 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14054)) := by
  have c90126 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90127 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90128 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90129 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90130 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90131 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1468_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_18
  simp only [← c90126, ← c90127, ← c90128] at e_1468_18
  have e_1468_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1468 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_19
  simp only [← c90129, ← c90130, k2, ← c90131] at e_1468_19
  have hr := e_1468_19
  simp only [e_1468_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14055)) := by
  have c90132 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90133 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90134 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90135 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90136 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90137 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_0
  simp only [← c90132, ← c90133, ← c90134] at e_1469_0
  have e_1469_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_1
  simp only [← c90135, ← c90136, k2, ← c90137] at e_1469_1
  have hr := e_1469_1
  simp only [e_1469_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14056)) := by
  have c90138 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90139 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90140 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90141 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90142 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90143 := (publicBatchWrapper89_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_2
  simp only [← c90138, ← c90139, ← c90140] at e_1469_2
  have e_1469_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_3
  simp only [← c90141, ← c90142, k2, ← c90143] at e_1469_3
  have hr := e_1469_3
  simp only [e_1469_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14057)) := by
  have c90144 := (publicBatchWrapper89_copies2817 a h).1
  have c90145 := (publicBatchWrapper89_copies2817 a h).2.1
  have c90146 := (publicBatchWrapper89_copies2817 a h).2.2.1
  have c90147 := (publicBatchWrapper89_copies2817 a h).2.2.2.1
  have c90148 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.1
  have c90149 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_4
  simp only [← c90144, ← c90145, ← c90146] at e_1469_4
  have e_1469_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_5
  simp only [← c90147, ← c90148, k2, ← c90149] at e_1469_5
  have hr := e_1469_5
  simp only [e_1469_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14058)) := by
  have c90150 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.1
  have c90151 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.1
  have c90152 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.1
  have c90153 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.1
  have c90154 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90155 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_6
  simp only [← c90150, ← c90151, ← c90152] at e_1469_6
  have e_1469_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_7
  simp only [← c90153, ← c90154, k2, ← c90155] at e_1469_7
  have hr := e_1469_7
  simp only [e_1469_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14059)) := by
  have c90156 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90157 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90158 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90159 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90160 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90161 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_8
  simp only [← c90156, ← c90157, ← c90158] at e_1469_8
  have e_1469_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_9
  simp only [← c90159, ← c90160, k2, ← c90161] at e_1469_9
  have hr := e_1469_9
  simp only [e_1469_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14060)) := by
  have c90162 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90163 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90164 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90165 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90166 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90167 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_10
  simp only [← c90162, ← c90163, ← c90164] at e_1469_10
  have e_1469_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_11
  simp only [← c90165, ← c90166, k2, ← c90167] at e_1469_11
  have hr := e_1469_11
  simp only [e_1469_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14061)) := by
  have c90168 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90169 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90170 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90171 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90172 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90173 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_12
  simp only [← c90168, ← c90169, ← c90170] at e_1469_12
  have e_1469_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_13
  simp only [← c90171, ← c90172, k2, ← c90173] at e_1469_13
  have hr := e_1469_13
  simp only [e_1469_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14062)) := by
  have c90174 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90175 := (publicBatchWrapper89_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90176 := (publicBatchWrapper89_copies2818 a h).1
  have c90177 := (publicBatchWrapper89_copies2818 a h).2.1
  have c90178 := (publicBatchWrapper89_copies2818 a h).2.2.1
  have c90179 := (publicBatchWrapper89_copies2818 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_14
  simp only [← c90174, ← c90175, ← c90176] at e_1469_14
  have e_1469_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_15
  simp only [← c90177, ← c90178, k2, ← c90179] at e_1469_15
  have hr := e_1469_15
  simp only [e_1469_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14063)) := by
  have c90180 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.1
  have c90181 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.1
  have c90182 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.1
  have c90183 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.1
  have c90184 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.1
  have c90185 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_16
  simp only [← c90180, ← c90181, ← c90182] at e_1469_16
  have e_1469_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_17
  simp only [← c90183, ← c90184, k2, ← c90185] at e_1469_17
  have hr := e_1469_17
  simp only [e_1469_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1469 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14064)) := by
  have c90186 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90187 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90188 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90189 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90190 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90191 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1469_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_18
  simp only [← c90186, ← c90187, ← c90188] at e_1469_18
  have e_1469_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1469 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_19
  simp only [← c90189, ← c90190, k2, ← c90191] at e_1469_19
  have hr := e_1469_19
  simp only [e_1469_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14065)) := by
  have c90192 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90193 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90194 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90195 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90196 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90197 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_0
  simp only [← c90192, ← c90193, ← c90194] at e_1470_0
  have e_1470_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_1
  simp only [← c90195, ← c90196, k2, ← c90197] at e_1470_1
  have hr := e_1470_1
  simp only [e_1470_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14066)) := by
  have c90198 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90199 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90200 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90201 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90202 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90203 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_2
  simp only [← c90198, ← c90199, ← c90200] at e_1470_2
  have e_1470_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_3
  simp only [← c90201, ← c90202, k2, ← c90203] at e_1470_3
  have hr := e_1470_3
  simp only [e_1470_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14067)) := by
  have c90204 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90205 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90206 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90207 := (publicBatchWrapper89_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90208 := (publicBatchWrapper89_copies2819 a h).1
  have c90209 := (publicBatchWrapper89_copies2819 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_4
  simp only [← c90204, ← c90205, ← c90206] at e_1470_4
  have e_1470_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_5
  simp only [← c90207, ← c90208, k2, ← c90209] at e_1470_5
  have hr := e_1470_5
  simp only [e_1470_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14068)) := by
  have c90210 := (publicBatchWrapper89_copies2819 a h).2.2.1
  have c90211 := (publicBatchWrapper89_copies2819 a h).2.2.2.1
  have c90212 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.1
  have c90213 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.1
  have c90214 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.1
  have c90215 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_6
  simp only [← c90210, ← c90211, ← c90212] at e_1470_6
  have e_1470_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_7
  simp only [← c90213, ← c90214, k2, ← c90215] at e_1470_7
  have hr := e_1470_7
  simp only [e_1470_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14069)) := by
  have c90216 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.1
  have c90217 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.1
  have c90218 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90219 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90220 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90221 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_8
  simp only [← c90216, ← c90217, ← c90218] at e_1470_8
  have e_1470_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_9
  simp only [← c90219, ← c90220, k2, ← c90221] at e_1470_9
  have hr := e_1470_9
  simp only [e_1470_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14070)) := by
  have c90222 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90223 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90224 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90225 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90226 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90227 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_10
  simp only [← c90222, ← c90223, ← c90224] at e_1470_10
  have e_1470_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_11
  simp only [← c90225, ← c90226, k2, ← c90227] at e_1470_11
  have hr := e_1470_11
  simp only [e_1470_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14071)) := by
  have c90228 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90229 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90230 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90231 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90232 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90233 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_12
  simp only [← c90228, ← c90229, ← c90230] at e_1470_12
  have e_1470_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_13
  simp only [← c90231, ← c90232, k2, ← c90233] at e_1470_13
  have hr := e_1470_13
  simp only [e_1470_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14072)) := by
  have c90234 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90235 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90236 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90237 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90238 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90239 := (publicBatchWrapper89_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_14
  simp only [← c90234, ← c90235, ← c90236] at e_1470_14
  have e_1470_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_15
  simp only [← c90237, ← c90238, k2, ← c90239] at e_1470_15
  have hr := e_1470_15
  simp only [e_1470_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14073)) := by
  have c90240 := (publicBatchWrapper89_copies2820 a h).1
  have c90241 := (publicBatchWrapper89_copies2820 a h).2.1
  have c90242 := (publicBatchWrapper89_copies2820 a h).2.2.1
  have c90243 := (publicBatchWrapper89_copies2820 a h).2.2.2.1
  have c90244 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.1
  have c90245 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_16
  simp only [← c90240, ← c90241, ← c90242] at e_1470_16
  have e_1470_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_17
  simp only [← c90243, ← c90244, k2, ← c90245] at e_1470_17
  have hr := e_1470_17
  simp only [e_1470_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1470 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14074)) := by
  have c90246 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.1
  have c90247 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.1
  have c90248 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.1
  have c90249 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.1
  have c90250 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90251 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1470_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_18
  simp only [← c90246, ← c90247, ← c90248] at e_1470_18
  have e_1470_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1470 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_19
  simp only [← c90249, ← c90250, k2, ← c90251] at e_1470_19
  have hr := e_1470_19
  simp only [e_1470_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14075)) := by
  have c90252 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90253 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90254 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90255 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90256 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90257 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_0
  simp only [← c90252, ← c90253, ← c90254] at e_1471_0
  have e_1471_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_1
  simp only [← c90255, ← c90256, k2, ← c90257] at e_1471_1
  have hr := e_1471_1
  simp only [e_1471_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14076)) := by
  have c90258 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90259 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90260 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90261 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90262 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90263 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_2
  simp only [← c90258, ← c90259, ← c90260] at e_1471_2
  have e_1471_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_3
  simp only [← c90261, ← c90262, k2, ← c90263] at e_1471_3
  have hr := e_1471_3
  simp only [e_1471_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14077)) := by
  have c90264 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90265 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90266 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90267 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90268 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90269 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_4
  simp only [← c90264, ← c90265, ← c90266] at e_1471_4
  have e_1471_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_5
  simp only [← c90267, ← c90268, k2, ← c90269] at e_1471_5
  have hr := e_1471_5
  simp only [e_1471_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14078)) := by
  have c90270 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90271 := (publicBatchWrapper89_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90272 := (publicBatchWrapper89_copies2821 a h).1
  have c90273 := (publicBatchWrapper89_copies2821 a h).2.1
  have c90274 := (publicBatchWrapper89_copies2821 a h).2.2.1
  have c90275 := (publicBatchWrapper89_copies2821 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_6
  simp only [← c90270, ← c90271, ← c90272] at e_1471_6
  have e_1471_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_7
  simp only [← c90273, ← c90274, k2, ← c90275] at e_1471_7
  have hr := e_1471_7
  simp only [e_1471_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14079)) := by
  have c90276 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.1
  have c90277 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.1
  have c90278 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.1
  have c90279 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.1
  have c90280 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.1
  have c90281 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_8
  simp only [← c90276, ← c90277, ← c90278] at e_1471_8
  have e_1471_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_9
  simp only [← c90279, ← c90280, k2, ← c90281] at e_1471_9
  have hr := e_1471_9
  simp only [e_1471_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14080)) := by
  have c90282 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90283 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90284 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90285 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90286 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90287 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_10
  simp only [← c90282, ← c90283, ← c90284] at e_1471_10
  have e_1471_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_11
  simp only [← c90285, ← c90286, k2, ← c90287] at e_1471_11
  have hr := e_1471_11
  simp only [e_1471_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14081)) := by
  have c90288 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90289 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90290 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90291 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90292 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90293 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_12
  simp only [← c90288, ← c90289, ← c90290] at e_1471_12
  have e_1471_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_13
  simp only [← c90291, ← c90292, k2, ← c90293] at e_1471_13
  have hr := e_1471_13
  simp only [e_1471_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14082)) := by
  have c90294 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90295 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90296 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90297 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90298 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90299 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_14
  simp only [← c90294, ← c90295, ← c90296] at e_1471_14
  have e_1471_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_15
  simp only [← c90297, ← c90298, k2, ← c90299] at e_1471_15
  have hr := e_1471_15
  simp only [e_1471_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14083)) := by
  have c90300 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90301 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90302 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90303 := (publicBatchWrapper89_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90304 := (publicBatchWrapper89_copies2822 a h).1
  have c90305 := (publicBatchWrapper89_copies2822 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_16
  simp only [← c90300, ← c90301, ← c90302] at e_1471_16
  have e_1471_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_17
  simp only [← c90303, ← c90304, k2, ← c90305] at e_1471_17
  have hr := e_1471_17
  simp only [e_1471_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1471 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14084)) := by
  have c90306 := (publicBatchWrapper89_copies2822 a h).2.2.1
  have c90307 := (publicBatchWrapper89_copies2822 a h).2.2.2.1
  have c90308 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.1
  have c90309 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.1
  have c90310 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.1
  have c90311 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1471_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_18
  simp only [← c90306, ← c90307, ← c90308] at e_1471_18
  have e_1471_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1471 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_19
  simp only [← c90309, ← c90310, k2, ← c90311] at e_1471_19
  have hr := e_1471_19
  simp only [e_1471_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14085)) := by
  have c90312 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.1
  have c90313 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.1
  have c90314 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90315 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90316 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90317 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_0
  simp only [← c90312, ← c90313, ← c90314] at e_1472_0
  have e_1472_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_1
  simp only [← c90315, ← c90316, k2, ← c90317] at e_1472_1
  have hr := e_1472_1
  simp only [e_1472_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14086)) := by
  have c90318 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90319 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90320 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90321 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90322 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90323 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_2
  simp only [← c90318, ← c90319, ← c90320] at e_1472_2
  have e_1472_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_3
  simp only [← c90321, ← c90322, k2, ← c90323] at e_1472_3
  have hr := e_1472_3
  simp only [e_1472_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14087)) := by
  have c90324 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90325 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90326 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90327 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90328 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90329 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_4
  simp only [← c90324, ← c90325, ← c90326] at e_1472_4
  have e_1472_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_5
  simp only [← c90327, ← c90328, k2, ← c90329] at e_1472_5
  have hr := e_1472_5
  simp only [e_1472_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14088)) := by
  have c90330 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90331 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90332 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90333 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90334 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90335 := (publicBatchWrapper89_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_6
  simp only [← c90330, ← c90331, ← c90332] at e_1472_6
  have e_1472_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_7
  simp only [← c90333, ← c90334, k2, ← c90335] at e_1472_7
  have hr := e_1472_7
  simp only [e_1472_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14089)) := by
  have c90336 := (publicBatchWrapper89_copies2823 a h).1
  have c90337 := (publicBatchWrapper89_copies2823 a h).2.1
  have c90338 := (publicBatchWrapper89_copies2823 a h).2.2.1
  have c90339 := (publicBatchWrapper89_copies2823 a h).2.2.2.1
  have c90340 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.1
  have c90341 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_8
  simp only [← c90336, ← c90337, ← c90338] at e_1472_8
  have e_1472_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_9
  simp only [← c90339, ← c90340, k2, ← c90341] at e_1472_9
  have hr := e_1472_9
  simp only [e_1472_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14090)) := by
  have c90342 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.1
  have c90343 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.1
  have c90344 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.1
  have c90345 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.1
  have c90346 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90347 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_10
  simp only [← c90342, ← c90343, ← c90344] at e_1472_10
  have e_1472_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_11
  simp only [← c90345, ← c90346, k2, ← c90347] at e_1472_11
  have hr := e_1472_11
  simp only [e_1472_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14091)) := by
  have c90348 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90349 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90350 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90351 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90352 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90353 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_12
  simp only [← c90348, ← c90349, ← c90350] at e_1472_12
  have e_1472_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_13
  simp only [← c90351, ← c90352, k2, ← c90353] at e_1472_13
  have hr := e_1472_13
  simp only [e_1472_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14092)) := by
  have c90354 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90355 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90356 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90357 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90358 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90359 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_14
  simp only [← c90354, ← c90355, ← c90356] at e_1472_14
  have e_1472_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_15
  simp only [← c90357, ← c90358, k2, ← c90359] at e_1472_15
  have hr := e_1472_15
  simp only [e_1472_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14093)) := by
  have c90360 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90361 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90362 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90363 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90364 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90365 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_16
  simp only [← c90360, ← c90361, ← c90362] at e_1472_16
  have e_1472_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_17
  simp only [← c90363, ← c90364, k2, ← c90365] at e_1472_17
  have hr := e_1472_17
  simp only [e_1472_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1472 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14094)) := by
  have c90366 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90367 := (publicBatchWrapper89_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90368 := (publicBatchWrapper89_copies2824 a h).1
  have c90369 := (publicBatchWrapper89_copies2824 a h).2.1
  have c90370 := (publicBatchWrapper89_copies2824 a h).2.2.1
  have c90371 := (publicBatchWrapper89_copies2824 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1472_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_18
  simp only [← c90366, ← c90367, ← c90368] at e_1472_18
  have e_1472_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1472 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_19
  simp only [← c90369, ← c90370, k2, ← c90371] at e_1472_19
  have hr := e_1472_19
  simp only [e_1472_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14095)) := by
  have c90372 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.1
  have c90373 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.1
  have c90374 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.1
  have c90375 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.1
  have c90376 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.1
  have c90377 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_0
  simp only [← c90372, ← c90373, ← c90374] at e_1473_0
  have e_1473_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_1
  simp only [← c90375, ← c90376, k2, ← c90377] at e_1473_1
  have hr := e_1473_1
  simp only [e_1473_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14096)) := by
  have c90378 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90379 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90380 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90381 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90382 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90383 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_2
  simp only [← c90378, ← c90379, ← c90380] at e_1473_2
  have e_1473_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_3
  simp only [← c90381, ← c90382, k2, ← c90383] at e_1473_3
  have hr := e_1473_3
  simp only [e_1473_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14097)) := by
  have c90384 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90385 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90386 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90387 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90388 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90389 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_4
  simp only [← c90384, ← c90385, ← c90386] at e_1473_4
  have e_1473_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_5
  simp only [← c90387, ← c90388, k2, ← c90389] at e_1473_5
  have hr := e_1473_5
  simp only [e_1473_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14098)) := by
  have c90390 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90391 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90392 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90393 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90394 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90395 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_6
  simp only [← c90390, ← c90391, ← c90392] at e_1473_6
  have e_1473_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_7
  simp only [← c90393, ← c90394, k2, ← c90395] at e_1473_7
  have hr := e_1473_7
  simp only [e_1473_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14099)) := by
  have c90396 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90397 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90398 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90399 := (publicBatchWrapper89_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90400 := (publicBatchWrapper89_copies2825 a h).1
  have c90401 := (publicBatchWrapper89_copies2825 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_8
  simp only [← c90396, ← c90397, ← c90398] at e_1473_8
  have e_1473_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_9
  simp only [← c90399, ← c90400, k2, ← c90401] at e_1473_9
  have hr := e_1473_9
  simp only [e_1473_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14100)) := by
  have c90402 := (publicBatchWrapper89_copies2825 a h).2.2.1
  have c90403 := (publicBatchWrapper89_copies2825 a h).2.2.2.1
  have c90404 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.1
  have c90405 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.1
  have c90406 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.1
  have c90407 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_10
  simp only [← c90402, ← c90403, ← c90404] at e_1473_10
  have e_1473_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_11
  simp only [← c90405, ← c90406, k2, ← c90407] at e_1473_11
  have hr := e_1473_11
  simp only [e_1473_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14101)) := by
  have c90408 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.1
  have c90409 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.1
  have c90410 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90411 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90412 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90413 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_12
  simp only [← c90408, ← c90409, ← c90410] at e_1473_12
  have e_1473_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_13
  simp only [← c90411, ← c90412, k2, ← c90413] at e_1473_13
  have hr := e_1473_13
  simp only [e_1473_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14102)) := by
  have c90414 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90415 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90416 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90417 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90418 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90419 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_14
  simp only [← c90414, ← c90415, ← c90416] at e_1473_14
  have e_1473_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_15
  simp only [← c90417, ← c90418, k2, ← c90419] at e_1473_15
  have hr := e_1473_15
  simp only [e_1473_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14103)) := by
  have c90420 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90421 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90422 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90423 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90424 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90425 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_16
  simp only [← c90420, ← c90421, ← c90422] at e_1473_16
  have e_1473_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_17
  simp only [← c90423, ← c90424, k2, ← c90425] at e_1473_17
  have hr := e_1473_17
  simp only [e_1473_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1473 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14104)) := by
  have c90426 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90427 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90428 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90429 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90430 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90431 := (publicBatchWrapper89_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1473_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_18
  simp only [← c90426, ← c90427, ← c90428] at e_1473_18
  have e_1473_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1473 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_19
  simp only [← c90429, ← c90430, k2, ← c90431] at e_1473_19
  have hr := e_1473_19
  simp only [e_1473_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14105)) := by
  have c90432 := (publicBatchWrapper89_copies2826 a h).1
  have c90433 := (publicBatchWrapper89_copies2826 a h).2.1
  have c90434 := (publicBatchWrapper89_copies2826 a h).2.2.1
  have c90435 := (publicBatchWrapper89_copies2826 a h).2.2.2.1
  have c90436 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.1
  have c90437 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_0
  simp only [← c90432, ← c90433, ← c90434] at e_1474_0
  have e_1474_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_1
  simp only [← c90435, ← c90436, k2, ← c90437] at e_1474_1
  have hr := e_1474_1
  simp only [e_1474_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14106)) := by
  have c90438 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.1
  have c90439 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.1
  have c90440 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.1
  have c90441 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.1
  have c90442 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90443 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_2
  simp only [← c90438, ← c90439, ← c90440] at e_1474_2
  have e_1474_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_3
  simp only [← c90441, ← c90442, k2, ← c90443] at e_1474_3
  have hr := e_1474_3
  simp only [e_1474_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14107)) := by
  have c90444 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90445 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90446 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90447 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90448 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90449 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_4
  simp only [← c90444, ← c90445, ← c90446] at e_1474_4
  have e_1474_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_5
  simp only [← c90447, ← c90448, k2, ← c90449] at e_1474_5
  have hr := e_1474_5
  simp only [e_1474_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14108)) := by
  have c90450 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90451 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90452 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90453 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90454 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90455 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_6
  simp only [← c90450, ← c90451, ← c90452] at e_1474_6
  have e_1474_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_7
  simp only [← c90453, ← c90454, k2, ← c90455] at e_1474_7
  have hr := e_1474_7
  simp only [e_1474_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14109)) := by
  have c90456 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90457 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90458 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90459 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90460 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90461 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_8
  simp only [← c90456, ← c90457, ← c90458] at e_1474_8
  have e_1474_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_9
  simp only [← c90459, ← c90460, k2, ← c90461] at e_1474_9
  have hr := e_1474_9
  simp only [e_1474_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14110)) := by
  have c90462 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90463 := (publicBatchWrapper89_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90464 := (publicBatchWrapper89_copies2827 a h).1
  have c90465 := (publicBatchWrapper89_copies2827 a h).2.1
  have c90466 := (publicBatchWrapper89_copies2827 a h).2.2.1
  have c90467 := (publicBatchWrapper89_copies2827 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_10
  simp only [← c90462, ← c90463, ← c90464] at e_1474_10
  have e_1474_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_11
  simp only [← c90465, ← c90466, k2, ← c90467] at e_1474_11
  have hr := e_1474_11
  simp only [e_1474_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14111)) := by
  have c90468 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.1
  have c90469 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.1
  have c90470 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.1
  have c90471 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.1
  have c90472 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.1
  have c90473 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_12
  simp only [← c90468, ← c90469, ← c90470] at e_1474_12
  have e_1474_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_13
  simp only [← c90471, ← c90472, k2, ← c90473] at e_1474_13
  have hr := e_1474_13
  simp only [e_1474_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14112)) := by
  have c90474 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90475 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90476 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90477 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90478 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90479 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_14
  simp only [← c90474, ← c90475, ← c90476] at e_1474_14
  have e_1474_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_15
  simp only [← c90477, ← c90478, k2, ← c90479] at e_1474_15
  have hr := e_1474_15
  simp only [e_1474_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14113)) := by
  have c90480 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90481 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90482 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90483 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90484 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90485 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_16
  simp only [← c90480, ← c90481, ← c90482] at e_1474_16
  have e_1474_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_17
  simp only [← c90483, ← c90484, k2, ← c90485] at e_1474_17
  have hr := e_1474_17
  simp only [e_1474_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1474 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14114)) := by
  have c90486 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90487 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90488 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90489 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90490 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90491 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1474_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_18
  simp only [← c90486, ← c90487, ← c90488] at e_1474_18
  have e_1474_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1474 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_19
  simp only [← c90489, ← c90490, k2, ← c90491] at e_1474_19
  have hr := e_1474_19
  simp only [e_1474_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14115)) := by
  have c90492 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90493 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90494 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90495 := (publicBatchWrapper89_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90496 := (publicBatchWrapper89_copies2828 a h).1
  have c90497 := (publicBatchWrapper89_copies2828 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_0
  simp only [← c90492, ← c90493, ← c90494] at e_1475_0
  have e_1475_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_1
  simp only [← c90495, ← c90496, k2, ← c90497] at e_1475_1
  have hr := e_1475_1
  simp only [e_1475_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14116)) := by
  have c90498 := (publicBatchWrapper89_copies2828 a h).2.2.1
  have c90499 := (publicBatchWrapper89_copies2828 a h).2.2.2.1
  have c90500 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.1
  have c90501 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.1
  have c90502 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.1
  have c90503 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_2
  simp only [← c90498, ← c90499, ← c90500] at e_1475_2
  have e_1475_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_3
  simp only [← c90501, ← c90502, k2, ← c90503] at e_1475_3
  have hr := e_1475_3
  simp only [e_1475_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14117)) := by
  have c90504 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.1
  have c90505 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.1
  have c90506 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90507 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90508 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90509 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_4
  simp only [← c90504, ← c90505, ← c90506] at e_1475_4
  have e_1475_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_5
  simp only [← c90507, ← c90508, k2, ← c90509] at e_1475_5
  have hr := e_1475_5
  simp only [e_1475_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14118)) := by
  have c90510 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90511 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90512 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90513 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90514 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90515 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_6
  simp only [← c90510, ← c90511, ← c90512] at e_1475_6
  have e_1475_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_7
  simp only [← c90513, ← c90514, k2, ← c90515] at e_1475_7
  have hr := e_1475_7
  simp only [e_1475_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14119)) := by
  have c90516 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90517 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90518 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90519 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90520 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90521 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_8
  simp only [← c90516, ← c90517, ← c90518] at e_1475_8
  have e_1475_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_9
  simp only [← c90519, ← c90520, k2, ← c90521] at e_1475_9
  have hr := e_1475_9
  simp only [e_1475_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14120)) := by
  have c90522 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90523 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90524 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90525 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90526 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90527 := (publicBatchWrapper89_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_10
  simp only [← c90522, ← c90523, ← c90524] at e_1475_10
  have e_1475_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_11
  simp only [← c90525, ← c90526, k2, ← c90527] at e_1475_11
  have hr := e_1475_11
  simp only [e_1475_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14121)) := by
  have c90528 := (publicBatchWrapper89_copies2829 a h).1
  have c90529 := (publicBatchWrapper89_copies2829 a h).2.1
  have c90530 := (publicBatchWrapper89_copies2829 a h).2.2.1
  have c90531 := (publicBatchWrapper89_copies2829 a h).2.2.2.1
  have c90532 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.1
  have c90533 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_12
  simp only [← c90528, ← c90529, ← c90530] at e_1475_12
  have e_1475_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_13
  simp only [← c90531, ← c90532, k2, ← c90533] at e_1475_13
  have hr := e_1475_13
  simp only [e_1475_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14122)) := by
  have c90534 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.1
  have c90535 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.1
  have c90536 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.1
  have c90537 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.1
  have c90538 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90539 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_14
  simp only [← c90534, ← c90535, ← c90536] at e_1475_14
  have e_1475_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_15
  simp only [← c90537, ← c90538, k2, ← c90539] at e_1475_15
  have hr := e_1475_15
  simp only [e_1475_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14123)) := by
  have c90540 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90541 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90542 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90543 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90544 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90545 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_16
  simp only [← c90540, ← c90541, ← c90542] at e_1475_16
  have e_1475_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_17
  simp only [← c90543, ← c90544, k2, ← c90545] at e_1475_17
  have hr := e_1475_17
  simp only [e_1475_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1475 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14124)) := by
  have c90546 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90547 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90548 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90549 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90550 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90551 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1475_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_18
  simp only [← c90546, ← c90547, ← c90548] at e_1475_18
  have e_1475_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1475 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_19
  simp only [← c90549, ← c90550, k2, ← c90551] at e_1475_19
  have hr := e_1475_19
  simp only [e_1475_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14125)) := by
  have c90552 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90553 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90554 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90555 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90556 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90557 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_0
  simp only [← c90552, ← c90553, ← c90554] at e_1476_0
  have e_1476_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_1
  simp only [← c90555, ← c90556, k2, ← c90557] at e_1476_1
  have hr := e_1476_1
  simp only [e_1476_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14126)) := by
  have c90558 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90559 := (publicBatchWrapper89_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90560 := (publicBatchWrapper89_copies2830 a h).1
  have c90561 := (publicBatchWrapper89_copies2830 a h).2.1
  have c90562 := (publicBatchWrapper89_copies2830 a h).2.2.1
  have c90563 := (publicBatchWrapper89_copies2830 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_2
  simp only [← c90558, ← c90559, ← c90560] at e_1476_2
  have e_1476_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_3
  simp only [← c90561, ← c90562, k2, ← c90563] at e_1476_3
  have hr := e_1476_3
  simp only [e_1476_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14127)) := by
  have c90564 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.1
  have c90565 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.1
  have c90566 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.1
  have c90567 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.1
  have c90568 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.1
  have c90569 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_4
  simp only [← c90564, ← c90565, ← c90566] at e_1476_4
  have e_1476_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_5
  simp only [← c90567, ← c90568, k2, ← c90569] at e_1476_5
  have hr := e_1476_5
  simp only [e_1476_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14128)) := by
  have c90570 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90571 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90572 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90573 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90574 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90575 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_6
  simp only [← c90570, ← c90571, ← c90572] at e_1476_6
  have e_1476_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_7
  simp only [← c90573, ← c90574, k2, ← c90575] at e_1476_7
  have hr := e_1476_7
  simp only [e_1476_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14129)) := by
  have c90576 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90577 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90578 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90579 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90580 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90581 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_8
  simp only [← c90576, ← c90577, ← c90578] at e_1476_8
  have e_1476_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_9
  simp only [← c90579, ← c90580, k2, ← c90581] at e_1476_9
  have hr := e_1476_9
  simp only [e_1476_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14130)) := by
  have c90582 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90583 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90584 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90585 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90586 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90587 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_10
  simp only [← c90582, ← c90583, ← c90584] at e_1476_10
  have e_1476_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_11
  simp only [← c90585, ← c90586, k2, ← c90587] at e_1476_11
  have hr := e_1476_11
  simp only [e_1476_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14131)) := by
  have c90588 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90589 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90590 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90591 := (publicBatchWrapper89_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90592 := (publicBatchWrapper89_copies2831 a h).1
  have c90593 := (publicBatchWrapper89_copies2831 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_12
  simp only [← c90588, ← c90589, ← c90590] at e_1476_12
  have e_1476_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_13
  simp only [← c90591, ← c90592, k2, ← c90593] at e_1476_13
  have hr := e_1476_13
  simp only [e_1476_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14132)) := by
  have c90594 := (publicBatchWrapper89_copies2831 a h).2.2.1
  have c90595 := (publicBatchWrapper89_copies2831 a h).2.2.2.1
  have c90596 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.1
  have c90597 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.1
  have c90598 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.1
  have c90599 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_14
  simp only [← c90594, ← c90595, ← c90596] at e_1476_14
  have e_1476_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_15
  simp only [← c90597, ← c90598, k2, ← c90599] at e_1476_15
  have hr := e_1476_15
  simp only [e_1476_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14133)) := by
  have c90600 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.1
  have c90601 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.1
  have c90602 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90603 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90604 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90605 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_16
  simp only [← c90600, ← c90601, ← c90602] at e_1476_16
  have e_1476_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_17
  simp only [← c90603, ← c90604, k2, ← c90605] at e_1476_17
  have hr := e_1476_17
  simp only [e_1476_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1476 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14134)) := by
  have c90606 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90607 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90608 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90609 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90610 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90611 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1476_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_18
  simp only [← c90606, ← c90607, ← c90608] at e_1476_18
  have e_1476_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1476 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_19
  simp only [← c90609, ← c90610, k2, ← c90611] at e_1476_19
  have hr := e_1476_19
  simp only [e_1476_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14135)) := by
  have c90612 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90613 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90614 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90615 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90616 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90617 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_0
  simp only [← c90612, ← c90613, ← c90614] at e_1477_0
  have e_1477_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_1
  simp only [← c90615, ← c90616, k2, ← c90617] at e_1477_1
  have hr := e_1477_1
  simp only [e_1477_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14136)) := by
  have c90618 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90619 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90620 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90621 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90622 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90623 := (publicBatchWrapper89_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_2
  simp only [← c90618, ← c90619, ← c90620] at e_1477_2
  have e_1477_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_3
  simp only [← c90621, ← c90622, k2, ← c90623] at e_1477_3
  have hr := e_1477_3
  simp only [e_1477_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14137)) := by
  have c90624 := (publicBatchWrapper89_copies2832 a h).1
  have c90625 := (publicBatchWrapper89_copies2832 a h).2.1
  have c90626 := (publicBatchWrapper89_copies2832 a h).2.2.1
  have c90627 := (publicBatchWrapper89_copies2832 a h).2.2.2.1
  have c90628 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.1
  have c90629 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_4
  simp only [← c90624, ← c90625, ← c90626] at e_1477_4
  have e_1477_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_5
  simp only [← c90627, ← c90628, k2, ← c90629] at e_1477_5
  have hr := e_1477_5
  simp only [e_1477_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14138)) := by
  have c90630 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.1
  have c90631 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.1
  have c90632 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.1
  have c90633 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.1
  have c90634 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90635 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_6
  simp only [← c90630, ← c90631, ← c90632] at e_1477_6
  have e_1477_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_7
  simp only [← c90633, ← c90634, k2, ← c90635] at e_1477_7
  have hr := e_1477_7
  simp only [e_1477_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14139)) := by
  have c90636 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90637 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90638 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90639 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90640 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90641 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_8
  simp only [← c90636, ← c90637, ← c90638] at e_1477_8
  have e_1477_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_9
  simp only [← c90639, ← c90640, k2, ← c90641] at e_1477_9
  have hr := e_1477_9
  simp only [e_1477_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14140)) := by
  have c90642 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90643 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90644 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90645 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90646 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90647 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_10
  simp only [← c90642, ← c90643, ← c90644] at e_1477_10
  have e_1477_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_11
  simp only [← c90645, ← c90646, k2, ← c90647] at e_1477_11
  have hr := e_1477_11
  simp only [e_1477_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14141)) := by
  have c90648 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90649 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90650 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90651 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90652 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90653 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_12
  simp only [← c90648, ← c90649, ← c90650] at e_1477_12
  have e_1477_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_13
  simp only [← c90651, ← c90652, k2, ← c90653] at e_1477_13
  have hr := e_1477_13
  simp only [e_1477_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14142)) := by
  have c90654 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90655 := (publicBatchWrapper89_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90656 := (publicBatchWrapper89_copies2833 a h).1
  have c90657 := (publicBatchWrapper89_copies2833 a h).2.1
  have c90658 := (publicBatchWrapper89_copies2833 a h).2.2.1
  have c90659 := (publicBatchWrapper89_copies2833 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_14
  simp only [← c90654, ← c90655, ← c90656] at e_1477_14
  have e_1477_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_15
  simp only [← c90657, ← c90658, k2, ← c90659] at e_1477_15
  have hr := e_1477_15
  simp only [e_1477_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14143)) := by
  have c90660 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.1
  have c90661 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.1
  have c90662 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.1
  have c90663 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.1
  have c90664 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.1
  have c90665 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_16
  simp only [← c90660, ← c90661, ← c90662] at e_1477_16
  have e_1477_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_17
  simp only [← c90663, ← c90664, k2, ← c90665] at e_1477_17
  have hr := e_1477_17
  simp only [e_1477_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1477 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14144)) := by
  have c90666 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90667 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90668 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90669 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90670 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90671 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1477_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_18
  simp only [← c90666, ← c90667, ← c90668] at e_1477_18
  have e_1477_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_19
  simp only [← c90669, ← c90670, k2, ← c90671] at e_1477_19
  have hr := e_1477_19
  simp only [e_1477_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14145)) := by
  have c90672 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90673 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90674 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90675 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90676 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90677 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_0
  simp only [← c90672, ← c90673, ← c90674] at e_1478_0
  have e_1478_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_1
  simp only [← c90675, ← c90676, k2, ← c90677] at e_1478_1
  have hr := e_1478_1
  simp only [e_1478_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14146)) := by
  have c90678 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90679 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90680 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90681 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90682 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90683 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_2
  simp only [← c90678, ← c90679, ← c90680] at e_1478_2
  have e_1478_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_3
  simp only [← c90681, ← c90682, k2, ← c90683] at e_1478_3
  have hr := e_1478_3
  simp only [e_1478_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14147)) := by
  have c90684 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90685 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90686 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90687 := (publicBatchWrapper89_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90688 := (publicBatchWrapper89_copies2834 a h).1
  have c90689 := (publicBatchWrapper89_copies2834 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_4
  simp only [← c90684, ← c90685, ← c90686] at e_1478_4
  have e_1478_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_5
  simp only [← c90687, ← c90688, k2, ← c90689] at e_1478_5
  have hr := e_1478_5
  simp only [e_1478_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14148)) := by
  have c90690 := (publicBatchWrapper89_copies2834 a h).2.2.1
  have c90691 := (publicBatchWrapper89_copies2834 a h).2.2.2.1
  have c90692 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.1
  have c90693 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.1
  have c90694 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.1
  have c90695 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_6
  simp only [← c90690, ← c90691, ← c90692] at e_1478_6
  have e_1478_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_7
  simp only [← c90693, ← c90694, k2, ← c90695] at e_1478_7
  have hr := e_1478_7
  simp only [e_1478_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14149)) := by
  have c90696 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.1
  have c90697 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.1
  have c90698 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90699 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90700 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90701 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_8
  simp only [← c90696, ← c90697, ← c90698] at e_1478_8
  have e_1478_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_9
  simp only [← c90699, ← c90700, k2, ← c90701] at e_1478_9
  have hr := e_1478_9
  simp only [e_1478_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14150)) := by
  have c90702 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90703 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90704 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90705 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90706 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90707 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_10
  simp only [← c90702, ← c90703, ← c90704] at e_1478_10
  have e_1478_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_11
  simp only [← c90705, ← c90706, k2, ← c90707] at e_1478_11
  have hr := e_1478_11
  simp only [e_1478_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14151)) := by
  have c90708 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90709 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90710 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90711 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90712 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90713 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_12
  simp only [← c90708, ← c90709, ← c90710] at e_1478_12
  have e_1478_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_13
  simp only [← c90711, ← c90712, k2, ← c90713] at e_1478_13
  have hr := e_1478_13
  simp only [e_1478_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14152)) := by
  have c90714 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90715 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90716 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90717 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90718 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90719 := (publicBatchWrapper89_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_14
  simp only [← c90714, ← c90715, ← c90716] at e_1478_14
  have e_1478_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_15
  simp only [← c90717, ← c90718, k2, ← c90719] at e_1478_15
  have hr := e_1478_15
  simp only [e_1478_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14153)) := by
  have c90720 := (publicBatchWrapper89_copies2835 a h).1
  have c90721 := (publicBatchWrapper89_copies2835 a h).2.1
  have c90722 := (publicBatchWrapper89_copies2835 a h).2.2.1
  have c90723 := (publicBatchWrapper89_copies2835 a h).2.2.2.1
  have c90724 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.1
  have c90725 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_16
  simp only [← c90720, ← c90721, ← c90722] at e_1478_16
  have e_1478_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_17
  simp only [← c90723, ← c90724, k2, ← c90725] at e_1478_17
  have hr := e_1478_17
  simp only [e_1478_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1478 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14154)) := by
  have c90726 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.1
  have c90727 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.1
  have c90728 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.1
  have c90729 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.1
  have c90730 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90731 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1478_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_18
  simp only [← c90726, ← c90727, ← c90728] at e_1478_18
  have e_1478_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_19
  simp only [← c90729, ← c90730, k2, ← c90731] at e_1478_19
  have hr := e_1478_19
  simp only [e_1478_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14155)) := by
  have c90732 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90733 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90734 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90735 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90736 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90737 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_0
  simp only [← c90732, ← c90733, ← c90734] at e_1479_0
  have e_1479_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_1
  simp only [← c90735, ← c90736, k2, ← c90737] at e_1479_1
  have hr := e_1479_1
  simp only [e_1479_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14156)) := by
  have c90738 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90739 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90740 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90741 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90742 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90743 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_2
  simp only [← c90738, ← c90739, ← c90740] at e_1479_2
  have e_1479_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_3
  simp only [← c90741, ← c90742, k2, ← c90743] at e_1479_3
  have hr := e_1479_3
  simp only [e_1479_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14157)) := by
  have c90744 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90745 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90746 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90747 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90748 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90749 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_4
  simp only [← c90744, ← c90745, ← c90746] at e_1479_4
  have e_1479_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_5
  simp only [← c90747, ← c90748, k2, ← c90749] at e_1479_5
  have hr := e_1479_5
  simp only [e_1479_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14158)) := by
  have c90750 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90751 := (publicBatchWrapper89_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90752 := (publicBatchWrapper89_copies2836 a h).1
  have c90753 := (publicBatchWrapper89_copies2836 a h).2.1
  have c90754 := (publicBatchWrapper89_copies2836 a h).2.2.1
  have c90755 := (publicBatchWrapper89_copies2836 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_6
  simp only [← c90750, ← c90751, ← c90752] at e_1479_6
  have e_1479_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_7
  simp only [← c90753, ← c90754, k2, ← c90755] at e_1479_7
  have hr := e_1479_7
  simp only [e_1479_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14159)) := by
  have c90756 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.1
  have c90757 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.1
  have c90758 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.1
  have c90759 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.1
  have c90760 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.1
  have c90761 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_8
  simp only [← c90756, ← c90757, ← c90758] at e_1479_8
  have e_1479_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_9
  simp only [← c90759, ← c90760, k2, ← c90761] at e_1479_9
  have hr := e_1479_9
  simp only [e_1479_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14160)) := by
  have c90762 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90763 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90764 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90765 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90766 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90767 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_10
  simp only [← c90762, ← c90763, ← c90764] at e_1479_10
  have e_1479_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_11
  simp only [← c90765, ← c90766, k2, ← c90767] at e_1479_11
  have hr := e_1479_11
  simp only [e_1479_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14161)) := by
  have c90768 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90769 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90770 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90771 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90772 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90773 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_12
  simp only [← c90768, ← c90769, ← c90770] at e_1479_12
  have e_1479_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_13
  simp only [← c90771, ← c90772, k2, ← c90773] at e_1479_13
  have hr := e_1479_13
  simp only [e_1479_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14162)) := by
  have c90774 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90775 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90776 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90777 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90778 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90779 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_14
  simp only [← c90774, ← c90775, ← c90776] at e_1479_14
  have e_1479_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_15
  simp only [← c90777, ← c90778, k2, ← c90779] at e_1479_15
  have hr := e_1479_15
  simp only [e_1479_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14163)) := by
  have c90780 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90781 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90782 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90783 := (publicBatchWrapper89_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90784 := (publicBatchWrapper89_copies2837 a h).1
  have c90785 := (publicBatchWrapper89_copies2837 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_16
  simp only [← c90780, ← c90781, ← c90782] at e_1479_16
  have e_1479_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_17
  simp only [← c90783, ← c90784, k2, ← c90785] at e_1479_17
  have hr := e_1479_17
  simp only [e_1479_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1479 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14164)) := by
  have c90786 := (publicBatchWrapper89_copies2837 a h).2.2.1
  have c90787 := (publicBatchWrapper89_copies2837 a h).2.2.2.1
  have c90788 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.1
  have c90789 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.1
  have c90790 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.1
  have c90791 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1479_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_18
  simp only [← c90786, ← c90787, ← c90788] at e_1479_18
  have e_1479_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_19
  simp only [← c90789, ← c90790, k2, ← c90791] at e_1479_19
  have hr := e_1479_19
  simp only [e_1479_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14165)) := by
  have c90792 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.1
  have c90793 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.1
  have c90794 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90795 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90796 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90797 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_0
  simp only [← c90792, ← c90793, ← c90794] at e_1480_0
  have e_1480_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_1
  simp only [← c90795, ← c90796, k2, ← c90797] at e_1480_1
  have hr := e_1480_1
  simp only [e_1480_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14166)) := by
  have c90798 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90799 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90800 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90801 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90802 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90803 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_2
  simp only [← c90798, ← c90799, ← c90800] at e_1480_2
  have e_1480_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_3
  simp only [← c90801, ← c90802, k2, ← c90803] at e_1480_3
  have hr := e_1480_3
  simp only [e_1480_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14167)) := by
  have c90804 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90805 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90806 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90807 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90808 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90809 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_4
  simp only [← c90804, ← c90805, ← c90806] at e_1480_4
  have e_1480_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_5
  simp only [← c90807, ← c90808, k2, ← c90809] at e_1480_5
  have hr := e_1480_5
  simp only [e_1480_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14168)) := by
  have c90810 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90811 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90812 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90813 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90814 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90815 := (publicBatchWrapper89_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_6
  simp only [← c90810, ← c90811, ← c90812] at e_1480_6
  have e_1480_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_7
  simp only [← c90813, ← c90814, k2, ← c90815] at e_1480_7
  have hr := e_1480_7
  simp only [e_1480_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14169)) := by
  have c90816 := (publicBatchWrapper89_copies2838 a h).1
  have c90817 := (publicBatchWrapper89_copies2838 a h).2.1
  have c90818 := (publicBatchWrapper89_copies2838 a h).2.2.1
  have c90819 := (publicBatchWrapper89_copies2838 a h).2.2.2.1
  have c90820 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.1
  have c90821 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_8
  simp only [← c90816, ← c90817, ← c90818] at e_1480_8
  have e_1480_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_9
  simp only [← c90819, ← c90820, k2, ← c90821] at e_1480_9
  have hr := e_1480_9
  simp only [e_1480_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14210)) := by
  have c90822 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.1
  have c90823 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.1
  have c90824 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.1
  have c90825 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.1
  have c90826 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90827 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_10
  simp only [← c90822, ← c90823, ← c90824] at e_1480_10
  have e_1480_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_11
  simp only [← c90825, ← c90826, k2, ← c90827] at e_1480_11
  have hr := e_1480_11
  simp only [e_1480_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14211)) := by
  have c90828 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90829 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90830 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90831 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90832 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90833 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_12
  simp only [← c90828, ← c90829, ← c90830] at e_1480_12
  have e_1480_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_13
  simp only [← c90831, ← c90832, k2, ← c90833] at e_1480_13
  have hr := e_1480_13
  simp only [e_1480_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14212)) := by
  have c90834 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90835 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90836 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90837 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90838 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90839 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_14
  simp only [← c90834, ← c90835, ← c90836] at e_1480_14
  have e_1480_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_15
  simp only [← c90837, ← c90838, k2, ← c90839] at e_1480_15
  have hr := e_1480_15
  simp only [e_1480_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14213)) := by
  have c90840 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90841 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90842 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90843 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90844 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90845 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_16
  simp only [← c90840, ← c90841, ← c90842] at e_1480_16
  have e_1480_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_17
  simp only [← c90843, ← c90844, k2, ← c90845] at e_1480_17
  have hr := e_1480_17
  simp only [e_1480_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1480 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14214)) := by
  have c90846 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90847 := (publicBatchWrapper89_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90848 := (publicBatchWrapper89_copies2839 a h).1
  have c90849 := (publicBatchWrapper89_copies2839 a h).2.1
  have c90850 := (publicBatchWrapper89_copies2839 a h).2.2.1
  have c90851 := (publicBatchWrapper89_copies2839 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1480_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_18
  simp only [← c90846, ← c90847, ← c90848] at e_1480_18
  have e_1480_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_19
  simp only [← c90849, ← c90850, k2, ← c90851] at e_1480_19
  have hr := e_1480_19
  simp only [e_1480_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14215)) := by
  have c90852 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.1
  have c90853 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.1
  have c90854 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.1
  have c90855 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.1
  have c90856 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.1
  have c90857 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_0
  simp only [← c90852, ← c90853, ← c90854] at e_1481_0
  have e_1481_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_1
  simp only [← c90855, ← c90856, k2, ← c90857] at e_1481_1
  have hr := e_1481_1
  simp only [e_1481_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14216)) := by
  have c90858 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90859 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90860 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90861 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90862 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90863 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_2
  simp only [← c90858, ← c90859, ← c90860] at e_1481_2
  have e_1481_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_3
  simp only [← c90861, ← c90862, k2, ← c90863] at e_1481_3
  have hr := e_1481_3
  simp only [e_1481_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14217)) := by
  have c90864 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90865 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90866 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90867 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90868 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90869 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_4
  simp only [← c90864, ← c90865, ← c90866] at e_1481_4
  have e_1481_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_5
  simp only [← c90867, ← c90868, k2, ← c90869] at e_1481_5
  have hr := e_1481_5
  simp only [e_1481_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14218)) := by
  have c90870 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90871 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90872 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90873 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90874 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90875 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_6
  simp only [← c90870, ← c90871, ← c90872] at e_1481_6
  have e_1481_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_7
  simp only [← c90873, ← c90874, k2, ← c90875] at e_1481_7
  have hr := e_1481_7
  simp only [e_1481_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14219)) := by
  have c90876 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90877 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90878 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90879 := (publicBatchWrapper89_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90880 := (publicBatchWrapper89_copies2840 a h).1
  have c90881 := (publicBatchWrapper89_copies2840 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_8
  simp only [← c90876, ← c90877, ← c90878] at e_1481_8
  have e_1481_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_9
  simp only [← c90879, ← c90880, k2, ← c90881] at e_1481_9
  have hr := e_1481_9
  simp only [e_1481_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14220)) := by
  have c90882 := (publicBatchWrapper89_copies2840 a h).2.2.1
  have c90883 := (publicBatchWrapper89_copies2840 a h).2.2.2.1
  have c90884 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.1
  have c90885 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.1
  have c90886 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.1
  have c90887 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_10
  simp only [← c90882, ← c90883, ← c90884] at e_1481_10
  have e_1481_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_11
  simp only [← c90885, ← c90886, k2, ← c90887] at e_1481_11
  have hr := e_1481_11
  simp only [e_1481_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14221)) := by
  have c90888 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.1
  have c90889 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.1
  have c90890 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90891 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90892 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90893 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_12
  simp only [← c90888, ← c90889, ← c90890] at e_1481_12
  have e_1481_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_13
  simp only [← c90891, ← c90892, k2, ← c90893] at e_1481_13
  have hr := e_1481_13
  simp only [e_1481_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14222)) := by
  have c90894 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90895 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90896 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90897 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90898 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90899 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_14
  simp only [← c90894, ← c90895, ← c90896] at e_1481_14
  have e_1481_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_15
  simp only [← c90897, ← c90898, k2, ← c90899] at e_1481_15
  have hr := e_1481_15
  simp only [e_1481_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14223)) := by
  have c90900 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90901 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90902 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90903 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90904 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90905 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_16
  simp only [← c90900, ← c90901, ← c90902] at e_1481_16
  have e_1481_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_17
  simp only [← c90903, ← c90904, k2, ← c90905] at e_1481_17
  have hr := e_1481_17
  simp only [e_1481_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1481 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14224)) := by
  have c90906 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90907 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90908 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90909 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90910 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90911 := (publicBatchWrapper89_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1481_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_18
  simp only [← c90906, ← c90907, ← c90908] at e_1481_18
  have e_1481_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_19
  simp only [← c90909, ← c90910, k2, ← c90911] at e_1481_19
  have hr := e_1481_19
  simp only [e_1481_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14225)) := by
  have c90912 := (publicBatchWrapper89_copies2841 a h).1
  have c90913 := (publicBatchWrapper89_copies2841 a h).2.1
  have c90914 := (publicBatchWrapper89_copies2841 a h).2.2.1
  have c90915 := (publicBatchWrapper89_copies2841 a h).2.2.2.1
  have c90916 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.1
  have c90917 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_0
  simp only [← c90912, ← c90913, ← c90914] at e_1482_0
  have e_1482_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_1
  simp only [← c90915, ← c90916, k2, ← c90917] at e_1482_1
  have hr := e_1482_1
  simp only [e_1482_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14226)) := by
  have c90918 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.1
  have c90919 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.1
  have c90920 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.1
  have c90921 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.1
  have c90922 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90923 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_2
  simp only [← c90918, ← c90919, ← c90920] at e_1482_2
  have e_1482_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_3
  simp only [← c90921, ← c90922, k2, ← c90923] at e_1482_3
  have hr := e_1482_3
  simp only [e_1482_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f14403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1482 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14227)) := by
  have c90924 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90925 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90926 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90927 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90928 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90929 := (publicBatchWrapper89_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1482_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_4
  simp only [← c90924, ← c90925, ← c90926] at e_1482_4
  have e_1482_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_5
  simp only [← c90927, ← c90928, k2, ← c90929] at e_1482_5
  have hr := e_1482_5
  simp only [e_1482_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
