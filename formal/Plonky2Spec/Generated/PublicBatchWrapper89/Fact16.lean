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

theorem publicBatchWrapper89_f12307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11611)) := by
  have c78348 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78349 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78350 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78351 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78352 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78353 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_12 := arithEq_of_rows h (row := 1272) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_12
  simp only [← c78348, ← c78349, ← c78350] at e_1272_12
  have e_1272_13 := arithEq_of_rows h (row := 1272) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_13
  simp only [← c78351, ← c78352, k2, ← c78353] at e_1272_13
  have hr := e_1272_13
  simp only [e_1272_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11612)) := by
  have c78354 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78355 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78356 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78357 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78358 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78359 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_14 := arithEq_of_rows h (row := 1272) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_14
  simp only [← c78354, ← c78355, ← c78356] at e_1272_14
  have e_1272_15 := arithEq_of_rows h (row := 1272) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_15
  simp only [← c78357, ← c78358, k2, ← c78359] at e_1272_15
  have hr := e_1272_15
  simp only [e_1272_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11613)) := by
  have c78360 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78361 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78362 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78363 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78364 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78365 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_16 := arithEq_of_rows h (row := 1272) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_16
  simp only [← c78360, ← c78361, ← c78362] at e_1272_16
  have e_1272_17 := arithEq_of_rows h (row := 1272) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_17
  simp only [← c78363, ← c78364, k2, ← c78365] at e_1272_17
  have hr := e_1272_17
  simp only [e_1272_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1272 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11614)) := by
  have c78366 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78367 := (publicBatchWrapper89_copies2448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78368 := (publicBatchWrapper89_copies2449 a h).1
  have c78369 := (publicBatchWrapper89_copies2449 a h).2.1
  have c78370 := (publicBatchWrapper89_copies2449 a h).2.2.1
  have c78371 := (publicBatchWrapper89_copies2449 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1272_18 := arithEq_of_rows h (row := 1272) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_18
  simp only [← c78366, ← c78367, ← c78368] at e_1272_18
  have e_1272_19 := arithEq_of_rows h (row := 1272) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1272_19
  simp only [← c78369, ← c78370, k2, ← c78371] at e_1272_19
  have hr := e_1272_19
  simp only [e_1272_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11615)) := by
  have c78372 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.1
  have c78373 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.1
  have c78374 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.1
  have c78375 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.1
  have c78376 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.1
  have c78377 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_0 := arithEq_of_rows h (row := 1273) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_0
  simp only [← c78372, ← c78373, ← c78374] at e_1273_0
  have e_1273_1 := arithEq_of_rows h (row := 1273) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_1
  simp only [← c78375, ← c78376, k2, ← c78377] at e_1273_1
  have hr := e_1273_1
  simp only [e_1273_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11616)) := by
  have c78378 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78379 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78380 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78381 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78382 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78383 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_2 := arithEq_of_rows h (row := 1273) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_2
  simp only [← c78378, ← c78379, ← c78380] at e_1273_2
  have e_1273_3 := arithEq_of_rows h (row := 1273) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_3
  simp only [← c78381, ← c78382, k2, ← c78383] at e_1273_3
  have hr := e_1273_3
  simp only [e_1273_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11617)) := by
  have c78384 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78385 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78386 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78387 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78388 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78389 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_4 := arithEq_of_rows h (row := 1273) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_4
  simp only [← c78384, ← c78385, ← c78386] at e_1273_4
  have e_1273_5 := arithEq_of_rows h (row := 1273) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_5
  simp only [← c78387, ← c78388, k2, ← c78389] at e_1273_5
  have hr := e_1273_5
  simp only [e_1273_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11618)) := by
  have c78390 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78391 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78392 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78393 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78394 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78395 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_6 := arithEq_of_rows h (row := 1273) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_6
  simp only [← c78390, ← c78391, ← c78392] at e_1273_6
  have e_1273_7 := arithEq_of_rows h (row := 1273) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_7
  simp only [← c78393, ← c78394, k2, ← c78395] at e_1273_7
  have hr := e_1273_7
  simp only [e_1273_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11619)) := by
  have c78396 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78397 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78398 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78399 := (publicBatchWrapper89_copies2449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78400 := (publicBatchWrapper89_copies2450 a h).1
  have c78401 := (publicBatchWrapper89_copies2450 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_8 := arithEq_of_rows h (row := 1273) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_8
  simp only [← c78396, ← c78397, ← c78398] at e_1273_8
  have e_1273_9 := arithEq_of_rows h (row := 1273) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_9
  simp only [← c78399, ← c78400, k2, ← c78401] at e_1273_9
  have hr := e_1273_9
  simp only [e_1273_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11620)) := by
  have c78402 := (publicBatchWrapper89_copies2450 a h).2.2.1
  have c78403 := (publicBatchWrapper89_copies2450 a h).2.2.2.1
  have c78404 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.1
  have c78405 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.1
  have c78406 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.1
  have c78407 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_10 := arithEq_of_rows h (row := 1273) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_10
  simp only [← c78402, ← c78403, ← c78404] at e_1273_10
  have e_1273_11 := arithEq_of_rows h (row := 1273) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_11
  simp only [← c78405, ← c78406, k2, ← c78407] at e_1273_11
  have hr := e_1273_11
  simp only [e_1273_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11621)) := by
  have c78408 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.1
  have c78409 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.1
  have c78410 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78411 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78412 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78413 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_12 := arithEq_of_rows h (row := 1273) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_12
  simp only [← c78408, ← c78409, ← c78410] at e_1273_12
  have e_1273_13 := arithEq_of_rows h (row := 1273) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_13
  simp only [← c78411, ← c78412, k2, ← c78413] at e_1273_13
  have hr := e_1273_13
  simp only [e_1273_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11622)) := by
  have c78414 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78415 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78416 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78417 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78418 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78419 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_14 := arithEq_of_rows h (row := 1273) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_14
  simp only [← c78414, ← c78415, ← c78416] at e_1273_14
  have e_1273_15 := arithEq_of_rows h (row := 1273) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_15
  simp only [← c78417, ← c78418, k2, ← c78419] at e_1273_15
  have hr := e_1273_15
  simp only [e_1273_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11623)) := by
  have c78420 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78421 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78422 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78423 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78424 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78425 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_16 := arithEq_of_rows h (row := 1273) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_16
  simp only [← c78420, ← c78421, ← c78422] at e_1273_16
  have e_1273_17 := arithEq_of_rows h (row := 1273) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_17
  simp only [← c78423, ← c78424, k2, ← c78425] at e_1273_17
  have hr := e_1273_17
  simp only [e_1273_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1273 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11624)) := by
  have c78426 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78427 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78428 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78429 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78430 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78431 := (publicBatchWrapper89_copies2450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1273_18 := arithEq_of_rows h (row := 1273) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_18
  simp only [← c78426, ← c78427, ← c78428] at e_1273_18
  have e_1273_19 := arithEq_of_rows h (row := 1273) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1273_19
  simp only [← c78429, ← c78430, k2, ← c78431] at e_1273_19
  have hr := e_1273_19
  simp only [e_1273_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11625)) := by
  have c78432 := (publicBatchWrapper89_copies2451 a h).1
  have c78433 := (publicBatchWrapper89_copies2451 a h).2.1
  have c78434 := (publicBatchWrapper89_copies2451 a h).2.2.1
  have c78435 := (publicBatchWrapper89_copies2451 a h).2.2.2.1
  have c78436 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.1
  have c78437 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_0 := arithEq_of_rows h (row := 1274) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_0
  simp only [← c78432, ← c78433, ← c78434] at e_1274_0
  have e_1274_1 := arithEq_of_rows h (row := 1274) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_1
  simp only [← c78435, ← c78436, k2, ← c78437] at e_1274_1
  have hr := e_1274_1
  simp only [e_1274_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11626)) := by
  have c78438 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.1
  have c78439 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.1
  have c78440 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.1
  have c78441 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.1
  have c78442 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78443 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_2 := arithEq_of_rows h (row := 1274) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_2
  simp only [← c78438, ← c78439, ← c78440] at e_1274_2
  have e_1274_3 := arithEq_of_rows h (row := 1274) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_3
  simp only [← c78441, ← c78442, k2, ← c78443] at e_1274_3
  have hr := e_1274_3
  simp only [e_1274_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11627)) := by
  have c78444 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78445 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78446 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78447 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78448 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78449 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_4 := arithEq_of_rows h (row := 1274) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_4
  simp only [← c78444, ← c78445, ← c78446] at e_1274_4
  have e_1274_5 := arithEq_of_rows h (row := 1274) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_5
  simp only [← c78447, ← c78448, k2, ← c78449] at e_1274_5
  have hr := e_1274_5
  simp only [e_1274_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11628)) := by
  have c78450 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78451 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78452 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78453 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78454 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78455 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_6 := arithEq_of_rows h (row := 1274) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_6
  simp only [← c78450, ← c78451, ← c78452] at e_1274_6
  have e_1274_7 := arithEq_of_rows h (row := 1274) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_7
  simp only [← c78453, ← c78454, k2, ← c78455] at e_1274_7
  have hr := e_1274_7
  simp only [e_1274_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11629)) := by
  have c78456 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78457 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78458 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78459 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78460 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78461 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_8 := arithEq_of_rows h (row := 1274) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_8
  simp only [← c78456, ← c78457, ← c78458] at e_1274_8
  have e_1274_9 := arithEq_of_rows h (row := 1274) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_9
  simp only [← c78459, ← c78460, k2, ← c78461] at e_1274_9
  have hr := e_1274_9
  simp only [e_1274_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11630)) := by
  have c78462 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78463 := (publicBatchWrapper89_copies2451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78464 := (publicBatchWrapper89_copies2452 a h).1
  have c78465 := (publicBatchWrapper89_copies2452 a h).2.1
  have c78466 := (publicBatchWrapper89_copies2452 a h).2.2.1
  have c78467 := (publicBatchWrapper89_copies2452 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_10 := arithEq_of_rows h (row := 1274) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_10
  simp only [← c78462, ← c78463, ← c78464] at e_1274_10
  have e_1274_11 := arithEq_of_rows h (row := 1274) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_11
  simp only [← c78465, ← c78466, k2, ← c78467] at e_1274_11
  have hr := e_1274_11
  simp only [e_1274_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11631)) := by
  have c78468 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.1
  have c78469 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.1
  have c78470 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.1
  have c78471 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.1
  have c78472 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.1
  have c78473 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_12 := arithEq_of_rows h (row := 1274) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_12
  simp only [← c78468, ← c78469, ← c78470] at e_1274_12
  have e_1274_13 := arithEq_of_rows h (row := 1274) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_13
  simp only [← c78471, ← c78472, k2, ← c78473] at e_1274_13
  have hr := e_1274_13
  simp only [e_1274_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11632)) := by
  have c78474 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78475 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78476 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78477 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78478 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78479 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_14 := arithEq_of_rows h (row := 1274) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_14
  simp only [← c78474, ← c78475, ← c78476] at e_1274_14
  have e_1274_15 := arithEq_of_rows h (row := 1274) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_15
  simp only [← c78477, ← c78478, k2, ← c78479] at e_1274_15
  have hr := e_1274_15
  simp only [e_1274_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11633)) := by
  have c78480 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78481 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78482 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78483 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78484 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78485 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_16 := arithEq_of_rows h (row := 1274) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_16
  simp only [← c78480, ← c78481, ← c78482] at e_1274_16
  have e_1274_17 := arithEq_of_rows h (row := 1274) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_17
  simp only [← c78483, ← c78484, k2, ← c78485] at e_1274_17
  have hr := e_1274_17
  simp only [e_1274_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1274 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11634)) := by
  have c78486 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78487 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78488 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78489 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78490 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78491 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1274_18 := arithEq_of_rows h (row := 1274) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_18
  simp only [← c78486, ← c78487, ← c78488] at e_1274_18
  have e_1274_19 := arithEq_of_rows h (row := 1274) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1274_19
  simp only [← c78489, ← c78490, k2, ← c78491] at e_1274_19
  have hr := e_1274_19
  simp only [e_1274_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11635)) := by
  have c78492 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78493 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78494 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78495 := (publicBatchWrapper89_copies2452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78496 := (publicBatchWrapper89_copies2453 a h).1
  have c78497 := (publicBatchWrapper89_copies2453 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_0 := arithEq_of_rows h (row := 1275) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_0
  simp only [← c78492, ← c78493, ← c78494] at e_1275_0
  have e_1275_1 := arithEq_of_rows h (row := 1275) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_1
  simp only [← c78495, ← c78496, k2, ← c78497] at e_1275_1
  have hr := e_1275_1
  simp only [e_1275_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11636)) := by
  have c78498 := (publicBatchWrapper89_copies2453 a h).2.2.1
  have c78499 := (publicBatchWrapper89_copies2453 a h).2.2.2.1
  have c78500 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.1
  have c78501 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.1
  have c78502 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.1
  have c78503 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_2 := arithEq_of_rows h (row := 1275) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_2
  simp only [← c78498, ← c78499, ← c78500] at e_1275_2
  have e_1275_3 := arithEq_of_rows h (row := 1275) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_3
  simp only [← c78501, ← c78502, k2, ← c78503] at e_1275_3
  have hr := e_1275_3
  simp only [e_1275_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11637)) := by
  have c78504 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.1
  have c78505 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.1
  have c78506 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78507 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78508 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78509 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_4 := arithEq_of_rows h (row := 1275) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_4
  simp only [← c78504, ← c78505, ← c78506] at e_1275_4
  have e_1275_5 := arithEq_of_rows h (row := 1275) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_5
  simp only [← c78507, ← c78508, k2, ← c78509] at e_1275_5
  have hr := e_1275_5
  simp only [e_1275_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11638)) := by
  have c78510 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78511 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78512 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78513 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78514 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78515 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_6 := arithEq_of_rows h (row := 1275) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_6
  simp only [← c78510, ← c78511, ← c78512] at e_1275_6
  have e_1275_7 := arithEq_of_rows h (row := 1275) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_7
  simp only [← c78513, ← c78514, k2, ← c78515] at e_1275_7
  have hr := e_1275_7
  simp only [e_1275_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11639)) := by
  have c78516 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78517 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78518 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78519 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78520 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78521 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_8 := arithEq_of_rows h (row := 1275) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_8
  simp only [← c78516, ← c78517, ← c78518] at e_1275_8
  have e_1275_9 := arithEq_of_rows h (row := 1275) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_9
  simp only [← c78519, ← c78520, k2, ← c78521] at e_1275_9
  have hr := e_1275_9
  simp only [e_1275_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11640)) := by
  have c78522 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78523 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78524 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78525 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78526 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78527 := (publicBatchWrapper89_copies2453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_10 := arithEq_of_rows h (row := 1275) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_10
  simp only [← c78522, ← c78523, ← c78524] at e_1275_10
  have e_1275_11 := arithEq_of_rows h (row := 1275) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_11
  simp only [← c78525, ← c78526, k2, ← c78527] at e_1275_11
  have hr := e_1275_11
  simp only [e_1275_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11641)) := by
  have c78528 := (publicBatchWrapper89_copies2454 a h).1
  have c78529 := (publicBatchWrapper89_copies2454 a h).2.1
  have c78530 := (publicBatchWrapper89_copies2454 a h).2.2.1
  have c78531 := (publicBatchWrapper89_copies2454 a h).2.2.2.1
  have c78532 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.1
  have c78533 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_12 := arithEq_of_rows h (row := 1275) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_12
  simp only [← c78528, ← c78529, ← c78530] at e_1275_12
  have e_1275_13 := arithEq_of_rows h (row := 1275) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_13
  simp only [← c78531, ← c78532, k2, ← c78533] at e_1275_13
  have hr := e_1275_13
  simp only [e_1275_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11642)) := by
  have c78534 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.1
  have c78535 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.1
  have c78536 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.1
  have c78537 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.1
  have c78538 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78539 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_14 := arithEq_of_rows h (row := 1275) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_14
  simp only [← c78534, ← c78535, ← c78536] at e_1275_14
  have e_1275_15 := arithEq_of_rows h (row := 1275) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_15
  simp only [← c78537, ← c78538, k2, ← c78539] at e_1275_15
  have hr := e_1275_15
  simp only [e_1275_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11643)) := by
  have c78540 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78541 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78542 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78543 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78544 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78545 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_16 := arithEq_of_rows h (row := 1275) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_16
  simp only [← c78540, ← c78541, ← c78542] at e_1275_16
  have e_1275_17 := arithEq_of_rows h (row := 1275) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_17
  simp only [← c78543, ← c78544, k2, ← c78545] at e_1275_17
  have hr := e_1275_17
  simp only [e_1275_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1275 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11644)) := by
  have c78546 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78547 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78548 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78549 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78550 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78551 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1275_18 := arithEq_of_rows h (row := 1275) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_18
  simp only [← c78546, ← c78547, ← c78548] at e_1275_18
  have e_1275_19 := arithEq_of_rows h (row := 1275) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1275_19
  simp only [← c78549, ← c78550, k2, ← c78551] at e_1275_19
  have hr := e_1275_19
  simp only [e_1275_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11645)) := by
  have c78552 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78553 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78554 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78555 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78556 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78557 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_0 := arithEq_of_rows h (row := 1276) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_0
  simp only [← c78552, ← c78553, ← c78554] at e_1276_0
  have e_1276_1 := arithEq_of_rows h (row := 1276) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_1
  simp only [← c78555, ← c78556, k2, ← c78557] at e_1276_1
  have hr := e_1276_1
  simp only [e_1276_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11646)) := by
  have c78558 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78559 := (publicBatchWrapper89_copies2454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78560 := (publicBatchWrapper89_copies2455 a h).1
  have c78561 := (publicBatchWrapper89_copies2455 a h).2.1
  have c78562 := (publicBatchWrapper89_copies2455 a h).2.2.1
  have c78563 := (publicBatchWrapper89_copies2455 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_2 := arithEq_of_rows h (row := 1276) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_2
  simp only [← c78558, ← c78559, ← c78560] at e_1276_2
  have e_1276_3 := arithEq_of_rows h (row := 1276) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_3
  simp only [← c78561, ← c78562, k2, ← c78563] at e_1276_3
  have hr := e_1276_3
  simp only [e_1276_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11647)) := by
  have c78564 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.1
  have c78565 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.1
  have c78566 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.1
  have c78567 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.1
  have c78568 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.1
  have c78569 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_4 := arithEq_of_rows h (row := 1276) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_4
  simp only [← c78564, ← c78565, ← c78566] at e_1276_4
  have e_1276_5 := arithEq_of_rows h (row := 1276) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_5
  simp only [← c78567, ← c78568, k2, ← c78569] at e_1276_5
  have hr := e_1276_5
  simp only [e_1276_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11648)) := by
  have c78570 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78571 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78572 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78573 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78574 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78575 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_6 := arithEq_of_rows h (row := 1276) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_6
  simp only [← c78570, ← c78571, ← c78572] at e_1276_6
  have e_1276_7 := arithEq_of_rows h (row := 1276) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_7
  simp only [← c78573, ← c78574, k2, ← c78575] at e_1276_7
  have hr := e_1276_7
  simp only [e_1276_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11649)) := by
  have c78576 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78577 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78578 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78579 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78580 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78581 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_8 := arithEq_of_rows h (row := 1276) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_8
  simp only [← c78576, ← c78577, ← c78578] at e_1276_8
  have e_1276_9 := arithEq_of_rows h (row := 1276) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_9
  simp only [← c78579, ← c78580, k2, ← c78581] at e_1276_9
  have hr := e_1276_9
  simp only [e_1276_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11650)) := by
  have c78582 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78583 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78584 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78585 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78586 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78587 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_10 := arithEq_of_rows h (row := 1276) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_10
  simp only [← c78582, ← c78583, ← c78584] at e_1276_10
  have e_1276_11 := arithEq_of_rows h (row := 1276) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_11
  simp only [← c78585, ← c78586, k2, ← c78587] at e_1276_11
  have hr := e_1276_11
  simp only [e_1276_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11651)) := by
  have c78588 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78589 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78590 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78591 := (publicBatchWrapper89_copies2455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78592 := (publicBatchWrapper89_copies2456 a h).1
  have c78593 := (publicBatchWrapper89_copies2456 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_12 := arithEq_of_rows h (row := 1276) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_12
  simp only [← c78588, ← c78589, ← c78590] at e_1276_12
  have e_1276_13 := arithEq_of_rows h (row := 1276) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_13
  simp only [← c78591, ← c78592, k2, ← c78593] at e_1276_13
  have hr := e_1276_13
  simp only [e_1276_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11652)) := by
  have c78594 := (publicBatchWrapper89_copies2456 a h).2.2.1
  have c78595 := (publicBatchWrapper89_copies2456 a h).2.2.2.1
  have c78596 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.1
  have c78597 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.1
  have c78598 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.1
  have c78599 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_14 := arithEq_of_rows h (row := 1276) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_14
  simp only [← c78594, ← c78595, ← c78596] at e_1276_14
  have e_1276_15 := arithEq_of_rows h (row := 1276) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_15
  simp only [← c78597, ← c78598, k2, ← c78599] at e_1276_15
  have hr := e_1276_15
  simp only [e_1276_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11653)) := by
  have c78600 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.1
  have c78601 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.1
  have c78602 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78603 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78604 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78605 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_16 := arithEq_of_rows h (row := 1276) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_16
  simp only [← c78600, ← c78601, ← c78602] at e_1276_16
  have e_1276_17 := arithEq_of_rows h (row := 1276) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_17
  simp only [← c78603, ← c78604, k2, ← c78605] at e_1276_17
  have hr := e_1276_17
  simp only [e_1276_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1276 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11654)) := by
  have c78606 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78607 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78608 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78609 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78610 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78611 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1276_18 := arithEq_of_rows h (row := 1276) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_18
  simp only [← c78606, ← c78607, ← c78608] at e_1276_18
  have e_1276_19 := arithEq_of_rows h (row := 1276) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1276_19
  simp only [← c78609, ← c78610, k2, ← c78611] at e_1276_19
  have hr := e_1276_19
  simp only [e_1276_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11655)) := by
  have c78612 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78613 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78614 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78615 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78616 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78617 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_0 := arithEq_of_rows h (row := 1277) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_0
  simp only [← c78612, ← c78613, ← c78614] at e_1277_0
  have e_1277_1 := arithEq_of_rows h (row := 1277) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_1
  simp only [← c78615, ← c78616, k2, ← c78617] at e_1277_1
  have hr := e_1277_1
  simp only [e_1277_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11656)) := by
  have c78618 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78619 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78620 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78621 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78622 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78623 := (publicBatchWrapper89_copies2456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_2 := arithEq_of_rows h (row := 1277) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_2
  simp only [← c78618, ← c78619, ← c78620] at e_1277_2
  have e_1277_3 := arithEq_of_rows h (row := 1277) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_3
  simp only [← c78621, ← c78622, k2, ← c78623] at e_1277_3
  have hr := e_1277_3
  simp only [e_1277_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11657)) := by
  have c78624 := (publicBatchWrapper89_copies2457 a h).1
  have c78625 := (publicBatchWrapper89_copies2457 a h).2.1
  have c78626 := (publicBatchWrapper89_copies2457 a h).2.2.1
  have c78627 := (publicBatchWrapper89_copies2457 a h).2.2.2.1
  have c78628 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.1
  have c78629 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_4 := arithEq_of_rows h (row := 1277) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_4
  simp only [← c78624, ← c78625, ← c78626] at e_1277_4
  have e_1277_5 := arithEq_of_rows h (row := 1277) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_5
  simp only [← c78627, ← c78628, k2, ← c78629] at e_1277_5
  have hr := e_1277_5
  simp only [e_1277_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11658)) := by
  have c78630 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.1
  have c78631 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.1
  have c78632 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.1
  have c78633 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.1
  have c78634 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78635 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_6 := arithEq_of_rows h (row := 1277) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_6
  simp only [← c78630, ← c78631, ← c78632] at e_1277_6
  have e_1277_7 := arithEq_of_rows h (row := 1277) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_7
  simp only [← c78633, ← c78634, k2, ← c78635] at e_1277_7
  have hr := e_1277_7
  simp only [e_1277_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11659)) := by
  have c78636 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78637 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78638 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78639 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78640 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78641 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_8 := arithEq_of_rows h (row := 1277) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_8
  simp only [← c78636, ← c78637, ← c78638] at e_1277_8
  have e_1277_9 := arithEq_of_rows h (row := 1277) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_9
  simp only [← c78639, ← c78640, k2, ← c78641] at e_1277_9
  have hr := e_1277_9
  simp only [e_1277_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11660)) := by
  have c78642 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78643 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78644 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78645 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78646 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78647 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_10 := arithEq_of_rows h (row := 1277) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_10
  simp only [← c78642, ← c78643, ← c78644] at e_1277_10
  have e_1277_11 := arithEq_of_rows h (row := 1277) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_11
  simp only [← c78645, ← c78646, k2, ← c78647] at e_1277_11
  have hr := e_1277_11
  simp only [e_1277_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11661)) := by
  have c78648 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78649 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78650 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78651 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78652 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78653 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_12 := arithEq_of_rows h (row := 1277) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_12
  simp only [← c78648, ← c78649, ← c78650] at e_1277_12
  have e_1277_13 := arithEq_of_rows h (row := 1277) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_13
  simp only [← c78651, ← c78652, k2, ← c78653] at e_1277_13
  have hr := e_1277_13
  simp only [e_1277_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11662)) := by
  have c78654 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78655 := (publicBatchWrapper89_copies2457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78656 := (publicBatchWrapper89_copies2458 a h).1
  have c78657 := (publicBatchWrapper89_copies2458 a h).2.1
  have c78658 := (publicBatchWrapper89_copies2458 a h).2.2.1
  have c78659 := (publicBatchWrapper89_copies2458 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_14 := arithEq_of_rows h (row := 1277) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_14
  simp only [← c78654, ← c78655, ← c78656] at e_1277_14
  have e_1277_15 := arithEq_of_rows h (row := 1277) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_15
  simp only [← c78657, ← c78658, k2, ← c78659] at e_1277_15
  have hr := e_1277_15
  simp only [e_1277_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11663)) := by
  have c78660 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.1
  have c78661 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.1
  have c78662 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.1
  have c78663 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.1
  have c78664 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.1
  have c78665 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_16 := arithEq_of_rows h (row := 1277) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_16
  simp only [← c78660, ← c78661, ← c78662] at e_1277_16
  have e_1277_17 := arithEq_of_rows h (row := 1277) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_17
  simp only [← c78663, ← c78664, k2, ← c78665] at e_1277_17
  have hr := e_1277_17
  simp only [e_1277_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1277 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11664)) := by
  have c78666 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78667 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78668 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78669 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78670 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78671 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1277_18 := arithEq_of_rows h (row := 1277) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_18
  simp only [← c78666, ← c78667, ← c78668] at e_1277_18
  have e_1277_19 := arithEq_of_rows h (row := 1277) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1277_19
  simp only [← c78669, ← c78670, k2, ← c78671] at e_1277_19
  have hr := e_1277_19
  simp only [e_1277_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11665)) := by
  have c78672 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78673 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78674 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78675 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78676 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78677 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_0 := arithEq_of_rows h (row := 1278) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_0
  simp only [← c78672, ← c78673, ← c78674] at e_1278_0
  have e_1278_1 := arithEq_of_rows h (row := 1278) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_1
  simp only [← c78675, ← c78676, k2, ← c78677] at e_1278_1
  have hr := e_1278_1
  simp only [e_1278_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11666)) := by
  have c78678 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78679 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78680 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78681 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78682 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78683 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_2 := arithEq_of_rows h (row := 1278) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_2
  simp only [← c78678, ← c78679, ← c78680] at e_1278_2
  have e_1278_3 := arithEq_of_rows h (row := 1278) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_3
  simp only [← c78681, ← c78682, k2, ← c78683] at e_1278_3
  have hr := e_1278_3
  simp only [e_1278_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11667)) := by
  have c78684 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78685 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78686 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78687 := (publicBatchWrapper89_copies2458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78688 := (publicBatchWrapper89_copies2459 a h).1
  have c78689 := (publicBatchWrapper89_copies2459 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_4 := arithEq_of_rows h (row := 1278) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_4
  simp only [← c78684, ← c78685, ← c78686] at e_1278_4
  have e_1278_5 := arithEq_of_rows h (row := 1278) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_5
  simp only [← c78687, ← c78688, k2, ← c78689] at e_1278_5
  have hr := e_1278_5
  simp only [e_1278_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11668)) := by
  have c78690 := (publicBatchWrapper89_copies2459 a h).2.2.1
  have c78691 := (publicBatchWrapper89_copies2459 a h).2.2.2.1
  have c78692 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.1
  have c78693 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.1
  have c78694 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.1
  have c78695 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_6 := arithEq_of_rows h (row := 1278) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_6
  simp only [← c78690, ← c78691, ← c78692] at e_1278_6
  have e_1278_7 := arithEq_of_rows h (row := 1278) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_7
  simp only [← c78693, ← c78694, k2, ← c78695] at e_1278_7
  have hr := e_1278_7
  simp only [e_1278_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11669)) := by
  have c78696 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.1
  have c78697 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.1
  have c78698 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78699 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78700 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78701 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_8 := arithEq_of_rows h (row := 1278) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_8
  simp only [← c78696, ← c78697, ← c78698] at e_1278_8
  have e_1278_9 := arithEq_of_rows h (row := 1278) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_9
  simp only [← c78699, ← c78700, k2, ← c78701] at e_1278_9
  have hr := e_1278_9
  simp only [e_1278_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11670)) := by
  have c78702 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78703 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78704 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78705 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78706 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78707 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_10 := arithEq_of_rows h (row := 1278) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_10
  simp only [← c78702, ← c78703, ← c78704] at e_1278_10
  have e_1278_11 := arithEq_of_rows h (row := 1278) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_11
  simp only [← c78705, ← c78706, k2, ← c78707] at e_1278_11
  have hr := e_1278_11
  simp only [e_1278_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11671)) := by
  have c78708 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78709 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78710 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78711 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78712 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78713 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_12 := arithEq_of_rows h (row := 1278) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_12
  simp only [← c78708, ← c78709, ← c78710] at e_1278_12
  have e_1278_13 := arithEq_of_rows h (row := 1278) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_13
  simp only [← c78711, ← c78712, k2, ← c78713] at e_1278_13
  have hr := e_1278_13
  simp only [e_1278_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11672)) := by
  have c78714 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78715 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78716 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78717 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78718 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78719 := (publicBatchWrapper89_copies2459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_14 := arithEq_of_rows h (row := 1278) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_14
  simp only [← c78714, ← c78715, ← c78716] at e_1278_14
  have e_1278_15 := arithEq_of_rows h (row := 1278) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_15
  simp only [← c78717, ← c78718, k2, ← c78719] at e_1278_15
  have hr := e_1278_15
  simp only [e_1278_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11673)) := by
  have c78720 := (publicBatchWrapper89_copies2460 a h).1
  have c78721 := (publicBatchWrapper89_copies2460 a h).2.1
  have c78722 := (publicBatchWrapper89_copies2460 a h).2.2.1
  have c78723 := (publicBatchWrapper89_copies2460 a h).2.2.2.1
  have c78724 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.1
  have c78725 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_16 := arithEq_of_rows h (row := 1278) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_16
  simp only [← c78720, ← c78721, ← c78722] at e_1278_16
  have e_1278_17 := arithEq_of_rows h (row := 1278) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_17
  simp only [← c78723, ← c78724, k2, ← c78725] at e_1278_17
  have hr := e_1278_17
  simp only [e_1278_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1278 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11674)) := by
  have c78726 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.1
  have c78727 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.1
  have c78728 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.1
  have c78729 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.1
  have c78730 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78731 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1278_18 := arithEq_of_rows h (row := 1278) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_18
  simp only [← c78726, ← c78727, ← c78728] at e_1278_18
  have e_1278_19 := arithEq_of_rows h (row := 1278) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1278_19
  simp only [← c78729, ← c78730, k2, ← c78731] at e_1278_19
  have hr := e_1278_19
  simp only [e_1278_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11675)) := by
  have c78732 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78733 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78734 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78735 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78736 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78737 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_0 := arithEq_of_rows h (row := 1279) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_0
  simp only [← c78732, ← c78733, ← c78734] at e_1279_0
  have e_1279_1 := arithEq_of_rows h (row := 1279) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_1
  simp only [← c78735, ← c78736, k2, ← c78737] at e_1279_1
  have hr := e_1279_1
  simp only [e_1279_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11676)) := by
  have c78738 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78739 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78740 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78741 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78742 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78743 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_2 := arithEq_of_rows h (row := 1279) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_2
  simp only [← c78738, ← c78739, ← c78740] at e_1279_2
  have e_1279_3 := arithEq_of_rows h (row := 1279) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_3
  simp only [← c78741, ← c78742, k2, ← c78743] at e_1279_3
  have hr := e_1279_3
  simp only [e_1279_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11677)) := by
  have c78744 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78745 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78746 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78747 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78748 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78749 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_4 := arithEq_of_rows h (row := 1279) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_4
  simp only [← c78744, ← c78745, ← c78746] at e_1279_4
  have e_1279_5 := arithEq_of_rows h (row := 1279) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_5
  simp only [← c78747, ← c78748, k2, ← c78749] at e_1279_5
  have hr := e_1279_5
  simp only [e_1279_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11678)) := by
  have c78750 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78751 := (publicBatchWrapper89_copies2460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78752 := (publicBatchWrapper89_copies2461 a h).1
  have c78753 := (publicBatchWrapper89_copies2461 a h).2.1
  have c78754 := (publicBatchWrapper89_copies2461 a h).2.2.1
  have c78755 := (publicBatchWrapper89_copies2461 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_6 := arithEq_of_rows h (row := 1279) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_6
  simp only [← c78750, ← c78751, ← c78752] at e_1279_6
  have e_1279_7 := arithEq_of_rows h (row := 1279) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_7
  simp only [← c78753, ← c78754, k2, ← c78755] at e_1279_7
  have hr := e_1279_7
  simp only [e_1279_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11679)) := by
  have c78756 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.1
  have c78757 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.1
  have c78758 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.1
  have c78759 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.1
  have c78760 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.1
  have c78761 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_8 := arithEq_of_rows h (row := 1279) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_8
  simp only [← c78756, ← c78757, ← c78758] at e_1279_8
  have e_1279_9 := arithEq_of_rows h (row := 1279) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_9
  simp only [← c78759, ← c78760, k2, ← c78761] at e_1279_9
  have hr := e_1279_9
  simp only [e_1279_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11680)) := by
  have c78762 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78763 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78764 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78765 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78766 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78767 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_10 := arithEq_of_rows h (row := 1279) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_10
  simp only [← c78762, ← c78763, ← c78764] at e_1279_10
  have e_1279_11 := arithEq_of_rows h (row := 1279) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_11
  simp only [← c78765, ← c78766, k2, ← c78767] at e_1279_11
  have hr := e_1279_11
  simp only [e_1279_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11681)) := by
  have c78768 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78769 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78770 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78771 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78772 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78773 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_12 := arithEq_of_rows h (row := 1279) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_12
  simp only [← c78768, ← c78769, ← c78770] at e_1279_12
  have e_1279_13 := arithEq_of_rows h (row := 1279) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_13
  simp only [← c78771, ← c78772, k2, ← c78773] at e_1279_13
  have hr := e_1279_13
  simp only [e_1279_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11682)) := by
  have c78774 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78775 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78776 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78777 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78778 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78779 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_14 := arithEq_of_rows h (row := 1279) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_14
  simp only [← c78774, ← c78775, ← c78776] at e_1279_14
  have e_1279_15 := arithEq_of_rows h (row := 1279) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_15
  simp only [← c78777, ← c78778, k2, ← c78779] at e_1279_15
  have hr := e_1279_15
  simp only [e_1279_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11683)) := by
  have c78780 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78781 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78782 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78783 := (publicBatchWrapper89_copies2461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78784 := (publicBatchWrapper89_copies2462 a h).1
  have c78785 := (publicBatchWrapper89_copies2462 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_16 := arithEq_of_rows h (row := 1279) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_16
  simp only [← c78780, ← c78781, ← c78782] at e_1279_16
  have e_1279_17 := arithEq_of_rows h (row := 1279) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_17
  simp only [← c78783, ← c78784, k2, ← c78785] at e_1279_17
  have hr := e_1279_17
  simp only [e_1279_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1279 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11684)) := by
  have c78786 := (publicBatchWrapper89_copies2462 a h).2.2.1
  have c78787 := (publicBatchWrapper89_copies2462 a h).2.2.2.1
  have c78788 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.1
  have c78789 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.1
  have c78790 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.1
  have c78791 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1279_18 := arithEq_of_rows h (row := 1279) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_18
  simp only [← c78786, ← c78787, ← c78788] at e_1279_18
  have e_1279_19 := arithEq_of_rows h (row := 1279) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1279_19
  simp only [← c78789, ← c78790, k2, ← c78791] at e_1279_19
  have hr := e_1279_19
  simp only [e_1279_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11685)) := by
  have c78792 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.1
  have c78793 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.1
  have c78794 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78795 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78796 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78797 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_0 := arithEq_of_rows h (row := 1280) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_0
  simp only [← c78792, ← c78793, ← c78794] at e_1280_0
  have e_1280_1 := arithEq_of_rows h (row := 1280) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_1
  simp only [← c78795, ← c78796, k2, ← c78797] at e_1280_1
  have hr := e_1280_1
  simp only [e_1280_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11686)) := by
  have c78798 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78799 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78800 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78801 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78802 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78803 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_2 := arithEq_of_rows h (row := 1280) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_2
  simp only [← c78798, ← c78799, ← c78800] at e_1280_2
  have e_1280_3 := arithEq_of_rows h (row := 1280) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_3
  simp only [← c78801, ← c78802, k2, ← c78803] at e_1280_3
  have hr := e_1280_3
  simp only [e_1280_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11687)) := by
  have c78804 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78805 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78806 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78807 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78808 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78809 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_4 := arithEq_of_rows h (row := 1280) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_4
  simp only [← c78804, ← c78805, ← c78806] at e_1280_4
  have e_1280_5 := arithEq_of_rows h (row := 1280) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_5
  simp only [← c78807, ← c78808, k2, ← c78809] at e_1280_5
  have hr := e_1280_5
  simp only [e_1280_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11688)) := by
  have c78810 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78811 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78812 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78813 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78814 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78815 := (publicBatchWrapper89_copies2462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_6 := arithEq_of_rows h (row := 1280) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_6
  simp only [← c78810, ← c78811, ← c78812] at e_1280_6
  have e_1280_7 := arithEq_of_rows h (row := 1280) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_7
  simp only [← c78813, ← c78814, k2, ← c78815] at e_1280_7
  have hr := e_1280_7
  simp only [e_1280_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11689)) := by
  have c78816 := (publicBatchWrapper89_copies2463 a h).1
  have c78817 := (publicBatchWrapper89_copies2463 a h).2.1
  have c78818 := (publicBatchWrapper89_copies2463 a h).2.2.1
  have c78819 := (publicBatchWrapper89_copies2463 a h).2.2.2.1
  have c78820 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.1
  have c78821 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_8 := arithEq_of_rows h (row := 1280) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_8
  simp only [← c78816, ← c78817, ← c78818] at e_1280_8
  have e_1280_9 := arithEq_of_rows h (row := 1280) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_9
  simp only [← c78819, ← c78820, k2, ← c78821] at e_1280_9
  have hr := e_1280_9
  simp only [e_1280_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11690)) := by
  have c78822 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.1
  have c78823 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.1
  have c78824 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.1
  have c78825 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.1
  have c78826 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78827 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_10 := arithEq_of_rows h (row := 1280) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_10
  simp only [← c78822, ← c78823, ← c78824] at e_1280_10
  have e_1280_11 := arithEq_of_rows h (row := 1280) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_11
  simp only [← c78825, ← c78826, k2, ← c78827] at e_1280_11
  have hr := e_1280_11
  simp only [e_1280_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11691)) := by
  have c78828 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78829 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78830 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78831 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78832 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78833 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_12 := arithEq_of_rows h (row := 1280) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_12
  simp only [← c78828, ← c78829, ← c78830] at e_1280_12
  have e_1280_13 := arithEq_of_rows h (row := 1280) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_13
  simp only [← c78831, ← c78832, k2, ← c78833] at e_1280_13
  have hr := e_1280_13
  simp only [e_1280_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11692)) := by
  have c78834 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78835 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78836 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78837 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78838 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78839 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_14 := arithEq_of_rows h (row := 1280) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_14
  simp only [← c78834, ← c78835, ← c78836] at e_1280_14
  have e_1280_15 := arithEq_of_rows h (row := 1280) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_15
  simp only [← c78837, ← c78838, k2, ← c78839] at e_1280_15
  have hr := e_1280_15
  simp only [e_1280_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11693)) := by
  have c78840 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78841 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78842 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78843 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78844 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78845 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_16 := arithEq_of_rows h (row := 1280) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_16
  simp only [← c78840, ← c78841, ← c78842] at e_1280_16
  have e_1280_17 := arithEq_of_rows h (row := 1280) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_17
  simp only [← c78843, ← c78844, k2, ← c78845] at e_1280_17
  have hr := e_1280_17
  simp only [e_1280_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1280 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11694)) := by
  have c78846 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78847 := (publicBatchWrapper89_copies2463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78848 := (publicBatchWrapper89_copies2464 a h).1
  have c78849 := (publicBatchWrapper89_copies2464 a h).2.1
  have c78850 := (publicBatchWrapper89_copies2464 a h).2.2.1
  have c78851 := (publicBatchWrapper89_copies2464 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1280_18 := arithEq_of_rows h (row := 1280) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_18
  simp only [← c78846, ← c78847, ← c78848] at e_1280_18
  have e_1280_19 := arithEq_of_rows h (row := 1280) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1280_19
  simp only [← c78849, ← c78850, k2, ← c78851] at e_1280_19
  have hr := e_1280_19
  simp only [e_1280_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11695)) := by
  have c78852 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.1
  have c78853 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.1
  have c78854 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.1
  have c78855 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.1
  have c78856 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.1
  have c78857 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_0 := arithEq_of_rows h (row := 1281) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_0
  simp only [← c78852, ← c78853, ← c78854] at e_1281_0
  have e_1281_1 := arithEq_of_rows h (row := 1281) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_1
  simp only [← c78855, ← c78856, k2, ← c78857] at e_1281_1
  have hr := e_1281_1
  simp only [e_1281_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11696)) := by
  have c78858 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78859 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78860 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78861 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78862 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78863 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_2 := arithEq_of_rows h (row := 1281) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_2
  simp only [← c78858, ← c78859, ← c78860] at e_1281_2
  have e_1281_3 := arithEq_of_rows h (row := 1281) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_3
  simp only [← c78861, ← c78862, k2, ← c78863] at e_1281_3
  have hr := e_1281_3
  simp only [e_1281_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11697)) := by
  have c78864 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78865 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78866 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78867 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78868 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78869 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_4 := arithEq_of_rows h (row := 1281) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_4
  simp only [← c78864, ← c78865, ← c78866] at e_1281_4
  have e_1281_5 := arithEq_of_rows h (row := 1281) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_5
  simp only [← c78867, ← c78868, k2, ← c78869] at e_1281_5
  have hr := e_1281_5
  simp only [e_1281_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11698)) := by
  have c78870 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78871 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78872 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78873 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78874 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78875 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_6 := arithEq_of_rows h (row := 1281) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_6
  simp only [← c78870, ← c78871, ← c78872] at e_1281_6
  have e_1281_7 := arithEq_of_rows h (row := 1281) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_7
  simp only [← c78873, ← c78874, k2, ← c78875] at e_1281_7
  have hr := e_1281_7
  simp only [e_1281_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11699)) := by
  have c78876 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78877 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78878 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78879 := (publicBatchWrapper89_copies2464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78880 := (publicBatchWrapper89_copies2465 a h).1
  have c78881 := (publicBatchWrapper89_copies2465 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_8 := arithEq_of_rows h (row := 1281) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_8
  simp only [← c78876, ← c78877, ← c78878] at e_1281_8
  have e_1281_9 := arithEq_of_rows h (row := 1281) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_9
  simp only [← c78879, ← c78880, k2, ← c78881] at e_1281_9
  have hr := e_1281_9
  simp only [e_1281_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11700)) := by
  have c78882 := (publicBatchWrapper89_copies2465 a h).2.2.1
  have c78883 := (publicBatchWrapper89_copies2465 a h).2.2.2.1
  have c78884 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.1
  have c78885 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.1
  have c78886 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.1
  have c78887 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_10 := arithEq_of_rows h (row := 1281) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_10
  simp only [← c78882, ← c78883, ← c78884] at e_1281_10
  have e_1281_11 := arithEq_of_rows h (row := 1281) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_11
  simp only [← c78885, ← c78886, k2, ← c78887] at e_1281_11
  have hr := e_1281_11
  simp only [e_1281_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11701)) := by
  have c78888 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.1
  have c78889 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.1
  have c78890 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78891 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78892 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78893 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_12 := arithEq_of_rows h (row := 1281) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_12
  simp only [← c78888, ← c78889, ← c78890] at e_1281_12
  have e_1281_13 := arithEq_of_rows h (row := 1281) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_13
  simp only [← c78891, ← c78892, k2, ← c78893] at e_1281_13
  have hr := e_1281_13
  simp only [e_1281_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11702)) := by
  have c78894 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78895 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78896 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78897 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78898 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78899 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_14 := arithEq_of_rows h (row := 1281) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_14
  simp only [← c78894, ← c78895, ← c78896] at e_1281_14
  have e_1281_15 := arithEq_of_rows h (row := 1281) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_15
  simp only [← c78897, ← c78898, k2, ← c78899] at e_1281_15
  have hr := e_1281_15
  simp only [e_1281_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11703)) := by
  have c78900 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78901 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78902 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78903 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78904 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78905 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_16 := arithEq_of_rows h (row := 1281) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_16
  simp only [← c78900, ← c78901, ← c78902] at e_1281_16
  have e_1281_17 := arithEq_of_rows h (row := 1281) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_17
  simp only [← c78903, ← c78904, k2, ← c78905] at e_1281_17
  have hr := e_1281_17
  simp only [e_1281_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1281 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11704)) := by
  have c78906 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78907 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78908 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78909 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78910 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78911 := (publicBatchWrapper89_copies2465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1281_18 := arithEq_of_rows h (row := 1281) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_18
  simp only [← c78906, ← c78907, ← c78908] at e_1281_18
  have e_1281_19 := arithEq_of_rows h (row := 1281) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1281_19
  simp only [← c78909, ← c78910, k2, ← c78911] at e_1281_19
  have hr := e_1281_19
  simp only [e_1281_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11705)) := by
  have c78912 := (publicBatchWrapper89_copies2466 a h).1
  have c78913 := (publicBatchWrapper89_copies2466 a h).2.1
  have c78914 := (publicBatchWrapper89_copies2466 a h).2.2.1
  have c78915 := (publicBatchWrapper89_copies2466 a h).2.2.2.1
  have c78916 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.1
  have c78917 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_0 := arithEq_of_rows h (row := 1282) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_0
  simp only [← c78912, ← c78913, ← c78914] at e_1282_0
  have e_1282_1 := arithEq_of_rows h (row := 1282) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_1
  simp only [← c78915, ← c78916, k2, ← c78917] at e_1282_1
  have hr := e_1282_1
  simp only [e_1282_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11706)) := by
  have c78918 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.1
  have c78919 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.1
  have c78920 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.1
  have c78921 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.1
  have c78922 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78923 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_2 := arithEq_of_rows h (row := 1282) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_2
  simp only [← c78918, ← c78919, ← c78920] at e_1282_2
  have e_1282_3 := arithEq_of_rows h (row := 1282) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_3
  simp only [← c78921, ← c78922, k2, ← c78923] at e_1282_3
  have hr := e_1282_3
  simp only [e_1282_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11707)) := by
  have c78924 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78925 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78926 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78927 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78928 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78929 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_4 := arithEq_of_rows h (row := 1282) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_4
  simp only [← c78924, ← c78925, ← c78926] at e_1282_4
  have e_1282_5 := arithEq_of_rows h (row := 1282) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_5
  simp only [← c78927, ← c78928, k2, ← c78929] at e_1282_5
  have hr := e_1282_5
  simp only [e_1282_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11708)) := by
  have c78930 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78931 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78932 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78933 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78934 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78935 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_6 := arithEq_of_rows h (row := 1282) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_6
  simp only [← c78930, ← c78931, ← c78932] at e_1282_6
  have e_1282_7 := arithEq_of_rows h (row := 1282) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_7
  simp only [← c78933, ← c78934, k2, ← c78935] at e_1282_7
  have hr := e_1282_7
  simp only [e_1282_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11709)) := by
  have c78936 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78937 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78938 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78939 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78940 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78941 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_8 := arithEq_of_rows h (row := 1282) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_8
  simp only [← c78936, ← c78937, ← c78938] at e_1282_8
  have e_1282_9 := arithEq_of_rows h (row := 1282) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_9
  simp only [← c78939, ← c78940, k2, ← c78941] at e_1282_9
  have hr := e_1282_9
  simp only [e_1282_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11710)) := by
  have c78942 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78943 := (publicBatchWrapper89_copies2466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78944 := (publicBatchWrapper89_copies2467 a h).1
  have c78945 := (publicBatchWrapper89_copies2467 a h).2.1
  have c78946 := (publicBatchWrapper89_copies2467 a h).2.2.1
  have c78947 := (publicBatchWrapper89_copies2467 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_10 := arithEq_of_rows h (row := 1282) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_10
  simp only [← c78942, ← c78943, ← c78944] at e_1282_10
  have e_1282_11 := arithEq_of_rows h (row := 1282) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_11
  simp only [← c78945, ← c78946, k2, ← c78947] at e_1282_11
  have hr := e_1282_11
  simp only [e_1282_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11711)) := by
  have c78948 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.1
  have c78949 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.1
  have c78950 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.1
  have c78951 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.1
  have c78952 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.1
  have c78953 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_12 := arithEq_of_rows h (row := 1282) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_12
  simp only [← c78948, ← c78949, ← c78950] at e_1282_12
  have e_1282_13 := arithEq_of_rows h (row := 1282) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_13
  simp only [← c78951, ← c78952, k2, ← c78953] at e_1282_13
  have hr := e_1282_13
  simp only [e_1282_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11712)) := by
  have c78954 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78955 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78956 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78957 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78958 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78959 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_14 := arithEq_of_rows h (row := 1282) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_14
  simp only [← c78954, ← c78955, ← c78956] at e_1282_14
  have e_1282_15 := arithEq_of_rows h (row := 1282) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_15
  simp only [← c78957, ← c78958, k2, ← c78959] at e_1282_15
  have hr := e_1282_15
  simp only [e_1282_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11713)) := by
  have c78960 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78961 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78962 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78963 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78964 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78965 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_16 := arithEq_of_rows h (row := 1282) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_16
  simp only [← c78960, ← c78961, ← c78962] at e_1282_16
  have e_1282_17 := arithEq_of_rows h (row := 1282) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_17
  simp only [← c78963, ← c78964, k2, ← c78965] at e_1282_17
  have hr := e_1282_17
  simp only [e_1282_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1282 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11714)) := by
  have c78966 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78967 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78968 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78969 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78970 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78971 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1282_18 := arithEq_of_rows h (row := 1282) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_18
  simp only [← c78966, ← c78967, ← c78968] at e_1282_18
  have e_1282_19 := arithEq_of_rows h (row := 1282) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1282_19
  simp only [← c78969, ← c78970, k2, ← c78971] at e_1282_19
  have hr := e_1282_19
  simp only [e_1282_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11715)) := by
  have c78972 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78973 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78974 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78975 := (publicBatchWrapper89_copies2467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c78976 := (publicBatchWrapper89_copies2468 a h).1
  have c78977 := (publicBatchWrapper89_copies2468 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_0 := arithEq_of_rows h (row := 1283) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_0
  simp only [← c78972, ← c78973, ← c78974] at e_1283_0
  have e_1283_1 := arithEq_of_rows h (row := 1283) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_1
  simp only [← c78975, ← c78976, k2, ← c78977] at e_1283_1
  have hr := e_1283_1
  simp only [e_1283_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11716)) := by
  have c78978 := (publicBatchWrapper89_copies2468 a h).2.2.1
  have c78979 := (publicBatchWrapper89_copies2468 a h).2.2.2.1
  have c78980 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.1
  have c78981 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.1
  have c78982 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.1
  have c78983 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_2 := arithEq_of_rows h (row := 1283) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_2
  simp only [← c78978, ← c78979, ← c78980] at e_1283_2
  have e_1283_3 := arithEq_of_rows h (row := 1283) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_3
  simp only [← c78981, ← c78982, k2, ← c78983] at e_1283_3
  have hr := e_1283_3
  simp only [e_1283_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11717)) := by
  have c78984 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.1
  have c78985 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.1
  have c78986 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c78987 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c78988 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78989 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_4 := arithEq_of_rows h (row := 1283) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_4
  simp only [← c78984, ← c78985, ← c78986] at e_1283_4
  have e_1283_5 := arithEq_of_rows h (row := 1283) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_5
  simp only [← c78987, ← c78988, k2, ← c78989] at e_1283_5
  have hr := e_1283_5
  simp only [e_1283_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11718)) := by
  have c78990 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78991 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78992 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78993 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78994 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78995 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_6 := arithEq_of_rows h (row := 1283) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_6
  simp only [← c78990, ← c78991, ← c78992] at e_1283_6
  have e_1283_7 := arithEq_of_rows h (row := 1283) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_7
  simp only [← c78993, ← c78994, k2, ← c78995] at e_1283_7
  have hr := e_1283_7
  simp only [e_1283_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11719)) := by
  have c78996 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78997 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78998 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c78999 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79000 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79001 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_8 := arithEq_of_rows h (row := 1283) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_8
  simp only [← c78996, ← c78997, ← c78998] at e_1283_8
  have e_1283_9 := arithEq_of_rows h (row := 1283) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_9
  simp only [← c78999, ← c79000, k2, ← c79001] at e_1283_9
  have hr := e_1283_9
  simp only [e_1283_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11720)) := by
  have c79002 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79003 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79004 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79005 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79006 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79007 := (publicBatchWrapper89_copies2468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_10 := arithEq_of_rows h (row := 1283) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_10
  simp only [← c79002, ← c79003, ← c79004] at e_1283_10
  have e_1283_11 := arithEq_of_rows h (row := 1283) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_11
  simp only [← c79005, ← c79006, k2, ← c79007] at e_1283_11
  have hr := e_1283_11
  simp only [e_1283_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11721)) := by
  have c79008 := (publicBatchWrapper89_copies2469 a h).1
  have c79009 := (publicBatchWrapper89_copies2469 a h).2.1
  have c79010 := (publicBatchWrapper89_copies2469 a h).2.2.1
  have c79011 := (publicBatchWrapper89_copies2469 a h).2.2.2.1
  have c79012 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.1
  have c79013 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_12 := arithEq_of_rows h (row := 1283) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_12
  simp only [← c79008, ← c79009, ← c79010] at e_1283_12
  have e_1283_13 := arithEq_of_rows h (row := 1283) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_13
  simp only [← c79011, ← c79012, k2, ← c79013] at e_1283_13
  have hr := e_1283_13
  simp only [e_1283_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11722)) := by
  have c79014 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.1
  have c79015 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.1
  have c79016 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.1
  have c79017 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.1
  have c79018 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79019 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_14 := arithEq_of_rows h (row := 1283) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_14
  simp only [← c79014, ← c79015, ← c79016] at e_1283_14
  have e_1283_15 := arithEq_of_rows h (row := 1283) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_15
  simp only [← c79017, ← c79018, k2, ← c79019] at e_1283_15
  have hr := e_1283_15
  simp only [e_1283_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11723)) := by
  have c79020 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79021 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79022 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79023 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79024 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79025 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_16 := arithEq_of_rows h (row := 1283) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_16
  simp only [← c79020, ← c79021, ← c79022] at e_1283_16
  have e_1283_17 := arithEq_of_rows h (row := 1283) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_17
  simp only [← c79023, ← c79024, k2, ← c79025] at e_1283_17
  have hr := e_1283_17
  simp only [e_1283_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1283 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11724)) := by
  have c79026 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79027 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79028 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79029 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79030 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79031 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1283_18 := arithEq_of_rows h (row := 1283) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_18
  simp only [← c79026, ← c79027, ← c79028] at e_1283_18
  have e_1283_19 := arithEq_of_rows h (row := 1283) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1283_19
  simp only [← c79029, ← c79030, k2, ← c79031] at e_1283_19
  have hr := e_1283_19
  simp only [e_1283_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11725)) := by
  have c79032 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79033 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79034 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79035 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79036 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79037 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_0 := arithEq_of_rows h (row := 1284) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_0
  simp only [← c79032, ← c79033, ← c79034] at e_1284_0
  have e_1284_1 := arithEq_of_rows h (row := 1284) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_1
  simp only [← c79035, ← c79036, k2, ← c79037] at e_1284_1
  have hr := e_1284_1
  simp only [e_1284_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11726)) := by
  have c79038 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79039 := (publicBatchWrapper89_copies2469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79040 := (publicBatchWrapper89_copies2470 a h).1
  have c79041 := (publicBatchWrapper89_copies2470 a h).2.1
  have c79042 := (publicBatchWrapper89_copies2470 a h).2.2.1
  have c79043 := (publicBatchWrapper89_copies2470 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_2 := arithEq_of_rows h (row := 1284) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_2
  simp only [← c79038, ← c79039, ← c79040] at e_1284_2
  have e_1284_3 := arithEq_of_rows h (row := 1284) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_3
  simp only [← c79041, ← c79042, k2, ← c79043] at e_1284_3
  have hr := e_1284_3
  simp only [e_1284_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11727)) := by
  have c79044 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.1
  have c79045 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.1
  have c79046 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.1
  have c79047 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.1
  have c79048 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.1
  have c79049 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_4 := arithEq_of_rows h (row := 1284) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_4
  simp only [← c79044, ← c79045, ← c79046] at e_1284_4
  have e_1284_5 := arithEq_of_rows h (row := 1284) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_5
  simp only [← c79047, ← c79048, k2, ← c79049] at e_1284_5
  have hr := e_1284_5
  simp only [e_1284_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11728)) := by
  have c79050 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79051 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79052 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79053 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79054 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79055 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_6 := arithEq_of_rows h (row := 1284) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_6
  simp only [← c79050, ← c79051, ← c79052] at e_1284_6
  have e_1284_7 := arithEq_of_rows h (row := 1284) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_7
  simp only [← c79053, ← c79054, k2, ← c79055] at e_1284_7
  have hr := e_1284_7
  simp only [e_1284_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11729)) := by
  have c79056 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79057 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79058 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79059 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79060 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79061 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_8 := arithEq_of_rows h (row := 1284) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_8
  simp only [← c79056, ← c79057, ← c79058] at e_1284_8
  have e_1284_9 := arithEq_of_rows h (row := 1284) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_9
  simp only [← c79059, ← c79060, k2, ← c79061] at e_1284_9
  have hr := e_1284_9
  simp only [e_1284_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11730)) := by
  have c79062 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79063 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79064 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79065 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79066 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79067 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_10 := arithEq_of_rows h (row := 1284) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_10
  simp only [← c79062, ← c79063, ← c79064] at e_1284_10
  have e_1284_11 := arithEq_of_rows h (row := 1284) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_11
  simp only [← c79065, ← c79066, k2, ← c79067] at e_1284_11
  have hr := e_1284_11
  simp only [e_1284_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11731)) := by
  have c79068 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79069 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79070 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79071 := (publicBatchWrapper89_copies2470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79072 := (publicBatchWrapper89_copies2471 a h).1
  have c79073 := (publicBatchWrapper89_copies2471 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_12 := arithEq_of_rows h (row := 1284) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_12
  simp only [← c79068, ← c79069, ← c79070] at e_1284_12
  have e_1284_13 := arithEq_of_rows h (row := 1284) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_13
  simp only [← c79071, ← c79072, k2, ← c79073] at e_1284_13
  have hr := e_1284_13
  simp only [e_1284_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11732)) := by
  have c79074 := (publicBatchWrapper89_copies2471 a h).2.2.1
  have c79075 := (publicBatchWrapper89_copies2471 a h).2.2.2.1
  have c79076 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.1
  have c79077 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.1
  have c79078 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.1
  have c79079 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_14 := arithEq_of_rows h (row := 1284) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_14
  simp only [← c79074, ← c79075, ← c79076] at e_1284_14
  have e_1284_15 := arithEq_of_rows h (row := 1284) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_15
  simp only [← c79077, ← c79078, k2, ← c79079] at e_1284_15
  have hr := e_1284_15
  simp only [e_1284_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11733)) := by
  have c79080 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.1
  have c79081 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.1
  have c79082 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79083 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79084 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79085 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_16 := arithEq_of_rows h (row := 1284) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_16
  simp only [← c79080, ← c79081, ← c79082] at e_1284_16
  have e_1284_17 := arithEq_of_rows h (row := 1284) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_17
  simp only [← c79083, ← c79084, k2, ← c79085] at e_1284_17
  have hr := e_1284_17
  simp only [e_1284_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1284 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11734)) := by
  have c79086 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79087 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79088 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79089 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79090 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79091 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1284_18 := arithEq_of_rows h (row := 1284) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_18
  simp only [← c79086, ← c79087, ← c79088] at e_1284_18
  have e_1284_19 := arithEq_of_rows h (row := 1284) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1284_19
  simp only [← c79089, ← c79090, k2, ← c79091] at e_1284_19
  have hr := e_1284_19
  simp only [e_1284_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11735)) := by
  have c79092 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79093 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79094 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79095 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79096 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79097 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_0 := arithEq_of_rows h (row := 1285) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_0
  simp only [← c79092, ← c79093, ← c79094] at e_1285_0
  have e_1285_1 := arithEq_of_rows h (row := 1285) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_1
  simp only [← c79095, ← c79096, k2, ← c79097] at e_1285_1
  have hr := e_1285_1
  simp only [e_1285_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11736)) := by
  have c79098 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79099 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79100 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79101 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79102 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79103 := (publicBatchWrapper89_copies2471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_2 := arithEq_of_rows h (row := 1285) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_2
  simp only [← c79098, ← c79099, ← c79100] at e_1285_2
  have e_1285_3 := arithEq_of_rows h (row := 1285) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_3
  simp only [← c79101, ← c79102, k2, ← c79103] at e_1285_3
  have hr := e_1285_3
  simp only [e_1285_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11737)) := by
  have c79104 := (publicBatchWrapper89_copies2472 a h).1
  have c79105 := (publicBatchWrapper89_copies2472 a h).2.1
  have c79106 := (publicBatchWrapper89_copies2472 a h).2.2.1
  have c79107 := (publicBatchWrapper89_copies2472 a h).2.2.2.1
  have c79108 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.1
  have c79109 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_4 := arithEq_of_rows h (row := 1285) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_4
  simp only [← c79104, ← c79105, ← c79106] at e_1285_4
  have e_1285_5 := arithEq_of_rows h (row := 1285) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_5
  simp only [← c79107, ← c79108, k2, ← c79109] at e_1285_5
  have hr := e_1285_5
  simp only [e_1285_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11738)) := by
  have c79110 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.1
  have c79111 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.1
  have c79112 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.1
  have c79113 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.1
  have c79114 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79115 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_6 := arithEq_of_rows h (row := 1285) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_6
  simp only [← c79110, ← c79111, ← c79112] at e_1285_6
  have e_1285_7 := arithEq_of_rows h (row := 1285) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_7
  simp only [← c79113, ← c79114, k2, ← c79115] at e_1285_7
  have hr := e_1285_7
  simp only [e_1285_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11739)) := by
  have c79116 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79117 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79118 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79119 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79120 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79121 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_8 := arithEq_of_rows h (row := 1285) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_8
  simp only [← c79116, ← c79117, ← c79118] at e_1285_8
  have e_1285_9 := arithEq_of_rows h (row := 1285) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_9
  simp only [← c79119, ← c79120, k2, ← c79121] at e_1285_9
  have hr := e_1285_9
  simp only [e_1285_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11740)) := by
  have c79122 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79123 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79124 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79125 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79126 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79127 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_10 := arithEq_of_rows h (row := 1285) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_10
  simp only [← c79122, ← c79123, ← c79124] at e_1285_10
  have e_1285_11 := arithEq_of_rows h (row := 1285) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_11
  simp only [← c79125, ← c79126, k2, ← c79127] at e_1285_11
  have hr := e_1285_11
  simp only [e_1285_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11741)) := by
  have c79128 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79129 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79130 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79131 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79132 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79133 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_12 := arithEq_of_rows h (row := 1285) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_12
  simp only [← c79128, ← c79129, ← c79130] at e_1285_12
  have e_1285_13 := arithEq_of_rows h (row := 1285) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_13
  simp only [← c79131, ← c79132, k2, ← c79133] at e_1285_13
  have hr := e_1285_13
  simp only [e_1285_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11742)) := by
  have c79134 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79135 := (publicBatchWrapper89_copies2472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79136 := (publicBatchWrapper89_copies2473 a h).1
  have c79137 := (publicBatchWrapper89_copies2473 a h).2.1
  have c79138 := (publicBatchWrapper89_copies2473 a h).2.2.1
  have c79139 := (publicBatchWrapper89_copies2473 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_14 := arithEq_of_rows h (row := 1285) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_14
  simp only [← c79134, ← c79135, ← c79136] at e_1285_14
  have e_1285_15 := arithEq_of_rows h (row := 1285) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_15
  simp only [← c79137, ← c79138, k2, ← c79139] at e_1285_15
  have hr := e_1285_15
  simp only [e_1285_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11743)) := by
  have c79140 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.1
  have c79141 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.1
  have c79142 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.1
  have c79143 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.1
  have c79144 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.1
  have c79145 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_16 := arithEq_of_rows h (row := 1285) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_16
  simp only [← c79140, ← c79141, ← c79142] at e_1285_16
  have e_1285_17 := arithEq_of_rows h (row := 1285) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_17
  simp only [← c79143, ← c79144, k2, ← c79145] at e_1285_17
  have hr := e_1285_17
  simp only [e_1285_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1285 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11744)) := by
  have c79146 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79147 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79148 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79149 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79150 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79151 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1285_18 := arithEq_of_rows h (row := 1285) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_18
  simp only [← c79146, ← c79147, ← c79148] at e_1285_18
  have e_1285_19 := arithEq_of_rows h (row := 1285) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1285_19
  simp only [← c79149, ← c79150, k2, ← c79151] at e_1285_19
  have hr := e_1285_19
  simp only [e_1285_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11745)) := by
  have c79152 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79153 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79154 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79155 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79156 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79157 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_0 := arithEq_of_rows h (row := 1286) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_0
  simp only [← c79152, ← c79153, ← c79154] at e_1286_0
  have e_1286_1 := arithEq_of_rows h (row := 1286) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_1
  simp only [← c79155, ← c79156, k2, ← c79157] at e_1286_1
  have hr := e_1286_1
  simp only [e_1286_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11746)) := by
  have c79158 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79159 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79160 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79161 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79162 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79163 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_2 := arithEq_of_rows h (row := 1286) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_2
  simp only [← c79158, ← c79159, ← c79160] at e_1286_2
  have e_1286_3 := arithEq_of_rows h (row := 1286) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_3
  simp only [← c79161, ← c79162, k2, ← c79163] at e_1286_3
  have hr := e_1286_3
  simp only [e_1286_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11747)) := by
  have c79164 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79165 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79166 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79167 := (publicBatchWrapper89_copies2473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79168 := (publicBatchWrapper89_copies2474 a h).1
  have c79169 := (publicBatchWrapper89_copies2474 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_4 := arithEq_of_rows h (row := 1286) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_4
  simp only [← c79164, ← c79165, ← c79166] at e_1286_4
  have e_1286_5 := arithEq_of_rows h (row := 1286) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_5
  simp only [← c79167, ← c79168, k2, ← c79169] at e_1286_5
  have hr := e_1286_5
  simp only [e_1286_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11748)) := by
  have c79170 := (publicBatchWrapper89_copies2474 a h).2.2.1
  have c79171 := (publicBatchWrapper89_copies2474 a h).2.2.2.1
  have c79172 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.1
  have c79173 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.1
  have c79174 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.1
  have c79175 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_6 := arithEq_of_rows h (row := 1286) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_6
  simp only [← c79170, ← c79171, ← c79172] at e_1286_6
  have e_1286_7 := arithEq_of_rows h (row := 1286) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_7
  simp only [← c79173, ← c79174, k2, ← c79175] at e_1286_7
  have hr := e_1286_7
  simp only [e_1286_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11749)) := by
  have c79176 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.1
  have c79177 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.1
  have c79178 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79179 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79180 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79181 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_8 := arithEq_of_rows h (row := 1286) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_8
  simp only [← c79176, ← c79177, ← c79178] at e_1286_8
  have e_1286_9 := arithEq_of_rows h (row := 1286) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_9
  simp only [← c79179, ← c79180, k2, ← c79181] at e_1286_9
  have hr := e_1286_9
  simp only [e_1286_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11750)) := by
  have c79182 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79183 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79184 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79185 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79186 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79187 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_10 := arithEq_of_rows h (row := 1286) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_10
  simp only [← c79182, ← c79183, ← c79184] at e_1286_10
  have e_1286_11 := arithEq_of_rows h (row := 1286) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_11
  simp only [← c79185, ← c79186, k2, ← c79187] at e_1286_11
  have hr := e_1286_11
  simp only [e_1286_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11751)) := by
  have c79188 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79189 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79190 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79191 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79192 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79193 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_12 := arithEq_of_rows h (row := 1286) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_12
  simp only [← c79188, ← c79189, ← c79190] at e_1286_12
  have e_1286_13 := arithEq_of_rows h (row := 1286) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_13
  simp only [← c79191, ← c79192, k2, ← c79193] at e_1286_13
  have hr := e_1286_13
  simp only [e_1286_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11752)) := by
  have c79194 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79195 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79196 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79197 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79198 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79199 := (publicBatchWrapper89_copies2474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_14 := arithEq_of_rows h (row := 1286) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_14
  simp only [← c79194, ← c79195, ← c79196] at e_1286_14
  have e_1286_15 := arithEq_of_rows h (row := 1286) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_15
  simp only [← c79197, ← c79198, k2, ← c79199] at e_1286_15
  have hr := e_1286_15
  simp only [e_1286_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11753)) := by
  have c79200 := (publicBatchWrapper89_copies2475 a h).1
  have c79201 := (publicBatchWrapper89_copies2475 a h).2.1
  have c79202 := (publicBatchWrapper89_copies2475 a h).2.2.1
  have c79203 := (publicBatchWrapper89_copies2475 a h).2.2.2.1
  have c79204 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.1
  have c79205 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_16 := arithEq_of_rows h (row := 1286) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_16
  simp only [← c79200, ← c79201, ← c79202] at e_1286_16
  have e_1286_17 := arithEq_of_rows h (row := 1286) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_17
  simp only [← c79203, ← c79204, k2, ← c79205] at e_1286_17
  have hr := e_1286_17
  simp only [e_1286_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1286 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11754)) := by
  have c79206 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.1
  have c79207 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.1
  have c79208 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.1
  have c79209 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.1
  have c79210 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79211 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1286_18 := arithEq_of_rows h (row := 1286) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_18
  simp only [← c79206, ← c79207, ← c79208] at e_1286_18
  have e_1286_19 := arithEq_of_rows h (row := 1286) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1286_19
  simp only [← c79209, ← c79210, k2, ← c79211] at e_1286_19
  have hr := e_1286_19
  simp only [e_1286_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11755)) := by
  have c79212 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79213 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79214 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79215 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79216 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79217 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_0 := arithEq_of_rows h (row := 1287) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_0
  simp only [← c79212, ← c79213, ← c79214] at e_1287_0
  have e_1287_1 := arithEq_of_rows h (row := 1287) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_1
  simp only [← c79215, ← c79216, k2, ← c79217] at e_1287_1
  have hr := e_1287_1
  simp only [e_1287_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11756)) := by
  have c79218 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79219 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79220 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79221 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79222 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79223 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_2 := arithEq_of_rows h (row := 1287) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_2
  simp only [← c79218, ← c79219, ← c79220] at e_1287_2
  have e_1287_3 := arithEq_of_rows h (row := 1287) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_3
  simp only [← c79221, ← c79222, k2, ← c79223] at e_1287_3
  have hr := e_1287_3
  simp only [e_1287_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11757)) := by
  have c79224 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79225 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79226 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79227 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79228 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79229 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_4 := arithEq_of_rows h (row := 1287) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_4
  simp only [← c79224, ← c79225, ← c79226] at e_1287_4
  have e_1287_5 := arithEq_of_rows h (row := 1287) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_5
  simp only [← c79227, ← c79228, k2, ← c79229] at e_1287_5
  have hr := e_1287_5
  simp only [e_1287_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11758)) := by
  have c79230 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79231 := (publicBatchWrapper89_copies2475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79232 := (publicBatchWrapper89_copies2476 a h).1
  have c79233 := (publicBatchWrapper89_copies2476 a h).2.1
  have c79234 := (publicBatchWrapper89_copies2476 a h).2.2.1
  have c79235 := (publicBatchWrapper89_copies2476 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_6 := arithEq_of_rows h (row := 1287) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_6
  simp only [← c79230, ← c79231, ← c79232] at e_1287_6
  have e_1287_7 := arithEq_of_rows h (row := 1287) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_7
  simp only [← c79233, ← c79234, k2, ← c79235] at e_1287_7
  have hr := e_1287_7
  simp only [e_1287_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11759)) := by
  have c79236 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.1
  have c79237 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.1
  have c79238 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.1
  have c79239 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.1
  have c79240 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.1
  have c79241 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_8 := arithEq_of_rows h (row := 1287) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_8
  simp only [← c79236, ← c79237, ← c79238] at e_1287_8
  have e_1287_9 := arithEq_of_rows h (row := 1287) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_9
  simp only [← c79239, ← c79240, k2, ← c79241] at e_1287_9
  have hr := e_1287_9
  simp only [e_1287_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11760)) := by
  have c79242 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79243 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79244 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79245 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79246 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79247 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_10 := arithEq_of_rows h (row := 1287) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_10
  simp only [← c79242, ← c79243, ← c79244] at e_1287_10
  have e_1287_11 := arithEq_of_rows h (row := 1287) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_11
  simp only [← c79245, ← c79246, k2, ← c79247] at e_1287_11
  have hr := e_1287_11
  simp only [e_1287_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11761)) := by
  have c79248 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79249 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79250 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79251 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79252 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79253 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_12 := arithEq_of_rows h (row := 1287) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_12
  simp only [← c79248, ← c79249, ← c79250] at e_1287_12
  have e_1287_13 := arithEq_of_rows h (row := 1287) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_13
  simp only [← c79251, ← c79252, k2, ← c79253] at e_1287_13
  have hr := e_1287_13
  simp only [e_1287_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11762)) := by
  have c79254 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79255 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79256 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79257 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79258 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79259 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_14 := arithEq_of_rows h (row := 1287) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_14
  simp only [← c79254, ← c79255, ← c79256] at e_1287_14
  have e_1287_15 := arithEq_of_rows h (row := 1287) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_15
  simp only [← c79257, ← c79258, k2, ← c79259] at e_1287_15
  have hr := e_1287_15
  simp only [e_1287_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11763)) := by
  have c79260 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79261 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79262 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79263 := (publicBatchWrapper89_copies2476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79264 := (publicBatchWrapper89_copies2477 a h).1
  have c79265 := (publicBatchWrapper89_copies2477 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_16 := arithEq_of_rows h (row := 1287) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_16
  simp only [← c79260, ← c79261, ← c79262] at e_1287_16
  have e_1287_17 := arithEq_of_rows h (row := 1287) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_17
  simp only [← c79263, ← c79264, k2, ← c79265] at e_1287_17
  have hr := e_1287_17
  simp only [e_1287_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1287 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11764)) := by
  have c79266 := (publicBatchWrapper89_copies2477 a h).2.2.1
  have c79267 := (publicBatchWrapper89_copies2477 a h).2.2.2.1
  have c79268 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.1
  have c79269 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.1
  have c79270 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.1
  have c79271 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1287_18 := arithEq_of_rows h (row := 1287) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_18
  simp only [← c79266, ← c79267, ← c79268] at e_1287_18
  have e_1287_19 := arithEq_of_rows h (row := 1287) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1287_19
  simp only [← c79269, ← c79270, k2, ← c79271] at e_1287_19
  have hr := e_1287_19
  simp only [e_1287_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11765)) := by
  have c79272 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.1
  have c79273 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.1
  have c79274 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79275 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79276 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79277 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_0 := arithEq_of_rows h (row := 1288) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_0
  simp only [← c79272, ← c79273, ← c79274] at e_1288_0
  have e_1288_1 := arithEq_of_rows h (row := 1288) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_1
  simp only [← c79275, ← c79276, k2, ← c79277] at e_1288_1
  have hr := e_1288_1
  simp only [e_1288_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11766)) := by
  have c79278 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79279 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79280 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79281 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79282 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79283 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_2 := arithEq_of_rows h (row := 1288) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_2
  simp only [← c79278, ← c79279, ← c79280] at e_1288_2
  have e_1288_3 := arithEq_of_rows h (row := 1288) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_3
  simp only [← c79281, ← c79282, k2, ← c79283] at e_1288_3
  have hr := e_1288_3
  simp only [e_1288_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11767)) := by
  have c79284 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79285 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79286 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79287 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79288 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79289 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_4 := arithEq_of_rows h (row := 1288) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_4
  simp only [← c79284, ← c79285, ← c79286] at e_1288_4
  have e_1288_5 := arithEq_of_rows h (row := 1288) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_5
  simp only [← c79287, ← c79288, k2, ← c79289] at e_1288_5
  have hr := e_1288_5
  simp only [e_1288_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11768)) := by
  have c79290 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79291 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79292 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79293 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79294 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79295 := (publicBatchWrapper89_copies2477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_6 := arithEq_of_rows h (row := 1288) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_6
  simp only [← c79290, ← c79291, ← c79292] at e_1288_6
  have e_1288_7 := arithEq_of_rows h (row := 1288) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_7
  simp only [← c79293, ← c79294, k2, ← c79295] at e_1288_7
  have hr := e_1288_7
  simp only [e_1288_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11769)) := by
  have c79296 := (publicBatchWrapper89_copies2478 a h).1
  have c79297 := (publicBatchWrapper89_copies2478 a h).2.1
  have c79298 := (publicBatchWrapper89_copies2478 a h).2.2.1
  have c79299 := (publicBatchWrapper89_copies2478 a h).2.2.2.1
  have c79300 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.1
  have c79301 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_8 := arithEq_of_rows h (row := 1288) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_8
  simp only [← c79296, ← c79297, ← c79298] at e_1288_8
  have e_1288_9 := arithEq_of_rows h (row := 1288) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_9
  simp only [← c79299, ← c79300, k2, ← c79301] at e_1288_9
  have hr := e_1288_9
  simp only [e_1288_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11810)) := by
  have c79302 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.1
  have c79303 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.1
  have c79304 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.1
  have c79305 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.1
  have c79306 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79307 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_10 := arithEq_of_rows h (row := 1288) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_10
  simp only [← c79302, ← c79303, ← c79304] at e_1288_10
  have e_1288_11 := arithEq_of_rows h (row := 1288) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_11
  simp only [← c79305, ← c79306, k2, ← c79307] at e_1288_11
  have hr := e_1288_11
  simp only [e_1288_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11811)) := by
  have c79308 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79309 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79310 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79311 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79312 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79313 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_12 := arithEq_of_rows h (row := 1288) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_12
  simp only [← c79308, ← c79309, ← c79310] at e_1288_12
  have e_1288_13 := arithEq_of_rows h (row := 1288) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_13
  simp only [← c79311, ← c79312, k2, ← c79313] at e_1288_13
  have hr := e_1288_13
  simp only [e_1288_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11812)) := by
  have c79314 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79315 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79316 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79317 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79318 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79319 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_14 := arithEq_of_rows h (row := 1288) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_14
  simp only [← c79314, ← c79315, ← c79316] at e_1288_14
  have e_1288_15 := arithEq_of_rows h (row := 1288) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_15
  simp only [← c79317, ← c79318, k2, ← c79319] at e_1288_15
  have hr := e_1288_15
  simp only [e_1288_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11813)) := by
  have c79320 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79321 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79322 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79323 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79324 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79325 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_16 := arithEq_of_rows h (row := 1288) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_16
  simp only [← c79320, ← c79321, ← c79322] at e_1288_16
  have e_1288_17 := arithEq_of_rows h (row := 1288) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_17
  simp only [← c79323, ← c79324, k2, ← c79325] at e_1288_17
  have hr := e_1288_17
  simp only [e_1288_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1288 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11814)) := by
  have c79326 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79327 := (publicBatchWrapper89_copies2478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79328 := (publicBatchWrapper89_copies2479 a h).1
  have c79329 := (publicBatchWrapper89_copies2479 a h).2.1
  have c79330 := (publicBatchWrapper89_copies2479 a h).2.2.1
  have c79331 := (publicBatchWrapper89_copies2479 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1288_18 := arithEq_of_rows h (row := 1288) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_18
  simp only [← c79326, ← c79327, ← c79328] at e_1288_18
  have e_1288_19 := arithEq_of_rows h (row := 1288) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1288_19
  simp only [← c79329, ← c79330, k2, ← c79331] at e_1288_19
  have hr := e_1288_19
  simp only [e_1288_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11815)) := by
  have c79332 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.1
  have c79333 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.1
  have c79334 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.1
  have c79335 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.1
  have c79336 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.1
  have c79337 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_0 := arithEq_of_rows h (row := 1289) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_0
  simp only [← c79332, ← c79333, ← c79334] at e_1289_0
  have e_1289_1 := arithEq_of_rows h (row := 1289) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_1
  simp only [← c79335, ← c79336, k2, ← c79337] at e_1289_1
  have hr := e_1289_1
  simp only [e_1289_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11816)) := by
  have c79338 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79339 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79340 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79341 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79342 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79343 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_2 := arithEq_of_rows h (row := 1289) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_2
  simp only [← c79338, ← c79339, ← c79340] at e_1289_2
  have e_1289_3 := arithEq_of_rows h (row := 1289) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_3
  simp only [← c79341, ← c79342, k2, ← c79343] at e_1289_3
  have hr := e_1289_3
  simp only [e_1289_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11817)) := by
  have c79344 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79345 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79346 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79347 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79348 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79349 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_4 := arithEq_of_rows h (row := 1289) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_4
  simp only [← c79344, ← c79345, ← c79346] at e_1289_4
  have e_1289_5 := arithEq_of_rows h (row := 1289) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_5
  simp only [← c79347, ← c79348, k2, ← c79349] at e_1289_5
  have hr := e_1289_5
  simp only [e_1289_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11818)) := by
  have c79350 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79351 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79352 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79353 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79354 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79355 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_6 := arithEq_of_rows h (row := 1289) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_6
  simp only [← c79350, ← c79351, ← c79352] at e_1289_6
  have e_1289_7 := arithEq_of_rows h (row := 1289) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_7
  simp only [← c79353, ← c79354, k2, ← c79355] at e_1289_7
  have hr := e_1289_7
  simp only [e_1289_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11819)) := by
  have c79356 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79357 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79358 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79359 := (publicBatchWrapper89_copies2479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79360 := (publicBatchWrapper89_copies2480 a h).1
  have c79361 := (publicBatchWrapper89_copies2480 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_8 := arithEq_of_rows h (row := 1289) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_8
  simp only [← c79356, ← c79357, ← c79358] at e_1289_8
  have e_1289_9 := arithEq_of_rows h (row := 1289) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_9
  simp only [← c79359, ← c79360, k2, ← c79361] at e_1289_9
  have hr := e_1289_9
  simp only [e_1289_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11820)) := by
  have c79362 := (publicBatchWrapper89_copies2480 a h).2.2.1
  have c79363 := (publicBatchWrapper89_copies2480 a h).2.2.2.1
  have c79364 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.1
  have c79365 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.1
  have c79366 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.1
  have c79367 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_10 := arithEq_of_rows h (row := 1289) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_10
  simp only [← c79362, ← c79363, ← c79364] at e_1289_10
  have e_1289_11 := arithEq_of_rows h (row := 1289) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_11
  simp only [← c79365, ← c79366, k2, ← c79367] at e_1289_11
  have hr := e_1289_11
  simp only [e_1289_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11821)) := by
  have c79368 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.1
  have c79369 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.1
  have c79370 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79371 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79372 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79373 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_12 := arithEq_of_rows h (row := 1289) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_12
  simp only [← c79368, ← c79369, ← c79370] at e_1289_12
  have e_1289_13 := arithEq_of_rows h (row := 1289) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_13
  simp only [← c79371, ← c79372, k2, ← c79373] at e_1289_13
  have hr := e_1289_13
  simp only [e_1289_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11822)) := by
  have c79374 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79375 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79376 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79377 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79378 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79379 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_14 := arithEq_of_rows h (row := 1289) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_14
  simp only [← c79374, ← c79375, ← c79376] at e_1289_14
  have e_1289_15 := arithEq_of_rows h (row := 1289) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_15
  simp only [← c79377, ← c79378, k2, ← c79379] at e_1289_15
  have hr := e_1289_15
  simp only [e_1289_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11823)) := by
  have c79380 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79381 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79382 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79383 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79384 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79385 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_16 := arithEq_of_rows h (row := 1289) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_16
  simp only [← c79380, ← c79381, ← c79382] at e_1289_16
  have e_1289_17 := arithEq_of_rows h (row := 1289) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_17
  simp only [← c79383, ← c79384, k2, ← c79385] at e_1289_17
  have hr := e_1289_17
  simp only [e_1289_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1289 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11824)) := by
  have c79386 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79387 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79388 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79389 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79390 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79391 := (publicBatchWrapper89_copies2480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1289_18 := arithEq_of_rows h (row := 1289) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_18
  simp only [← c79386, ← c79387, ← c79388] at e_1289_18
  have e_1289_19 := arithEq_of_rows h (row := 1289) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1289_19
  simp only [← c79389, ← c79390, k2, ← c79391] at e_1289_19
  have hr := e_1289_19
  simp only [e_1289_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11825)) := by
  have c79392 := (publicBatchWrapper89_copies2481 a h).1
  have c79393 := (publicBatchWrapper89_copies2481 a h).2.1
  have c79394 := (publicBatchWrapper89_copies2481 a h).2.2.1
  have c79395 := (publicBatchWrapper89_copies2481 a h).2.2.2.1
  have c79396 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.1
  have c79397 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_0 := arithEq_of_rows h (row := 1290) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_0
  simp only [← c79392, ← c79393, ← c79394] at e_1290_0
  have e_1290_1 := arithEq_of_rows h (row := 1290) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_1
  simp only [← c79395, ← c79396, k2, ← c79397] at e_1290_1
  have hr := e_1290_1
  simp only [e_1290_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11826)) := by
  have c79398 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.1
  have c79399 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.1
  have c79400 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.1
  have c79401 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.1
  have c79402 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79403 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_2 := arithEq_of_rows h (row := 1290) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_2
  simp only [← c79398, ← c79399, ← c79400] at e_1290_2
  have e_1290_3 := arithEq_of_rows h (row := 1290) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_3
  simp only [← c79401, ← c79402, k2, ← c79403] at e_1290_3
  have hr := e_1290_3
  simp only [e_1290_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11827)) := by
  have c79404 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79405 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79406 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79407 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79408 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79409 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_4 := arithEq_of_rows h (row := 1290) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_4
  simp only [← c79404, ← c79405, ← c79406] at e_1290_4
  have e_1290_5 := arithEq_of_rows h (row := 1290) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_5
  simp only [← c79407, ← c79408, k2, ← c79409] at e_1290_5
  have hr := e_1290_5
  simp only [e_1290_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11828)) := by
  have c79410 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79411 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79412 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79413 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79414 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79415 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_6 := arithEq_of_rows h (row := 1290) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_6
  simp only [← c79410, ← c79411, ← c79412] at e_1290_6
  have e_1290_7 := arithEq_of_rows h (row := 1290) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_7
  simp only [← c79413, ← c79414, k2, ← c79415] at e_1290_7
  have hr := e_1290_7
  simp only [e_1290_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11829)) := by
  have c79416 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79417 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79418 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79419 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79420 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79421 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_8 := arithEq_of_rows h (row := 1290) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_8
  simp only [← c79416, ← c79417, ← c79418] at e_1290_8
  have e_1290_9 := arithEq_of_rows h (row := 1290) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_9
  simp only [← c79419, ← c79420, k2, ← c79421] at e_1290_9
  have hr := e_1290_9
  simp only [e_1290_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11830)) := by
  have c79422 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79423 := (publicBatchWrapper89_copies2481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79424 := (publicBatchWrapper89_copies2482 a h).1
  have c79425 := (publicBatchWrapper89_copies2482 a h).2.1
  have c79426 := (publicBatchWrapper89_copies2482 a h).2.2.1
  have c79427 := (publicBatchWrapper89_copies2482 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_10 := arithEq_of_rows h (row := 1290) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_10
  simp only [← c79422, ← c79423, ← c79424] at e_1290_10
  have e_1290_11 := arithEq_of_rows h (row := 1290) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_11
  simp only [← c79425, ← c79426, k2, ← c79427] at e_1290_11
  have hr := e_1290_11
  simp only [e_1290_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11831)) := by
  have c79428 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.1
  have c79429 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.1
  have c79430 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.1
  have c79431 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.1
  have c79432 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.1
  have c79433 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_12 := arithEq_of_rows h (row := 1290) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_12
  simp only [← c79428, ← c79429, ← c79430] at e_1290_12
  have e_1290_13 := arithEq_of_rows h (row := 1290) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_13
  simp only [← c79431, ← c79432, k2, ← c79433] at e_1290_13
  have hr := e_1290_13
  simp only [e_1290_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11832)) := by
  have c79434 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79435 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79436 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79437 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79438 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79439 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_14 := arithEq_of_rows h (row := 1290) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_14
  simp only [← c79434, ← c79435, ← c79436] at e_1290_14
  have e_1290_15 := arithEq_of_rows h (row := 1290) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_15
  simp only [← c79437, ← c79438, k2, ← c79439] at e_1290_15
  have hr := e_1290_15
  simp only [e_1290_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11833)) := by
  have c79440 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79441 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79442 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79443 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79444 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79445 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_16 := arithEq_of_rows h (row := 1290) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_16
  simp only [← c79440, ← c79441, ← c79442] at e_1290_16
  have e_1290_17 := arithEq_of_rows h (row := 1290) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_17
  simp only [← c79443, ← c79444, k2, ← c79445] at e_1290_17
  have hr := e_1290_17
  simp only [e_1290_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1290 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11834)) := by
  have c79446 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79447 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79448 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79449 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79450 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79451 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1290_18 := arithEq_of_rows h (row := 1290) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_18
  simp only [← c79446, ← c79447, ← c79448] at e_1290_18
  have e_1290_19 := arithEq_of_rows h (row := 1290) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1290_19
  simp only [← c79449, ← c79450, k2, ← c79451] at e_1290_19
  have hr := e_1290_19
  simp only [e_1290_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11835)) := by
  have c79452 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79453 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79454 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79455 := (publicBatchWrapper89_copies2482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79456 := (publicBatchWrapper89_copies2483 a h).1
  have c79457 := (publicBatchWrapper89_copies2483 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_0 := arithEq_of_rows h (row := 1291) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_0
  simp only [← c79452, ← c79453, ← c79454] at e_1291_0
  have e_1291_1 := arithEq_of_rows h (row := 1291) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_1
  simp only [← c79455, ← c79456, k2, ← c79457] at e_1291_1
  have hr := e_1291_1
  simp only [e_1291_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11836)) := by
  have c79458 := (publicBatchWrapper89_copies2483 a h).2.2.1
  have c79459 := (publicBatchWrapper89_copies2483 a h).2.2.2.1
  have c79460 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.1
  have c79461 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.1
  have c79462 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.1
  have c79463 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_2 := arithEq_of_rows h (row := 1291) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_2
  simp only [← c79458, ← c79459, ← c79460] at e_1291_2
  have e_1291_3 := arithEq_of_rows h (row := 1291) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_3
  simp only [← c79461, ← c79462, k2, ← c79463] at e_1291_3
  have hr := e_1291_3
  simp only [e_1291_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11837)) := by
  have c79464 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.1
  have c79465 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.1
  have c79466 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79467 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79468 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79469 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_4 := arithEq_of_rows h (row := 1291) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_4
  simp only [← c79464, ← c79465, ← c79466] at e_1291_4
  have e_1291_5 := arithEq_of_rows h (row := 1291) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_5
  simp only [← c79467, ← c79468, k2, ← c79469] at e_1291_5
  have hr := e_1291_5
  simp only [e_1291_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11838)) := by
  have c79470 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79471 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79472 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79473 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79474 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79475 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_6 := arithEq_of_rows h (row := 1291) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_6
  simp only [← c79470, ← c79471, ← c79472] at e_1291_6
  have e_1291_7 := arithEq_of_rows h (row := 1291) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_7
  simp only [← c79473, ← c79474, k2, ← c79475] at e_1291_7
  have hr := e_1291_7
  simp only [e_1291_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11839)) := by
  have c79476 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79477 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79478 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79479 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79480 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79481 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_8 := arithEq_of_rows h (row := 1291) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_8
  simp only [← c79476, ← c79477, ← c79478] at e_1291_8
  have e_1291_9 := arithEq_of_rows h (row := 1291) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_9
  simp only [← c79479, ← c79480, k2, ← c79481] at e_1291_9
  have hr := e_1291_9
  simp only [e_1291_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11840)) := by
  have c79482 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79483 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79484 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79485 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79486 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79487 := (publicBatchWrapper89_copies2483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_10 := arithEq_of_rows h (row := 1291) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_10
  simp only [← c79482, ← c79483, ← c79484] at e_1291_10
  have e_1291_11 := arithEq_of_rows h (row := 1291) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_11
  simp only [← c79485, ← c79486, k2, ← c79487] at e_1291_11
  have hr := e_1291_11
  simp only [e_1291_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11841)) := by
  have c79488 := (publicBatchWrapper89_copies2484 a h).1
  have c79489 := (publicBatchWrapper89_copies2484 a h).2.1
  have c79490 := (publicBatchWrapper89_copies2484 a h).2.2.1
  have c79491 := (publicBatchWrapper89_copies2484 a h).2.2.2.1
  have c79492 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.1
  have c79493 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_12 := arithEq_of_rows h (row := 1291) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_12
  simp only [← c79488, ← c79489, ← c79490] at e_1291_12
  have e_1291_13 := arithEq_of_rows h (row := 1291) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_13
  simp only [← c79491, ← c79492, k2, ← c79493] at e_1291_13
  have hr := e_1291_13
  simp only [e_1291_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11842)) := by
  have c79494 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.1
  have c79495 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.1
  have c79496 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.1
  have c79497 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.1
  have c79498 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79499 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_14 := arithEq_of_rows h (row := 1291) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_14
  simp only [← c79494, ← c79495, ← c79496] at e_1291_14
  have e_1291_15 := arithEq_of_rows h (row := 1291) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_15
  simp only [← c79497, ← c79498, k2, ← c79499] at e_1291_15
  have hr := e_1291_15
  simp only [e_1291_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11843)) := by
  have c79500 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79501 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79502 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79503 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79504 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79505 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_16 := arithEq_of_rows h (row := 1291) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_16
  simp only [← c79500, ← c79501, ← c79502] at e_1291_16
  have e_1291_17 := arithEq_of_rows h (row := 1291) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_17
  simp only [← c79503, ← c79504, k2, ← c79505] at e_1291_17
  have hr := e_1291_17
  simp only [e_1291_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1291 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11844)) := by
  have c79506 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79507 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79508 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79509 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79510 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79511 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1291_18 := arithEq_of_rows h (row := 1291) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_18
  simp only [← c79506, ← c79507, ← c79508] at e_1291_18
  have e_1291_19 := arithEq_of_rows h (row := 1291) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1291_19
  simp only [← c79509, ← c79510, k2, ← c79511] at e_1291_19
  have hr := e_1291_19
  simp only [e_1291_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11845)) := by
  have c79512 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79513 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79514 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79515 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79516 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79517 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_0 := arithEq_of_rows h (row := 1292) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_0
  simp only [← c79512, ← c79513, ← c79514] at e_1292_0
  have e_1292_1 := arithEq_of_rows h (row := 1292) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_1
  simp only [← c79515, ← c79516, k2, ← c79517] at e_1292_1
  have hr := e_1292_1
  simp only [e_1292_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11846)) := by
  have c79518 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79519 := (publicBatchWrapper89_copies2484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79520 := (publicBatchWrapper89_copies2485 a h).1
  have c79521 := (publicBatchWrapper89_copies2485 a h).2.1
  have c79522 := (publicBatchWrapper89_copies2485 a h).2.2.1
  have c79523 := (publicBatchWrapper89_copies2485 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_2 := arithEq_of_rows h (row := 1292) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_2
  simp only [← c79518, ← c79519, ← c79520] at e_1292_2
  have e_1292_3 := arithEq_of_rows h (row := 1292) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_3
  simp only [← c79521, ← c79522, k2, ← c79523] at e_1292_3
  have hr := e_1292_3
  simp only [e_1292_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11847)) := by
  have c79524 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.1
  have c79525 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.1
  have c79526 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.1
  have c79527 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.1
  have c79528 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.1
  have c79529 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_4 := arithEq_of_rows h (row := 1292) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_4
  simp only [← c79524, ← c79525, ← c79526] at e_1292_4
  have e_1292_5 := arithEq_of_rows h (row := 1292) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_5
  simp only [← c79527, ← c79528, k2, ← c79529] at e_1292_5
  have hr := e_1292_5
  simp only [e_1292_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11848)) := by
  have c79530 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79531 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79532 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79533 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79534 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79535 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_6 := arithEq_of_rows h (row := 1292) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_6
  simp only [← c79530, ← c79531, ← c79532] at e_1292_6
  have e_1292_7 := arithEq_of_rows h (row := 1292) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_7
  simp only [← c79533, ← c79534, k2, ← c79535] at e_1292_7
  have hr := e_1292_7
  simp only [e_1292_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11849)) := by
  have c79536 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79537 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79538 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79539 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79540 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79541 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_8 := arithEq_of_rows h (row := 1292) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_8
  simp only [← c79536, ← c79537, ← c79538] at e_1292_8
  have e_1292_9 := arithEq_of_rows h (row := 1292) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_9
  simp only [← c79539, ← c79540, k2, ← c79541] at e_1292_9
  have hr := e_1292_9
  simp only [e_1292_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11850)) := by
  have c79542 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79543 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79544 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79545 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79546 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79547 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_10 := arithEq_of_rows h (row := 1292) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_10
  simp only [← c79542, ← c79543, ← c79544] at e_1292_10
  have e_1292_11 := arithEq_of_rows h (row := 1292) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_11
  simp only [← c79545, ← c79546, k2, ← c79547] at e_1292_11
  have hr := e_1292_11
  simp only [e_1292_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11851)) := by
  have c79548 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79549 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79550 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79551 := (publicBatchWrapper89_copies2485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79552 := (publicBatchWrapper89_copies2486 a h).1
  have c79553 := (publicBatchWrapper89_copies2486 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_12 := arithEq_of_rows h (row := 1292) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_12
  simp only [← c79548, ← c79549, ← c79550] at e_1292_12
  have e_1292_13 := arithEq_of_rows h (row := 1292) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_13
  simp only [← c79551, ← c79552, k2, ← c79553] at e_1292_13
  have hr := e_1292_13
  simp only [e_1292_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11852)) := by
  have c79554 := (publicBatchWrapper89_copies2486 a h).2.2.1
  have c79555 := (publicBatchWrapper89_copies2486 a h).2.2.2.1
  have c79556 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.1
  have c79557 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.1
  have c79558 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.1
  have c79559 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_14 := arithEq_of_rows h (row := 1292) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_14
  simp only [← c79554, ← c79555, ← c79556] at e_1292_14
  have e_1292_15 := arithEq_of_rows h (row := 1292) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_15
  simp only [← c79557, ← c79558, k2, ← c79559] at e_1292_15
  have hr := e_1292_15
  simp only [e_1292_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11853)) := by
  have c79560 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.1
  have c79561 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.1
  have c79562 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79563 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79564 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79565 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_16 := arithEq_of_rows h (row := 1292) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_16
  simp only [← c79560, ← c79561, ← c79562] at e_1292_16
  have e_1292_17 := arithEq_of_rows h (row := 1292) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_17
  simp only [← c79563, ← c79564, k2, ← c79565] at e_1292_17
  have hr := e_1292_17
  simp only [e_1292_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1292 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11854)) := by
  have c79566 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79567 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79568 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79569 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79570 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79571 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1292_18 := arithEq_of_rows h (row := 1292) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_18
  simp only [← c79566, ← c79567, ← c79568] at e_1292_18
  have e_1292_19 := arithEq_of_rows h (row := 1292) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1292_19
  simp only [← c79569, ← c79570, k2, ← c79571] at e_1292_19
  have hr := e_1292_19
  simp only [e_1292_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11855)) := by
  have c79572 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79573 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79574 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79575 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79576 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79577 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_0 := arithEq_of_rows h (row := 1293) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_0
  simp only [← c79572, ← c79573, ← c79574] at e_1293_0
  have e_1293_1 := arithEq_of_rows h (row := 1293) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_1
  simp only [← c79575, ← c79576, k2, ← c79577] at e_1293_1
  have hr := e_1293_1
  simp only [e_1293_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11856)) := by
  have c79578 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79579 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79580 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79581 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79582 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79583 := (publicBatchWrapper89_copies2486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_2 := arithEq_of_rows h (row := 1293) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_2
  simp only [← c79578, ← c79579, ← c79580] at e_1293_2
  have e_1293_3 := arithEq_of_rows h (row := 1293) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_3
  simp only [← c79581, ← c79582, k2, ← c79583] at e_1293_3
  have hr := e_1293_3
  simp only [e_1293_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11857)) := by
  have c79584 := (publicBatchWrapper89_copies2487 a h).1
  have c79585 := (publicBatchWrapper89_copies2487 a h).2.1
  have c79586 := (publicBatchWrapper89_copies2487 a h).2.2.1
  have c79587 := (publicBatchWrapper89_copies2487 a h).2.2.2.1
  have c79588 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.1
  have c79589 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_4 := arithEq_of_rows h (row := 1293) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_4
  simp only [← c79584, ← c79585, ← c79586] at e_1293_4
  have e_1293_5 := arithEq_of_rows h (row := 1293) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_5
  simp only [← c79587, ← c79588, k2, ← c79589] at e_1293_5
  have hr := e_1293_5
  simp only [e_1293_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11858)) := by
  have c79590 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.1
  have c79591 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.1
  have c79592 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.1
  have c79593 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.1
  have c79594 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79595 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_6 := arithEq_of_rows h (row := 1293) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_6
  simp only [← c79590, ← c79591, ← c79592] at e_1293_6
  have e_1293_7 := arithEq_of_rows h (row := 1293) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_7
  simp only [← c79593, ← c79594, k2, ← c79595] at e_1293_7
  have hr := e_1293_7
  simp only [e_1293_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11859)) := by
  have c79596 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79597 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79598 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79599 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79600 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79601 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_8 := arithEq_of_rows h (row := 1293) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_8
  simp only [← c79596, ← c79597, ← c79598] at e_1293_8
  have e_1293_9 := arithEq_of_rows h (row := 1293) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_9
  simp only [← c79599, ← c79600, k2, ← c79601] at e_1293_9
  have hr := e_1293_9
  simp only [e_1293_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11860)) := by
  have c79602 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79603 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79604 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79605 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79606 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79607 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_10 := arithEq_of_rows h (row := 1293) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_10
  simp only [← c79602, ← c79603, ← c79604] at e_1293_10
  have e_1293_11 := arithEq_of_rows h (row := 1293) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_11
  simp only [← c79605, ← c79606, k2, ← c79607] at e_1293_11
  have hr := e_1293_11
  simp only [e_1293_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11861)) := by
  have c79608 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79609 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79610 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79611 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79612 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79613 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_12 := arithEq_of_rows h (row := 1293) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_12
  simp only [← c79608, ← c79609, ← c79610] at e_1293_12
  have e_1293_13 := arithEq_of_rows h (row := 1293) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_13
  simp only [← c79611, ← c79612, k2, ← c79613] at e_1293_13
  have hr := e_1293_13
  simp only [e_1293_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11862)) := by
  have c79614 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79615 := (publicBatchWrapper89_copies2487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79616 := (publicBatchWrapper89_copies2488 a h).1
  have c79617 := (publicBatchWrapper89_copies2488 a h).2.1
  have c79618 := (publicBatchWrapper89_copies2488 a h).2.2.1
  have c79619 := (publicBatchWrapper89_copies2488 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_14 := arithEq_of_rows h (row := 1293) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_14
  simp only [← c79614, ← c79615, ← c79616] at e_1293_14
  have e_1293_15 := arithEq_of_rows h (row := 1293) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_15
  simp only [← c79617, ← c79618, k2, ← c79619] at e_1293_15
  have hr := e_1293_15
  simp only [e_1293_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11863)) := by
  have c79620 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.1
  have c79621 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.1
  have c79622 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.1
  have c79623 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.1
  have c79624 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.1
  have c79625 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_16 := arithEq_of_rows h (row := 1293) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_16
  simp only [← c79620, ← c79621, ← c79622] at e_1293_16
  have e_1293_17 := arithEq_of_rows h (row := 1293) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_17
  simp only [← c79623, ← c79624, k2, ← c79625] at e_1293_17
  have hr := e_1293_17
  simp only [e_1293_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1293 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11864)) := by
  have c79626 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79627 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79628 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79629 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79630 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79631 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1293_18 := arithEq_of_rows h (row := 1293) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_18
  simp only [← c79626, ← c79627, ← c79628] at e_1293_18
  have e_1293_19 := arithEq_of_rows h (row := 1293) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1293_19
  simp only [← c79629, ← c79630, k2, ← c79631] at e_1293_19
  have hr := e_1293_19
  simp only [e_1293_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11865)) := by
  have c79632 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79633 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79634 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79635 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79636 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79637 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_0 := arithEq_of_rows h (row := 1294) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_0
  simp only [← c79632, ← c79633, ← c79634] at e_1294_0
  have e_1294_1 := arithEq_of_rows h (row := 1294) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_1
  simp only [← c79635, ← c79636, k2, ← c79637] at e_1294_1
  have hr := e_1294_1
  simp only [e_1294_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11866)) := by
  have c79638 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79639 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79640 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79641 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79642 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79643 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_2 := arithEq_of_rows h (row := 1294) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_2
  simp only [← c79638, ← c79639, ← c79640] at e_1294_2
  have e_1294_3 := arithEq_of_rows h (row := 1294) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_3
  simp only [← c79641, ← c79642, k2, ← c79643] at e_1294_3
  have hr := e_1294_3
  simp only [e_1294_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11867)) := by
  have c79644 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79645 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79646 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79647 := (publicBatchWrapper89_copies2488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79648 := (publicBatchWrapper89_copies2489 a h).1
  have c79649 := (publicBatchWrapper89_copies2489 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_4 := arithEq_of_rows h (row := 1294) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_4
  simp only [← c79644, ← c79645, ← c79646] at e_1294_4
  have e_1294_5 := arithEq_of_rows h (row := 1294) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_5
  simp only [← c79647, ← c79648, k2, ← c79649] at e_1294_5
  have hr := e_1294_5
  simp only [e_1294_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11868)) := by
  have c79650 := (publicBatchWrapper89_copies2489 a h).2.2.1
  have c79651 := (publicBatchWrapper89_copies2489 a h).2.2.2.1
  have c79652 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.1
  have c79653 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.1
  have c79654 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.1
  have c79655 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_6 := arithEq_of_rows h (row := 1294) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_6
  simp only [← c79650, ← c79651, ← c79652] at e_1294_6
  have e_1294_7 := arithEq_of_rows h (row := 1294) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_7
  simp only [← c79653, ← c79654, k2, ← c79655] at e_1294_7
  have hr := e_1294_7
  simp only [e_1294_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11869)) := by
  have c79656 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.1
  have c79657 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.1
  have c79658 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79659 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79660 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79661 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_8 := arithEq_of_rows h (row := 1294) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_8
  simp only [← c79656, ← c79657, ← c79658] at e_1294_8
  have e_1294_9 := arithEq_of_rows h (row := 1294) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_9
  simp only [← c79659, ← c79660, k2, ← c79661] at e_1294_9
  have hr := e_1294_9
  simp only [e_1294_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11870)) := by
  have c79662 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79663 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79664 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79665 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79666 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79667 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_10 := arithEq_of_rows h (row := 1294) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_10
  simp only [← c79662, ← c79663, ← c79664] at e_1294_10
  have e_1294_11 := arithEq_of_rows h (row := 1294) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_11
  simp only [← c79665, ← c79666, k2, ← c79667] at e_1294_11
  have hr := e_1294_11
  simp only [e_1294_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11871)) := by
  have c79668 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79669 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79670 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79671 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79672 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79673 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_12 := arithEq_of_rows h (row := 1294) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_12
  simp only [← c79668, ← c79669, ← c79670] at e_1294_12
  have e_1294_13 := arithEq_of_rows h (row := 1294) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_13
  simp only [← c79671, ← c79672, k2, ← c79673] at e_1294_13
  have hr := e_1294_13
  simp only [e_1294_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11872)) := by
  have c79674 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79675 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79676 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79677 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79678 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79679 := (publicBatchWrapper89_copies2489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_14 := arithEq_of_rows h (row := 1294) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_14
  simp only [← c79674, ← c79675, ← c79676] at e_1294_14
  have e_1294_15 := arithEq_of_rows h (row := 1294) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_15
  simp only [← c79677, ← c79678, k2, ← c79679] at e_1294_15
  have hr := e_1294_15
  simp only [e_1294_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11873)) := by
  have c79680 := (publicBatchWrapper89_copies2490 a h).1
  have c79681 := (publicBatchWrapper89_copies2490 a h).2.1
  have c79682 := (publicBatchWrapper89_copies2490 a h).2.2.1
  have c79683 := (publicBatchWrapper89_copies2490 a h).2.2.2.1
  have c79684 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.1
  have c79685 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_16 := arithEq_of_rows h (row := 1294) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_16
  simp only [← c79680, ← c79681, ← c79682] at e_1294_16
  have e_1294_17 := arithEq_of_rows h (row := 1294) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_17
  simp only [← c79683, ← c79684, k2, ← c79685] at e_1294_17
  have hr := e_1294_17
  simp only [e_1294_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1294 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11874)) := by
  have c79686 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.1
  have c79687 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.1
  have c79688 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.1
  have c79689 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.1
  have c79690 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79691 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1294_18 := arithEq_of_rows h (row := 1294) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_18
  simp only [← c79686, ← c79687, ← c79688] at e_1294_18
  have e_1294_19 := arithEq_of_rows h (row := 1294) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1294_19
  simp only [← c79689, ← c79690, k2, ← c79691] at e_1294_19
  have hr := e_1294_19
  simp only [e_1294_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11875)) := by
  have c79692 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79693 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79694 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79695 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79696 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79697 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_0 := arithEq_of_rows h (row := 1295) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_0
  simp only [← c79692, ← c79693, ← c79694] at e_1295_0
  have e_1295_1 := arithEq_of_rows h (row := 1295) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_1
  simp only [← c79695, ← c79696, k2, ← c79697] at e_1295_1
  have hr := e_1295_1
  simp only [e_1295_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11876)) := by
  have c79698 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79699 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79700 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79701 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79702 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79703 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_2 := arithEq_of_rows h (row := 1295) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_2
  simp only [← c79698, ← c79699, ← c79700] at e_1295_2
  have e_1295_3 := arithEq_of_rows h (row := 1295) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_3
  simp only [← c79701, ← c79702, k2, ← c79703] at e_1295_3
  have hr := e_1295_3
  simp only [e_1295_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11877)) := by
  have c79704 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79705 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79706 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79707 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79708 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79709 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_4 := arithEq_of_rows h (row := 1295) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_4
  simp only [← c79704, ← c79705, ← c79706] at e_1295_4
  have e_1295_5 := arithEq_of_rows h (row := 1295) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_5
  simp only [← c79707, ← c79708, k2, ← c79709] at e_1295_5
  have hr := e_1295_5
  simp only [e_1295_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11878)) := by
  have c79710 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79711 := (publicBatchWrapper89_copies2490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79712 := (publicBatchWrapper89_copies2491 a h).1
  have c79713 := (publicBatchWrapper89_copies2491 a h).2.1
  have c79714 := (publicBatchWrapper89_copies2491 a h).2.2.1
  have c79715 := (publicBatchWrapper89_copies2491 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_6 := arithEq_of_rows h (row := 1295) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_6
  simp only [← c79710, ← c79711, ← c79712] at e_1295_6
  have e_1295_7 := arithEq_of_rows h (row := 1295) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_7
  simp only [← c79713, ← c79714, k2, ← c79715] at e_1295_7
  have hr := e_1295_7
  simp only [e_1295_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11879)) := by
  have c79716 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.1
  have c79717 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.1
  have c79718 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.1
  have c79719 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.1
  have c79720 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.1
  have c79721 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_8 := arithEq_of_rows h (row := 1295) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_8
  simp only [← c79716, ← c79717, ← c79718] at e_1295_8
  have e_1295_9 := arithEq_of_rows h (row := 1295) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_9
  simp only [← c79719, ← c79720, k2, ← c79721] at e_1295_9
  have hr := e_1295_9
  simp only [e_1295_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11880)) := by
  have c79722 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79723 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79724 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79725 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79726 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79727 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_10 := arithEq_of_rows h (row := 1295) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_10
  simp only [← c79722, ← c79723, ← c79724] at e_1295_10
  have e_1295_11 := arithEq_of_rows h (row := 1295) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_11
  simp only [← c79725, ← c79726, k2, ← c79727] at e_1295_11
  have hr := e_1295_11
  simp only [e_1295_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11881)) := by
  have c79728 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79729 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79730 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79731 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79732 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79733 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_12 := arithEq_of_rows h (row := 1295) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_12
  simp only [← c79728, ← c79729, ← c79730] at e_1295_12
  have e_1295_13 := arithEq_of_rows h (row := 1295) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_13
  simp only [← c79731, ← c79732, k2, ← c79733] at e_1295_13
  have hr := e_1295_13
  simp only [e_1295_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11882)) := by
  have c79734 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79735 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79736 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79737 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79738 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79739 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_14 := arithEq_of_rows h (row := 1295) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_14
  simp only [← c79734, ← c79735, ← c79736] at e_1295_14
  have e_1295_15 := arithEq_of_rows h (row := 1295) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_15
  simp only [← c79737, ← c79738, k2, ← c79739] at e_1295_15
  have hr := e_1295_15
  simp only [e_1295_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11883)) := by
  have c79740 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79741 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79742 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79743 := (publicBatchWrapper89_copies2491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79744 := (publicBatchWrapper89_copies2492 a h).1
  have c79745 := (publicBatchWrapper89_copies2492 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_16 := arithEq_of_rows h (row := 1295) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_16
  simp only [← c79740, ← c79741, ← c79742] at e_1295_16
  have e_1295_17 := arithEq_of_rows h (row := 1295) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_17
  simp only [← c79743, ← c79744, k2, ← c79745] at e_1295_17
  have hr := e_1295_17
  simp only [e_1295_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1295 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11884)) := by
  have c79746 := (publicBatchWrapper89_copies2492 a h).2.2.1
  have c79747 := (publicBatchWrapper89_copies2492 a h).2.2.2.1
  have c79748 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.1
  have c79749 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.1
  have c79750 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.1
  have c79751 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1295_18 := arithEq_of_rows h (row := 1295) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_18
  simp only [← c79746, ← c79747, ← c79748] at e_1295_18
  have e_1295_19 := arithEq_of_rows h (row := 1295) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1295_19
  simp only [← c79749, ← c79750, k2, ← c79751] at e_1295_19
  have hr := e_1295_19
  simp only [e_1295_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11885)) := by
  have c79752 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.1
  have c79753 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.1
  have c79754 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79755 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79756 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79757 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_0 := arithEq_of_rows h (row := 1296) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_0
  simp only [← c79752, ← c79753, ← c79754] at e_1296_0
  have e_1296_1 := arithEq_of_rows h (row := 1296) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_1
  simp only [← c79755, ← c79756, k2, ← c79757] at e_1296_1
  have hr := e_1296_1
  simp only [e_1296_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11886)) := by
  have c79758 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79759 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79760 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79761 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79762 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79763 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_2 := arithEq_of_rows h (row := 1296) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_2
  simp only [← c79758, ← c79759, ← c79760] at e_1296_2
  have e_1296_3 := arithEq_of_rows h (row := 1296) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_3
  simp only [← c79761, ← c79762, k2, ← c79763] at e_1296_3
  have hr := e_1296_3
  simp only [e_1296_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11887)) := by
  have c79764 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79765 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79766 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79767 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79768 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79769 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_4 := arithEq_of_rows h (row := 1296) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_4
  simp only [← c79764, ← c79765, ← c79766] at e_1296_4
  have e_1296_5 := arithEq_of_rows h (row := 1296) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_5
  simp only [← c79767, ← c79768, k2, ← c79769] at e_1296_5
  have hr := e_1296_5
  simp only [e_1296_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11888)) := by
  have c79770 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79771 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79772 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79773 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79774 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79775 := (publicBatchWrapper89_copies2492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_6 := arithEq_of_rows h (row := 1296) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_6
  simp only [← c79770, ← c79771, ← c79772] at e_1296_6
  have e_1296_7 := arithEq_of_rows h (row := 1296) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_7
  simp only [← c79773, ← c79774, k2, ← c79775] at e_1296_7
  have hr := e_1296_7
  simp only [e_1296_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11889)) := by
  have c79776 := (publicBatchWrapper89_copies2493 a h).1
  have c79777 := (publicBatchWrapper89_copies2493 a h).2.1
  have c79778 := (publicBatchWrapper89_copies2493 a h).2.2.1
  have c79779 := (publicBatchWrapper89_copies2493 a h).2.2.2.1
  have c79780 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.1
  have c79781 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_8 := arithEq_of_rows h (row := 1296) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_8
  simp only [← c79776, ← c79777, ← c79778] at e_1296_8
  have e_1296_9 := arithEq_of_rows h (row := 1296) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_9
  simp only [← c79779, ← c79780, k2, ← c79781] at e_1296_9
  have hr := e_1296_9
  simp only [e_1296_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11890)) := by
  have c79782 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.1
  have c79783 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.1
  have c79784 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.1
  have c79785 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.1
  have c79786 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79787 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_10 := arithEq_of_rows h (row := 1296) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_10
  simp only [← c79782, ← c79783, ← c79784] at e_1296_10
  have e_1296_11 := arithEq_of_rows h (row := 1296) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_11
  simp only [← c79785, ← c79786, k2, ← c79787] at e_1296_11
  have hr := e_1296_11
  simp only [e_1296_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11891)) := by
  have c79788 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79789 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79790 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79791 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79792 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79793 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_12 := arithEq_of_rows h (row := 1296) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_12
  simp only [← c79788, ← c79789, ← c79790] at e_1296_12
  have e_1296_13 := arithEq_of_rows h (row := 1296) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_13
  simp only [← c79791, ← c79792, k2, ← c79793] at e_1296_13
  have hr := e_1296_13
  simp only [e_1296_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11892)) := by
  have c79794 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79795 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79796 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79797 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79798 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79799 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_14 := arithEq_of_rows h (row := 1296) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_14
  simp only [← c79794, ← c79795, ← c79796] at e_1296_14
  have e_1296_15 := arithEq_of_rows h (row := 1296) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_15
  simp only [← c79797, ← c79798, k2, ← c79799] at e_1296_15
  have hr := e_1296_15
  simp only [e_1296_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11893)) := by
  have c79800 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79801 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79802 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79803 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79804 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79805 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_16 := arithEq_of_rows h (row := 1296) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_16
  simp only [← c79800, ← c79801, ← c79802] at e_1296_16
  have e_1296_17 := arithEq_of_rows h (row := 1296) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_17
  simp only [← c79803, ← c79804, k2, ← c79805] at e_1296_17
  have hr := e_1296_17
  simp only [e_1296_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1296 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11894)) := by
  have c79806 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79807 := (publicBatchWrapper89_copies2493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79808 := (publicBatchWrapper89_copies2494 a h).1
  have c79809 := (publicBatchWrapper89_copies2494 a h).2.1
  have c79810 := (publicBatchWrapper89_copies2494 a h).2.2.1
  have c79811 := (publicBatchWrapper89_copies2494 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1296_18 := arithEq_of_rows h (row := 1296) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_18
  simp only [← c79806, ← c79807, ← c79808] at e_1296_18
  have e_1296_19 := arithEq_of_rows h (row := 1296) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1296_19
  simp only [← c79809, ← c79810, k2, ← c79811] at e_1296_19
  have hr := e_1296_19
  simp only [e_1296_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11895)) := by
  have c79812 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.1
  have c79813 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.1
  have c79814 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.1
  have c79815 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.1
  have c79816 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.1
  have c79817 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_0 := arithEq_of_rows h (row := 1297) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_0
  simp only [← c79812, ← c79813, ← c79814] at e_1297_0
  have e_1297_1 := arithEq_of_rows h (row := 1297) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_1
  simp only [← c79815, ← c79816, k2, ← c79817] at e_1297_1
  have hr := e_1297_1
  simp only [e_1297_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11896)) := by
  have c79818 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79819 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79820 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79821 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79822 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79823 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_2 := arithEq_of_rows h (row := 1297) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_2
  simp only [← c79818, ← c79819, ← c79820] at e_1297_2
  have e_1297_3 := arithEq_of_rows h (row := 1297) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_3
  simp only [← c79821, ← c79822, k2, ← c79823] at e_1297_3
  have hr := e_1297_3
  simp only [e_1297_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11897)) := by
  have c79824 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79825 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79826 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79827 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79828 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79829 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_4 := arithEq_of_rows h (row := 1297) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_4
  simp only [← c79824, ← c79825, ← c79826] at e_1297_4
  have e_1297_5 := arithEq_of_rows h (row := 1297) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_5
  simp only [← c79827, ← c79828, k2, ← c79829] at e_1297_5
  have hr := e_1297_5
  simp only [e_1297_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11898)) := by
  have c79830 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79831 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79832 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79833 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79834 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79835 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_6 := arithEq_of_rows h (row := 1297) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_6
  simp only [← c79830, ← c79831, ← c79832] at e_1297_6
  have e_1297_7 := arithEq_of_rows h (row := 1297) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_7
  simp only [← c79833, ← c79834, k2, ← c79835] at e_1297_7
  have hr := e_1297_7
  simp only [e_1297_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11899)) := by
  have c79836 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79837 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79838 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79839 := (publicBatchWrapper89_copies2494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79840 := (publicBatchWrapper89_copies2495 a h).1
  have c79841 := (publicBatchWrapper89_copies2495 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_8 := arithEq_of_rows h (row := 1297) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_8
  simp only [← c79836, ← c79837, ← c79838] at e_1297_8
  have e_1297_9 := arithEq_of_rows h (row := 1297) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_9
  simp only [← c79839, ← c79840, k2, ← c79841] at e_1297_9
  have hr := e_1297_9
  simp only [e_1297_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11900)) := by
  have c79842 := (publicBatchWrapper89_copies2495 a h).2.2.1
  have c79843 := (publicBatchWrapper89_copies2495 a h).2.2.2.1
  have c79844 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.1
  have c79845 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.1
  have c79846 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.1
  have c79847 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_10 := arithEq_of_rows h (row := 1297) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_10
  simp only [← c79842, ← c79843, ← c79844] at e_1297_10
  have e_1297_11 := arithEq_of_rows h (row := 1297) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_11
  simp only [← c79845, ← c79846, k2, ← c79847] at e_1297_11
  have hr := e_1297_11
  simp only [e_1297_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11901)) := by
  have c79848 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.1
  have c79849 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.1
  have c79850 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79851 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79852 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79853 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_12 := arithEq_of_rows h (row := 1297) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_12
  simp only [← c79848, ← c79849, ← c79850] at e_1297_12
  have e_1297_13 := arithEq_of_rows h (row := 1297) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_13
  simp only [← c79851, ← c79852, k2, ← c79853] at e_1297_13
  have hr := e_1297_13
  simp only [e_1297_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11902)) := by
  have c79854 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79855 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79856 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79857 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79858 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79859 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_14 := arithEq_of_rows h (row := 1297) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_14
  simp only [← c79854, ← c79855, ← c79856] at e_1297_14
  have e_1297_15 := arithEq_of_rows h (row := 1297) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_15
  simp only [← c79857, ← c79858, k2, ← c79859] at e_1297_15
  have hr := e_1297_15
  simp only [e_1297_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11903)) := by
  have c79860 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79861 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79862 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79863 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79864 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79865 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_16 := arithEq_of_rows h (row := 1297) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_16
  simp only [← c79860, ← c79861, ← c79862] at e_1297_16
  have e_1297_17 := arithEq_of_rows h (row := 1297) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_17
  simp only [← c79863, ← c79864, k2, ← c79865] at e_1297_17
  have hr := e_1297_17
  simp only [e_1297_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1297 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11904)) := by
  have c79866 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79867 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79868 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79869 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79870 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79871 := (publicBatchWrapper89_copies2495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1297_18 := arithEq_of_rows h (row := 1297) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_18
  simp only [← c79866, ← c79867, ← c79868] at e_1297_18
  have e_1297_19 := arithEq_of_rows h (row := 1297) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1297_19
  simp only [← c79869, ← c79870, k2, ← c79871] at e_1297_19
  have hr := e_1297_19
  simp only [e_1297_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11905)) := by
  have c79872 := (publicBatchWrapper89_copies2496 a h).1
  have c79873 := (publicBatchWrapper89_copies2496 a h).2.1
  have c79874 := (publicBatchWrapper89_copies2496 a h).2.2.1
  have c79875 := (publicBatchWrapper89_copies2496 a h).2.2.2.1
  have c79876 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.1
  have c79877 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_0 := arithEq_of_rows h (row := 1298) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_0
  simp only [← c79872, ← c79873, ← c79874] at e_1298_0
  have e_1298_1 := arithEq_of_rows h (row := 1298) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_1
  simp only [← c79875, ← c79876, k2, ← c79877] at e_1298_1
  have hr := e_1298_1
  simp only [e_1298_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11906)) := by
  have c79878 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.1
  have c79879 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.1
  have c79880 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.1
  have c79881 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.1
  have c79882 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79883 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_2 := arithEq_of_rows h (row := 1298) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_2
  simp only [← c79878, ← c79879, ← c79880] at e_1298_2
  have e_1298_3 := arithEq_of_rows h (row := 1298) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_3
  simp only [← c79881, ← c79882, k2, ← c79883] at e_1298_3
  have hr := e_1298_3
  simp only [e_1298_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11907)) := by
  have c79884 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79885 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79886 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79887 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79888 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79889 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_4 := arithEq_of_rows h (row := 1298) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_4
  simp only [← c79884, ← c79885, ← c79886] at e_1298_4
  have e_1298_5 := arithEq_of_rows h (row := 1298) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_5
  simp only [← c79887, ← c79888, k2, ← c79889] at e_1298_5
  have hr := e_1298_5
  simp only [e_1298_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11908)) := by
  have c79890 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79891 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79892 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79893 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79894 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79895 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_6 := arithEq_of_rows h (row := 1298) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_6
  simp only [← c79890, ← c79891, ← c79892] at e_1298_6
  have e_1298_7 := arithEq_of_rows h (row := 1298) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_7
  simp only [← c79893, ← c79894, k2, ← c79895] at e_1298_7
  have hr := e_1298_7
  simp only [e_1298_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11909)) := by
  have c79896 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79897 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79898 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79899 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79900 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79901 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_8 := arithEq_of_rows h (row := 1298) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_8
  simp only [← c79896, ← c79897, ← c79898] at e_1298_8
  have e_1298_9 := arithEq_of_rows h (row := 1298) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_9
  simp only [← c79899, ← c79900, k2, ← c79901] at e_1298_9
  have hr := e_1298_9
  simp only [e_1298_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11910)) := by
  have c79902 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79903 := (publicBatchWrapper89_copies2496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79904 := (publicBatchWrapper89_copies2497 a h).1
  have c79905 := (publicBatchWrapper89_copies2497 a h).2.1
  have c79906 := (publicBatchWrapper89_copies2497 a h).2.2.1
  have c79907 := (publicBatchWrapper89_copies2497 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_10 := arithEq_of_rows h (row := 1298) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_10
  simp only [← c79902, ← c79903, ← c79904] at e_1298_10
  have e_1298_11 := arithEq_of_rows h (row := 1298) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_11
  simp only [← c79905, ← c79906, k2, ← c79907] at e_1298_11
  have hr := e_1298_11
  simp only [e_1298_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11911)) := by
  have c79908 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.1
  have c79909 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.1
  have c79910 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.1
  have c79911 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.1
  have c79912 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.1
  have c79913 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_12 := arithEq_of_rows h (row := 1298) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_12
  simp only [← c79908, ← c79909, ← c79910] at e_1298_12
  have e_1298_13 := arithEq_of_rows h (row := 1298) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_13
  simp only [← c79911, ← c79912, k2, ← c79913] at e_1298_13
  have hr := e_1298_13
  simp only [e_1298_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11912)) := by
  have c79914 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79915 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79916 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79917 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79918 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79919 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_14 := arithEq_of_rows h (row := 1298) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_14
  simp only [← c79914, ← c79915, ← c79916] at e_1298_14
  have e_1298_15 := arithEq_of_rows h (row := 1298) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_15
  simp only [← c79917, ← c79918, k2, ← c79919] at e_1298_15
  have hr := e_1298_15
  simp only [e_1298_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11913)) := by
  have c79920 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79921 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79922 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79923 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79924 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79925 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_16 := arithEq_of_rows h (row := 1298) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_16
  simp only [← c79920, ← c79921, ← c79922] at e_1298_16
  have e_1298_17 := arithEq_of_rows h (row := 1298) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_17
  simp only [← c79923, ← c79924, k2, ← c79925] at e_1298_17
  have hr := e_1298_17
  simp only [e_1298_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1298 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11914)) := by
  have c79926 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79927 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79928 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79929 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79930 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79931 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1298_18 := arithEq_of_rows h (row := 1298) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_18
  simp only [← c79926, ← c79927, ← c79928] at e_1298_18
  have e_1298_19 := arithEq_of_rows h (row := 1298) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1298_19
  simp only [← c79929, ← c79930, k2, ← c79931] at e_1298_19
  have hr := e_1298_19
  simp only [e_1298_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11915)) := by
  have c79932 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79933 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79934 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79935 := (publicBatchWrapper89_copies2497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c79936 := (publicBatchWrapper89_copies2498 a h).1
  have c79937 := (publicBatchWrapper89_copies2498 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_0 := arithEq_of_rows h (row := 1299) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_0
  simp only [← c79932, ← c79933, ← c79934] at e_1299_0
  have e_1299_1 := arithEq_of_rows h (row := 1299) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_1
  simp only [← c79935, ← c79936, k2, ← c79937] at e_1299_1
  have hr := e_1299_1
  simp only [e_1299_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11916)) := by
  have c79938 := (publicBatchWrapper89_copies2498 a h).2.2.1
  have c79939 := (publicBatchWrapper89_copies2498 a h).2.2.2.1
  have c79940 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.1
  have c79941 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.1
  have c79942 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.1
  have c79943 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_2 := arithEq_of_rows h (row := 1299) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_2
  simp only [← c79938, ← c79939, ← c79940] at e_1299_2
  have e_1299_3 := arithEq_of_rows h (row := 1299) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_3
  simp only [← c79941, ← c79942, k2, ← c79943] at e_1299_3
  have hr := e_1299_3
  simp only [e_1299_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11917)) := by
  have c79944 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.1
  have c79945 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.1
  have c79946 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79947 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c79948 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79949 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_4 := arithEq_of_rows h (row := 1299) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_4
  simp only [← c79944, ← c79945, ← c79946] at e_1299_4
  have e_1299_5 := arithEq_of_rows h (row := 1299) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_5
  simp only [← c79947, ← c79948, k2, ← c79949] at e_1299_5
  have hr := e_1299_5
  simp only [e_1299_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11918)) := by
  have c79950 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79951 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79952 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79953 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79954 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79955 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_6 := arithEq_of_rows h (row := 1299) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_6
  simp only [← c79950, ← c79951, ← c79952] at e_1299_6
  have e_1299_7 := arithEq_of_rows h (row := 1299) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_7
  simp only [← c79953, ← c79954, k2, ← c79955] at e_1299_7
  have hr := e_1299_7
  simp only [e_1299_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11919)) := by
  have c79956 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79957 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79958 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79959 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79960 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79961 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_8 := arithEq_of_rows h (row := 1299) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_8
  simp only [← c79956, ← c79957, ← c79958] at e_1299_8
  have e_1299_9 := arithEq_of_rows h (row := 1299) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_9
  simp only [← c79959, ← c79960, k2, ← c79961] at e_1299_9
  have hr := e_1299_9
  simp only [e_1299_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11920)) := by
  have c79962 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79963 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79964 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79965 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79966 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79967 := (publicBatchWrapper89_copies2498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_10 := arithEq_of_rows h (row := 1299) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_10
  simp only [← c79962, ← c79963, ← c79964] at e_1299_10
  have e_1299_11 := arithEq_of_rows h (row := 1299) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_11
  simp only [← c79965, ← c79966, k2, ← c79967] at e_1299_11
  have hr := e_1299_11
  simp only [e_1299_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11921)) := by
  have c79968 := (publicBatchWrapper89_copies2499 a h).1
  have c79969 := (publicBatchWrapper89_copies2499 a h).2.1
  have c79970 := (publicBatchWrapper89_copies2499 a h).2.2.1
  have c79971 := (publicBatchWrapper89_copies2499 a h).2.2.2.1
  have c79972 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.1
  have c79973 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_12 := arithEq_of_rows h (row := 1299) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_12
  simp only [← c79968, ← c79969, ← c79970] at e_1299_12
  have e_1299_13 := arithEq_of_rows h (row := 1299) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_13
  simp only [← c79971, ← c79972, k2, ← c79973] at e_1299_13
  have hr := e_1299_13
  simp only [e_1299_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11922)) := by
  have c79974 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.1
  have c79975 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.1
  have c79976 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.1
  have c79977 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.1
  have c79978 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c79979 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_14 := arithEq_of_rows h (row := 1299) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_14
  simp only [← c79974, ← c79975, ← c79976] at e_1299_14
  have e_1299_15 := arithEq_of_rows h (row := 1299) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_15
  simp only [← c79977, ← c79978, k2, ← c79979] at e_1299_15
  have hr := e_1299_15
  simp only [e_1299_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11923)) := by
  have c79980 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79981 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79982 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79983 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79984 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79985 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_16 := arithEq_of_rows h (row := 1299) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_16
  simp only [← c79980, ← c79981, ← c79982] at e_1299_16
  have e_1299_17 := arithEq_of_rows h (row := 1299) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_17
  simp only [← c79983, ← c79984, k2, ← c79985] at e_1299_17
  have hr := e_1299_17
  simp only [e_1299_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1299 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11924)) := by
  have c79986 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79987 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79988 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79989 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79990 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79991 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1299_18 := arithEq_of_rows h (row := 1299) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_18
  simp only [← c79986, ← c79987, ← c79988] at e_1299_18
  have e_1299_19 := arithEq_of_rows h (row := 1299) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1299_19
  simp only [← c79989, ← c79990, k2, ← c79991] at e_1299_19
  have hr := e_1299_19
  simp only [e_1299_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11925)) := by
  have c79992 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79993 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79994 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79995 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79996 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79997 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_0 := arithEq_of_rows h (row := 1300) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_0
  simp only [← c79992, ← c79993, ← c79994] at e_1300_0
  have e_1300_1 := arithEq_of_rows h (row := 1300) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_1
  simp only [← c79995, ← c79996, k2, ← c79997] at e_1300_1
  have hr := e_1300_1
  simp only [e_1300_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11926)) := by
  have c79998 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c79999 := (publicBatchWrapper89_copies2499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80000 := (publicBatchWrapper89_copies2500 a h).1
  have c80001 := (publicBatchWrapper89_copies2500 a h).2.1
  have c80002 := (publicBatchWrapper89_copies2500 a h).2.2.1
  have c80003 := (publicBatchWrapper89_copies2500 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_2 := arithEq_of_rows h (row := 1300) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_2
  simp only [← c79998, ← c79999, ← c80000] at e_1300_2
  have e_1300_3 := arithEq_of_rows h (row := 1300) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_3
  simp only [← c80001, ← c80002, k2, ← c80003] at e_1300_3
  have hr := e_1300_3
  simp only [e_1300_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11927)) := by
  have c80004 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.1
  have c80005 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.1
  have c80006 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.1
  have c80007 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.1
  have c80008 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.1
  have c80009 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_4 := arithEq_of_rows h (row := 1300) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_4
  simp only [← c80004, ← c80005, ← c80006] at e_1300_4
  have e_1300_5 := arithEq_of_rows h (row := 1300) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_5
  simp only [← c80007, ← c80008, k2, ← c80009] at e_1300_5
  have hr := e_1300_5
  simp only [e_1300_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11928)) := by
  have c80010 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80011 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80012 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80013 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80014 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80015 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_6 := arithEq_of_rows h (row := 1300) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_6
  simp only [← c80010, ← c80011, ← c80012] at e_1300_6
  have e_1300_7 := arithEq_of_rows h (row := 1300) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_7
  simp only [← c80013, ← c80014, k2, ← c80015] at e_1300_7
  have hr := e_1300_7
  simp only [e_1300_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11929)) := by
  have c80016 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80017 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80018 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80019 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80020 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80021 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_8 := arithEq_of_rows h (row := 1300) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_8
  simp only [← c80016, ← c80017, ← c80018] at e_1300_8
  have e_1300_9 := arithEq_of_rows h (row := 1300) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_9
  simp only [← c80019, ← c80020, k2, ← c80021] at e_1300_9
  have hr := e_1300_9
  simp only [e_1300_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11930)) := by
  have c80022 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80023 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80024 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80025 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80026 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80027 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_10 := arithEq_of_rows h (row := 1300) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_10
  simp only [← c80022, ← c80023, ← c80024] at e_1300_10
  have e_1300_11 := arithEq_of_rows h (row := 1300) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_11
  simp only [← c80025, ← c80026, k2, ← c80027] at e_1300_11
  have hr := e_1300_11
  simp only [e_1300_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11931)) := by
  have c80028 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80029 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80030 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80031 := (publicBatchWrapper89_copies2500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80032 := (publicBatchWrapper89_copies2501 a h).1
  have c80033 := (publicBatchWrapper89_copies2501 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_12 := arithEq_of_rows h (row := 1300) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_12
  simp only [← c80028, ← c80029, ← c80030] at e_1300_12
  have e_1300_13 := arithEq_of_rows h (row := 1300) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_13
  simp only [← c80031, ← c80032, k2, ← c80033] at e_1300_13
  have hr := e_1300_13
  simp only [e_1300_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11932)) := by
  have c80034 := (publicBatchWrapper89_copies2501 a h).2.2.1
  have c80035 := (publicBatchWrapper89_copies2501 a h).2.2.2.1
  have c80036 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.1
  have c80037 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.1
  have c80038 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.1
  have c80039 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_14 := arithEq_of_rows h (row := 1300) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_14
  simp only [← c80034, ← c80035, ← c80036] at e_1300_14
  have e_1300_15 := arithEq_of_rows h (row := 1300) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_15
  simp only [← c80037, ← c80038, k2, ← c80039] at e_1300_15
  have hr := e_1300_15
  simp only [e_1300_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11933)) := by
  have c80040 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.1
  have c80041 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.1
  have c80042 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80043 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80044 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80045 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_16 := arithEq_of_rows h (row := 1300) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_16
  simp only [← c80040, ← c80041, ← c80042] at e_1300_16
  have e_1300_17 := arithEq_of_rows h (row := 1300) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_17
  simp only [← c80043, ← c80044, k2, ← c80045] at e_1300_17
  have hr := e_1300_17
  simp only [e_1300_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1300 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11934)) := by
  have c80046 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80047 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80048 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80049 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80050 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80051 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1300_18 := arithEq_of_rows h (row := 1300) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_18
  simp only [← c80046, ← c80047, ← c80048] at e_1300_18
  have e_1300_19 := arithEq_of_rows h (row := 1300) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1300_19
  simp only [← c80049, ← c80050, k2, ← c80051] at e_1300_19
  have hr := e_1300_19
  simp only [e_1300_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11935)) := by
  have c80052 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80053 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80054 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80055 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80056 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80057 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_0 := arithEq_of_rows h (row := 1301) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_0
  simp only [← c80052, ← c80053, ← c80054] at e_1301_0
  have e_1301_1 := arithEq_of_rows h (row := 1301) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_1
  simp only [← c80055, ← c80056, k2, ← c80057] at e_1301_1
  have hr := e_1301_1
  simp only [e_1301_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11936)) := by
  have c80058 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80059 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80060 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80061 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80062 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80063 := (publicBatchWrapper89_copies2501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_2 := arithEq_of_rows h (row := 1301) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_2
  simp only [← c80058, ← c80059, ← c80060] at e_1301_2
  have e_1301_3 := arithEq_of_rows h (row := 1301) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_3
  simp only [← c80061, ← c80062, k2, ← c80063] at e_1301_3
  have hr := e_1301_3
  simp only [e_1301_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11937)) := by
  have c80064 := (publicBatchWrapper89_copies2502 a h).1
  have c80065 := (publicBatchWrapper89_copies2502 a h).2.1
  have c80066 := (publicBatchWrapper89_copies2502 a h).2.2.1
  have c80067 := (publicBatchWrapper89_copies2502 a h).2.2.2.1
  have c80068 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.1
  have c80069 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_4 := arithEq_of_rows h (row := 1301) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_4
  simp only [← c80064, ← c80065, ← c80066] at e_1301_4
  have e_1301_5 := arithEq_of_rows h (row := 1301) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_5
  simp only [← c80067, ← c80068, k2, ← c80069] at e_1301_5
  have hr := e_1301_5
  simp only [e_1301_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11938)) := by
  have c80070 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.1
  have c80071 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.1
  have c80072 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.1
  have c80073 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.1
  have c80074 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80075 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_6 := arithEq_of_rows h (row := 1301) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_6
  simp only [← c80070, ← c80071, ← c80072] at e_1301_6
  have e_1301_7 := arithEq_of_rows h (row := 1301) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_7
  simp only [← c80073, ← c80074, k2, ← c80075] at e_1301_7
  have hr := e_1301_7
  simp only [e_1301_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11939)) := by
  have c80076 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80077 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80078 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80079 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80080 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80081 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_8 := arithEq_of_rows h (row := 1301) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_8
  simp only [← c80076, ← c80077, ← c80078] at e_1301_8
  have e_1301_9 := arithEq_of_rows h (row := 1301) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_9
  simp only [← c80079, ← c80080, k2, ← c80081] at e_1301_9
  have hr := e_1301_9
  simp only [e_1301_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11940)) := by
  have c80082 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80083 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80084 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80085 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80086 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80087 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_10 := arithEq_of_rows h (row := 1301) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_10
  simp only [← c80082, ← c80083, ← c80084] at e_1301_10
  have e_1301_11 := arithEq_of_rows h (row := 1301) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_11
  simp only [← c80085, ← c80086, k2, ← c80087] at e_1301_11
  have hr := e_1301_11
  simp only [e_1301_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11941)) := by
  have c80088 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80089 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80090 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80091 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80092 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80093 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_12 := arithEq_of_rows h (row := 1301) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_12
  simp only [← c80088, ← c80089, ← c80090] at e_1301_12
  have e_1301_13 := arithEq_of_rows h (row := 1301) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_13
  simp only [← c80091, ← c80092, k2, ← c80093] at e_1301_13
  have hr := e_1301_13
  simp only [e_1301_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11942)) := by
  have c80094 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80095 := (publicBatchWrapper89_copies2502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80096 := (publicBatchWrapper89_copies2503 a h).1
  have c80097 := (publicBatchWrapper89_copies2503 a h).2.1
  have c80098 := (publicBatchWrapper89_copies2503 a h).2.2.1
  have c80099 := (publicBatchWrapper89_copies2503 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_14 := arithEq_of_rows h (row := 1301) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_14
  simp only [← c80094, ← c80095, ← c80096] at e_1301_14
  have e_1301_15 := arithEq_of_rows h (row := 1301) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_15
  simp only [← c80097, ← c80098, k2, ← c80099] at e_1301_15
  have hr := e_1301_15
  simp only [e_1301_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11943)) := by
  have c80100 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.1
  have c80101 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.1
  have c80102 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.1
  have c80103 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.1
  have c80104 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.1
  have c80105 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_16 := arithEq_of_rows h (row := 1301) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_16
  simp only [← c80100, ← c80101, ← c80102] at e_1301_16
  have e_1301_17 := arithEq_of_rows h (row := 1301) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_17
  simp only [← c80103, ← c80104, k2, ← c80105] at e_1301_17
  have hr := e_1301_17
  simp only [e_1301_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1301 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11944)) := by
  have c80106 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80107 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80108 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80109 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80110 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80111 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1301_18 := arithEq_of_rows h (row := 1301) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_18
  simp only [← c80106, ← c80107, ← c80108] at e_1301_18
  have e_1301_19 := arithEq_of_rows h (row := 1301) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1301_19
  simp only [← c80109, ← c80110, k2, ← c80111] at e_1301_19
  have hr := e_1301_19
  simp only [e_1301_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11945)) := by
  have c80112 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80113 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80114 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80115 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80116 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80117 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_0 := arithEq_of_rows h (row := 1302) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_0
  simp only [← c80112, ← c80113, ← c80114] at e_1302_0
  have e_1302_1 := arithEq_of_rows h (row := 1302) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_1
  simp only [← c80115, ← c80116, k2, ← c80117] at e_1302_1
  have hr := e_1302_1
  simp only [e_1302_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11946)) := by
  have c80118 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80119 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80120 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80121 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80122 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80123 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_2 := arithEq_of_rows h (row := 1302) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_2
  simp only [← c80118, ← c80119, ← c80120] at e_1302_2
  have e_1302_3 := arithEq_of_rows h (row := 1302) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_3
  simp only [← c80121, ← c80122, k2, ← c80123] at e_1302_3
  have hr := e_1302_3
  simp only [e_1302_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11947)) := by
  have c80124 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80125 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80126 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80127 := (publicBatchWrapper89_copies2503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80128 := (publicBatchWrapper89_copies2504 a h).1
  have c80129 := (publicBatchWrapper89_copies2504 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_4 := arithEq_of_rows h (row := 1302) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_4
  simp only [← c80124, ← c80125, ← c80126] at e_1302_4
  have e_1302_5 := arithEq_of_rows h (row := 1302) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_5
  simp only [← c80127, ← c80128, k2, ← c80129] at e_1302_5
  have hr := e_1302_5
  simp only [e_1302_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11948)) := by
  have c80130 := (publicBatchWrapper89_copies2504 a h).2.2.1
  have c80131 := (publicBatchWrapper89_copies2504 a h).2.2.2.1
  have c80132 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.1
  have c80133 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.1
  have c80134 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.1
  have c80135 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_6 := arithEq_of_rows h (row := 1302) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_6
  simp only [← c80130, ← c80131, ← c80132] at e_1302_6
  have e_1302_7 := arithEq_of_rows h (row := 1302) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_7
  simp only [← c80133, ← c80134, k2, ← c80135] at e_1302_7
  have hr := e_1302_7
  simp only [e_1302_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11949)) := by
  have c80136 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.1
  have c80137 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.1
  have c80138 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80139 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80140 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80141 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_8 := arithEq_of_rows h (row := 1302) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_8
  simp only [← c80136, ← c80137, ← c80138] at e_1302_8
  have e_1302_9 := arithEq_of_rows h (row := 1302) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_9
  simp only [← c80139, ← c80140, k2, ← c80141] at e_1302_9
  have hr := e_1302_9
  simp only [e_1302_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11950)) := by
  have c80142 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80143 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80144 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80145 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80146 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80147 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_10 := arithEq_of_rows h (row := 1302) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_10
  simp only [← c80142, ← c80143, ← c80144] at e_1302_10
  have e_1302_11 := arithEq_of_rows h (row := 1302) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_11
  simp only [← c80145, ← c80146, k2, ← c80147] at e_1302_11
  have hr := e_1302_11
  simp only [e_1302_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11951)) := by
  have c80148 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80149 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80150 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80151 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80152 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80153 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_12 := arithEq_of_rows h (row := 1302) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_12
  simp only [← c80148, ← c80149, ← c80150] at e_1302_12
  have e_1302_13 := arithEq_of_rows h (row := 1302) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_13
  simp only [← c80151, ← c80152, k2, ← c80153] at e_1302_13
  have hr := e_1302_13
  simp only [e_1302_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11952)) := by
  have c80154 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80155 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80156 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80157 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80158 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80159 := (publicBatchWrapper89_copies2504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_14 := arithEq_of_rows h (row := 1302) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_14
  simp only [← c80154, ← c80155, ← c80156] at e_1302_14
  have e_1302_15 := arithEq_of_rows h (row := 1302) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_15
  simp only [← c80157, ← c80158, k2, ← c80159] at e_1302_15
  have hr := e_1302_15
  simp only [e_1302_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11953)) := by
  have c80160 := (publicBatchWrapper89_copies2505 a h).1
  have c80161 := (publicBatchWrapper89_copies2505 a h).2.1
  have c80162 := (publicBatchWrapper89_copies2505 a h).2.2.1
  have c80163 := (publicBatchWrapper89_copies2505 a h).2.2.2.1
  have c80164 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.1
  have c80165 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_16 := arithEq_of_rows h (row := 1302) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_16
  simp only [← c80160, ← c80161, ← c80162] at e_1302_16
  have e_1302_17 := arithEq_of_rows h (row := 1302) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_17
  simp only [← c80163, ← c80164, k2, ← c80165] at e_1302_17
  have hr := e_1302_17
  simp only [e_1302_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1302 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11954)) := by
  have c80166 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.1
  have c80167 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.1
  have c80168 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.1
  have c80169 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.1
  have c80170 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80171 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1302_18 := arithEq_of_rows h (row := 1302) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_18
  simp only [← c80166, ← c80167, ← c80168] at e_1302_18
  have e_1302_19 := arithEq_of_rows h (row := 1302) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1302_19
  simp only [← c80169, ← c80170, k2, ← c80171] at e_1302_19
  have hr := e_1302_19
  simp only [e_1302_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11955)) := by
  have c80172 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80173 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80174 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80175 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80176 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80177 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_0 := arithEq_of_rows h (row := 1303) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_0
  simp only [← c80172, ← c80173, ← c80174] at e_1303_0
  have e_1303_1 := arithEq_of_rows h (row := 1303) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_1
  simp only [← c80175, ← c80176, k2, ← c80177] at e_1303_1
  have hr := e_1303_1
  simp only [e_1303_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11956)) := by
  have c80178 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80179 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80180 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80181 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80182 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80183 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_2 := arithEq_of_rows h (row := 1303) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_2
  simp only [← c80178, ← c80179, ← c80180] at e_1303_2
  have e_1303_3 := arithEq_of_rows h (row := 1303) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_3
  simp only [← c80181, ← c80182, k2, ← c80183] at e_1303_3
  have hr := e_1303_3
  simp only [e_1303_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11957)) := by
  have c80184 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80185 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80186 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80187 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80188 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80189 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_4 := arithEq_of_rows h (row := 1303) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_4
  simp only [← c80184, ← c80185, ← c80186] at e_1303_4
  have e_1303_5 := arithEq_of_rows h (row := 1303) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_5
  simp only [← c80187, ← c80188, k2, ← c80189] at e_1303_5
  have hr := e_1303_5
  simp only [e_1303_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11958)) := by
  have c80190 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80191 := (publicBatchWrapper89_copies2505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80192 := (publicBatchWrapper89_copies2506 a h).1
  have c80193 := (publicBatchWrapper89_copies2506 a h).2.1
  have c80194 := (publicBatchWrapper89_copies2506 a h).2.2.1
  have c80195 := (publicBatchWrapper89_copies2506 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_6 := arithEq_of_rows h (row := 1303) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_6
  simp only [← c80190, ← c80191, ← c80192] at e_1303_6
  have e_1303_7 := arithEq_of_rows h (row := 1303) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_7
  simp only [← c80193, ← c80194, k2, ← c80195] at e_1303_7
  have hr := e_1303_7
  simp only [e_1303_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11959)) := by
  have c80196 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.1
  have c80197 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.1
  have c80198 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.1
  have c80199 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.1
  have c80200 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.1
  have c80201 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_8 := arithEq_of_rows h (row := 1303) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_8
  simp only [← c80196, ← c80197, ← c80198] at e_1303_8
  have e_1303_9 := arithEq_of_rows h (row := 1303) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_9
  simp only [← c80199, ← c80200, k2, ← c80201] at e_1303_9
  have hr := e_1303_9
  simp only [e_1303_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11960)) := by
  have c80202 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80203 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80204 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80205 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80206 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80207 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_10 := arithEq_of_rows h (row := 1303) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_10
  simp only [← c80202, ← c80203, ← c80204] at e_1303_10
  have e_1303_11 := arithEq_of_rows h (row := 1303) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_11
  simp only [← c80205, ← c80206, k2, ← c80207] at e_1303_11
  have hr := e_1303_11
  simp only [e_1303_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11961)) := by
  have c80208 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80209 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80210 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80211 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80212 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80213 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_12 := arithEq_of_rows h (row := 1303) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_12
  simp only [← c80208, ← c80209, ← c80210] at e_1303_12
  have e_1303_13 := arithEq_of_rows h (row := 1303) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_13
  simp only [← c80211, ← c80212, k2, ← c80213] at e_1303_13
  have hr := e_1303_13
  simp only [e_1303_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11962)) := by
  have c80214 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80215 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80216 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80217 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80218 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80219 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_14 := arithEq_of_rows h (row := 1303) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_14
  simp only [← c80214, ← c80215, ← c80216] at e_1303_14
  have e_1303_15 := arithEq_of_rows h (row := 1303) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_15
  simp only [← c80217, ← c80218, k2, ← c80219] at e_1303_15
  have hr := e_1303_15
  simp only [e_1303_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11963)) := by
  have c80220 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80221 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80222 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80223 := (publicBatchWrapper89_copies2506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80224 := (publicBatchWrapper89_copies2507 a h).1
  have c80225 := (publicBatchWrapper89_copies2507 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_16 := arithEq_of_rows h (row := 1303) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_16
  simp only [← c80220, ← c80221, ← c80222] at e_1303_16
  have e_1303_17 := arithEq_of_rows h (row := 1303) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_17
  simp only [← c80223, ← c80224, k2, ← c80225] at e_1303_17
  have hr := e_1303_17
  simp only [e_1303_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1303 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11964)) := by
  have c80226 := (publicBatchWrapper89_copies2507 a h).2.2.1
  have c80227 := (publicBatchWrapper89_copies2507 a h).2.2.2.1
  have c80228 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.1
  have c80229 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.1
  have c80230 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.1
  have c80231 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1303_18 := arithEq_of_rows h (row := 1303) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_18
  simp only [← c80226, ← c80227, ← c80228] at e_1303_18
  have e_1303_19 := arithEq_of_rows h (row := 1303) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1303_19
  simp only [← c80229, ← c80230, k2, ← c80231] at e_1303_19
  have hr := e_1303_19
  simp only [e_1303_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11965)) := by
  have c80232 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.1
  have c80233 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.1
  have c80234 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80235 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80236 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80237 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_0 := arithEq_of_rows h (row := 1304) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_0
  simp only [← c80232, ← c80233, ← c80234] at e_1304_0
  have e_1304_1 := arithEq_of_rows h (row := 1304) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_1
  simp only [← c80235, ← c80236, k2, ← c80237] at e_1304_1
  have hr := e_1304_1
  simp only [e_1304_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11966)) := by
  have c80238 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80239 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80240 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80241 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80242 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80243 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_2 := arithEq_of_rows h (row := 1304) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_2
  simp only [← c80238, ← c80239, ← c80240] at e_1304_2
  have e_1304_3 := arithEq_of_rows h (row := 1304) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_3
  simp only [← c80241, ← c80242, k2, ← c80243] at e_1304_3
  have hr := e_1304_3
  simp only [e_1304_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11967)) := by
  have c80244 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80245 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80246 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80247 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80248 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80249 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_4 := arithEq_of_rows h (row := 1304) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_4
  simp only [← c80244, ← c80245, ← c80246] at e_1304_4
  have e_1304_5 := arithEq_of_rows h (row := 1304) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_5
  simp only [← c80247, ← c80248, k2, ← c80249] at e_1304_5
  have hr := e_1304_5
  simp only [e_1304_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11968)) := by
  have c80250 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80251 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80252 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80253 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80254 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80255 := (publicBatchWrapper89_copies2507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_6 := arithEq_of_rows h (row := 1304) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_6
  simp only [← c80250, ← c80251, ← c80252] at e_1304_6
  have e_1304_7 := arithEq_of_rows h (row := 1304) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_7
  simp only [← c80253, ← c80254, k2, ← c80255] at e_1304_7
  have hr := e_1304_7
  simp only [e_1304_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11969)) := by
  have c80256 := (publicBatchWrapper89_copies2508 a h).1
  have c80257 := (publicBatchWrapper89_copies2508 a h).2.1
  have c80258 := (publicBatchWrapper89_copies2508 a h).2.2.1
  have c80259 := (publicBatchWrapper89_copies2508 a h).2.2.2.1
  have c80260 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.1
  have c80261 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_8 := arithEq_of_rows h (row := 1304) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_8
  simp only [← c80256, ← c80257, ← c80258] at e_1304_8
  have e_1304_9 := arithEq_of_rows h (row := 1304) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_9
  simp only [← c80259, ← c80260, k2, ← c80261] at e_1304_9
  have hr := e_1304_9
  simp only [e_1304_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12010)) := by
  have c80262 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.1
  have c80263 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.1
  have c80264 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.1
  have c80265 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.1
  have c80266 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80267 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_10 := arithEq_of_rows h (row := 1304) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_10
  simp only [← c80262, ← c80263, ← c80264] at e_1304_10
  have e_1304_11 := arithEq_of_rows h (row := 1304) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_11
  simp only [← c80265, ← c80266, k2, ← c80267] at e_1304_11
  have hr := e_1304_11
  simp only [e_1304_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12011)) := by
  have c80268 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80269 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80270 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80271 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80272 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80273 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_12 := arithEq_of_rows h (row := 1304) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_12
  simp only [← c80268, ← c80269, ← c80270] at e_1304_12
  have e_1304_13 := arithEq_of_rows h (row := 1304) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_13
  simp only [← c80271, ← c80272, k2, ← c80273] at e_1304_13
  have hr := e_1304_13
  simp only [e_1304_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12012)) := by
  have c80274 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80275 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80276 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80277 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80278 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80279 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_14 := arithEq_of_rows h (row := 1304) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_14
  simp only [← c80274, ← c80275, ← c80276] at e_1304_14
  have e_1304_15 := arithEq_of_rows h (row := 1304) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_15
  simp only [← c80277, ← c80278, k2, ← c80279] at e_1304_15
  have hr := e_1304_15
  simp only [e_1304_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12013)) := by
  have c80280 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80281 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80282 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80283 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80284 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80285 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_16 := arithEq_of_rows h (row := 1304) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_16
  simp only [← c80280, ← c80281, ← c80282] at e_1304_16
  have e_1304_17 := arithEq_of_rows h (row := 1304) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_17
  simp only [← c80283, ← c80284, k2, ← c80285] at e_1304_17
  have hr := e_1304_17
  simp only [e_1304_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1304 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12014)) := by
  have c80286 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80287 := (publicBatchWrapper89_copies2508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80288 := (publicBatchWrapper89_copies2509 a h).1
  have c80289 := (publicBatchWrapper89_copies2509 a h).2.1
  have c80290 := (publicBatchWrapper89_copies2509 a h).2.2.1
  have c80291 := (publicBatchWrapper89_copies2509 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1304_18 := arithEq_of_rows h (row := 1304) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_18
  simp only [← c80286, ← c80287, ← c80288] at e_1304_18
  have e_1304_19 := arithEq_of_rows h (row := 1304) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1304_19
  simp only [← c80289, ← c80290, k2, ← c80291] at e_1304_19
  have hr := e_1304_19
  simp only [e_1304_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12015)) := by
  have c80292 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.1
  have c80293 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.1
  have c80294 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.1
  have c80295 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.1
  have c80296 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.1
  have c80297 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_0 := arithEq_of_rows h (row := 1305) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_0
  simp only [← c80292, ← c80293, ← c80294] at e_1305_0
  have e_1305_1 := arithEq_of_rows h (row := 1305) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_1
  simp only [← c80295, ← c80296, k2, ← c80297] at e_1305_1
  have hr := e_1305_1
  simp only [e_1305_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12016)) := by
  have c80298 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80299 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80300 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80301 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80302 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80303 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_2 := arithEq_of_rows h (row := 1305) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_2
  simp only [← c80298, ← c80299, ← c80300] at e_1305_2
  have e_1305_3 := arithEq_of_rows h (row := 1305) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_3
  simp only [← c80301, ← c80302, k2, ← c80303] at e_1305_3
  have hr := e_1305_3
  simp only [e_1305_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12017)) := by
  have c80304 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80305 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80306 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80307 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80308 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80309 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_4 := arithEq_of_rows h (row := 1305) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_4
  simp only [← c80304, ← c80305, ← c80306] at e_1305_4
  have e_1305_5 := arithEq_of_rows h (row := 1305) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_5
  simp only [← c80307, ← c80308, k2, ← c80309] at e_1305_5
  have hr := e_1305_5
  simp only [e_1305_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12018)) := by
  have c80310 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80311 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80312 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80313 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80314 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80315 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_6 := arithEq_of_rows h (row := 1305) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_6
  simp only [← c80310, ← c80311, ← c80312] at e_1305_6
  have e_1305_7 := arithEq_of_rows h (row := 1305) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_7
  simp only [← c80313, ← c80314, k2, ← c80315] at e_1305_7
  have hr := e_1305_7
  simp only [e_1305_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12019)) := by
  have c80316 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80317 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80318 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80319 := (publicBatchWrapper89_copies2509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80320 := (publicBatchWrapper89_copies2510 a h).1
  have c80321 := (publicBatchWrapper89_copies2510 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_8 := arithEq_of_rows h (row := 1305) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_8
  simp only [← c80316, ← c80317, ← c80318] at e_1305_8
  have e_1305_9 := arithEq_of_rows h (row := 1305) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_9
  simp only [← c80319, ← c80320, k2, ← c80321] at e_1305_9
  have hr := e_1305_9
  simp only [e_1305_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12020)) := by
  have c80322 := (publicBatchWrapper89_copies2510 a h).2.2.1
  have c80323 := (publicBatchWrapper89_copies2510 a h).2.2.2.1
  have c80324 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.1
  have c80325 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.1
  have c80326 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.1
  have c80327 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_10 := arithEq_of_rows h (row := 1305) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_10
  simp only [← c80322, ← c80323, ← c80324] at e_1305_10
  have e_1305_11 := arithEq_of_rows h (row := 1305) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_11
  simp only [← c80325, ← c80326, k2, ← c80327] at e_1305_11
  have hr := e_1305_11
  simp only [e_1305_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12021)) := by
  have c80328 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.1
  have c80329 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.1
  have c80330 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80331 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80332 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80333 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_12 := arithEq_of_rows h (row := 1305) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_12
  simp only [← c80328, ← c80329, ← c80330] at e_1305_12
  have e_1305_13 := arithEq_of_rows h (row := 1305) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_13
  simp only [← c80331, ← c80332, k2, ← c80333] at e_1305_13
  have hr := e_1305_13
  simp only [e_1305_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12022)) := by
  have c80334 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80335 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80336 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80337 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80338 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80339 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_14 := arithEq_of_rows h (row := 1305) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_14
  simp only [← c80334, ← c80335, ← c80336] at e_1305_14
  have e_1305_15 := arithEq_of_rows h (row := 1305) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_15
  simp only [← c80337, ← c80338, k2, ← c80339] at e_1305_15
  have hr := e_1305_15
  simp only [e_1305_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12023)) := by
  have c80340 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80341 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80342 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80343 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80344 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80345 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_16 := arithEq_of_rows h (row := 1305) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_16
  simp only [← c80340, ← c80341, ← c80342] at e_1305_16
  have e_1305_17 := arithEq_of_rows h (row := 1305) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_17
  simp only [← c80343, ← c80344, k2, ← c80345] at e_1305_17
  have hr := e_1305_17
  simp only [e_1305_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1305 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12024)) := by
  have c80346 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80347 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80348 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80349 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80350 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80351 := (publicBatchWrapper89_copies2510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1305_18 := arithEq_of_rows h (row := 1305) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_18
  simp only [← c80346, ← c80347, ← c80348] at e_1305_18
  have e_1305_19 := arithEq_of_rows h (row := 1305) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1305_19
  simp only [← c80349, ← c80350, k2, ← c80351] at e_1305_19
  have hr := e_1305_19
  simp only [e_1305_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12025)) := by
  have c80352 := (publicBatchWrapper89_copies2511 a h).1
  have c80353 := (publicBatchWrapper89_copies2511 a h).2.1
  have c80354 := (publicBatchWrapper89_copies2511 a h).2.2.1
  have c80355 := (publicBatchWrapper89_copies2511 a h).2.2.2.1
  have c80356 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.1
  have c80357 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_0 := arithEq_of_rows h (row := 1306) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_0
  simp only [← c80352, ← c80353, ← c80354] at e_1306_0
  have e_1306_1 := arithEq_of_rows h (row := 1306) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_1
  simp only [← c80355, ← c80356, k2, ← c80357] at e_1306_1
  have hr := e_1306_1
  simp only [e_1306_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12026)) := by
  have c80358 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.1
  have c80359 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.1
  have c80360 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.1
  have c80361 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.1
  have c80362 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80363 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_2 := arithEq_of_rows h (row := 1306) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_2
  simp only [← c80358, ← c80359, ← c80360] at e_1306_2
  have e_1306_3 := arithEq_of_rows h (row := 1306) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_3
  simp only [← c80361, ← c80362, k2, ← c80363] at e_1306_3
  have hr := e_1306_3
  simp only [e_1306_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12027)) := by
  have c80364 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80365 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80366 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80367 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80368 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80369 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_4 := arithEq_of_rows h (row := 1306) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_4
  simp only [← c80364, ← c80365, ← c80366] at e_1306_4
  have e_1306_5 := arithEq_of_rows h (row := 1306) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_5
  simp only [← c80367, ← c80368, k2, ← c80369] at e_1306_5
  have hr := e_1306_5
  simp only [e_1306_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12028)) := by
  have c80370 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80371 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80372 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80373 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80374 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80375 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_6 := arithEq_of_rows h (row := 1306) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_6
  simp only [← c80370, ← c80371, ← c80372] at e_1306_6
  have e_1306_7 := arithEq_of_rows h (row := 1306) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_7
  simp only [← c80373, ← c80374, k2, ← c80375] at e_1306_7
  have hr := e_1306_7
  simp only [e_1306_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12029)) := by
  have c80376 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80377 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80378 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80379 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80380 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80381 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_8 := arithEq_of_rows h (row := 1306) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_8
  simp only [← c80376, ← c80377, ← c80378] at e_1306_8
  have e_1306_9 := arithEq_of_rows h (row := 1306) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_9
  simp only [← c80379, ← c80380, k2, ← c80381] at e_1306_9
  have hr := e_1306_9
  simp only [e_1306_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12030)) := by
  have c80382 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80383 := (publicBatchWrapper89_copies2511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80384 := (publicBatchWrapper89_copies2512 a h).1
  have c80385 := (publicBatchWrapper89_copies2512 a h).2.1
  have c80386 := (publicBatchWrapper89_copies2512 a h).2.2.1
  have c80387 := (publicBatchWrapper89_copies2512 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_10 := arithEq_of_rows h (row := 1306) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_10
  simp only [← c80382, ← c80383, ← c80384] at e_1306_10
  have e_1306_11 := arithEq_of_rows h (row := 1306) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_11
  simp only [← c80385, ← c80386, k2, ← c80387] at e_1306_11
  have hr := e_1306_11
  simp only [e_1306_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12031)) := by
  have c80388 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.1
  have c80389 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.1
  have c80390 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.1
  have c80391 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.1
  have c80392 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.1
  have c80393 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_12 := arithEq_of_rows h (row := 1306) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_12
  simp only [← c80388, ← c80389, ← c80390] at e_1306_12
  have e_1306_13 := arithEq_of_rows h (row := 1306) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_13
  simp only [← c80391, ← c80392, k2, ← c80393] at e_1306_13
  have hr := e_1306_13
  simp only [e_1306_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12032)) := by
  have c80394 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80395 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80396 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80397 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80398 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80399 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_14 := arithEq_of_rows h (row := 1306) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_14
  simp only [← c80394, ← c80395, ← c80396] at e_1306_14
  have e_1306_15 := arithEq_of_rows h (row := 1306) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_15
  simp only [← c80397, ← c80398, k2, ← c80399] at e_1306_15
  have hr := e_1306_15
  simp only [e_1306_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12033)) := by
  have c80400 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80401 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80402 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80403 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80404 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80405 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_16 := arithEq_of_rows h (row := 1306) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_16
  simp only [← c80400, ← c80401, ← c80402] at e_1306_16
  have e_1306_17 := arithEq_of_rows h (row := 1306) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_17
  simp only [← c80403, ← c80404, k2, ← c80405] at e_1306_17
  have hr := e_1306_17
  simp only [e_1306_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1306 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12034)) := by
  have c80406 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80407 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80408 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80409 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80410 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80411 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1306_18 := arithEq_of_rows h (row := 1306) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_18
  simp only [← c80406, ← c80407, ← c80408] at e_1306_18
  have e_1306_19 := arithEq_of_rows h (row := 1306) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1306_19
  simp only [← c80409, ← c80410, k2, ← c80411] at e_1306_19
  have hr := e_1306_19
  simp only [e_1306_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12035)) := by
  have c80412 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80413 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80414 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80415 := (publicBatchWrapper89_copies2512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80416 := (publicBatchWrapper89_copies2513 a h).1
  have c80417 := (publicBatchWrapper89_copies2513 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_0 := arithEq_of_rows h (row := 1307) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_0
  simp only [← c80412, ← c80413, ← c80414] at e_1307_0
  have e_1307_1 := arithEq_of_rows h (row := 1307) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_1
  simp only [← c80415, ← c80416, k2, ← c80417] at e_1307_1
  have hr := e_1307_1
  simp only [e_1307_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12036)) := by
  have c80418 := (publicBatchWrapper89_copies2513 a h).2.2.1
  have c80419 := (publicBatchWrapper89_copies2513 a h).2.2.2.1
  have c80420 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.1
  have c80421 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.1
  have c80422 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.1
  have c80423 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_2 := arithEq_of_rows h (row := 1307) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_2
  simp only [← c80418, ← c80419, ← c80420] at e_1307_2
  have e_1307_3 := arithEq_of_rows h (row := 1307) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_3
  simp only [← c80421, ← c80422, k2, ← c80423] at e_1307_3
  have hr := e_1307_3
  simp only [e_1307_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12037)) := by
  have c80424 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.1
  have c80425 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.1
  have c80426 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80427 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80428 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80429 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_4 := arithEq_of_rows h (row := 1307) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_4
  simp only [← c80424, ← c80425, ← c80426] at e_1307_4
  have e_1307_5 := arithEq_of_rows h (row := 1307) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_5
  simp only [← c80427, ← c80428, k2, ← c80429] at e_1307_5
  have hr := e_1307_5
  simp only [e_1307_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12038)) := by
  have c80430 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80431 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80432 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80433 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80434 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80435 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_6 := arithEq_of_rows h (row := 1307) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_6
  simp only [← c80430, ← c80431, ← c80432] at e_1307_6
  have e_1307_7 := arithEq_of_rows h (row := 1307) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_7
  simp only [← c80433, ← c80434, k2, ← c80435] at e_1307_7
  have hr := e_1307_7
  simp only [e_1307_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12039)) := by
  have c80436 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80437 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80438 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80439 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80440 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80441 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_8 := arithEq_of_rows h (row := 1307) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_8
  simp only [← c80436, ← c80437, ← c80438] at e_1307_8
  have e_1307_9 := arithEq_of_rows h (row := 1307) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_9
  simp only [← c80439, ← c80440, k2, ← c80441] at e_1307_9
  have hr := e_1307_9
  simp only [e_1307_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12040)) := by
  have c80442 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80443 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80444 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80445 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80446 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80447 := (publicBatchWrapper89_copies2513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_10 := arithEq_of_rows h (row := 1307) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_10
  simp only [← c80442, ← c80443, ← c80444] at e_1307_10
  have e_1307_11 := arithEq_of_rows h (row := 1307) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_11
  simp only [← c80445, ← c80446, k2, ← c80447] at e_1307_11
  have hr := e_1307_11
  simp only [e_1307_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12041)) := by
  have c80448 := (publicBatchWrapper89_copies2514 a h).1
  have c80449 := (publicBatchWrapper89_copies2514 a h).2.1
  have c80450 := (publicBatchWrapper89_copies2514 a h).2.2.1
  have c80451 := (publicBatchWrapper89_copies2514 a h).2.2.2.1
  have c80452 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.1
  have c80453 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_12 := arithEq_of_rows h (row := 1307) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_12
  simp only [← c80448, ← c80449, ← c80450] at e_1307_12
  have e_1307_13 := arithEq_of_rows h (row := 1307) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_13
  simp only [← c80451, ← c80452, k2, ← c80453] at e_1307_13
  have hr := e_1307_13
  simp only [e_1307_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12042)) := by
  have c80454 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.1
  have c80455 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.1
  have c80456 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.1
  have c80457 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.1
  have c80458 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80459 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_14 := arithEq_of_rows h (row := 1307) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_14
  simp only [← c80454, ← c80455, ← c80456] at e_1307_14
  have e_1307_15 := arithEq_of_rows h (row := 1307) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_15
  simp only [← c80457, ← c80458, k2, ← c80459] at e_1307_15
  have hr := e_1307_15
  simp only [e_1307_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12043)) := by
  have c80460 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80461 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80462 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80463 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80464 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80465 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_16 := arithEq_of_rows h (row := 1307) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_16
  simp only [← c80460, ← c80461, ← c80462] at e_1307_16
  have e_1307_17 := arithEq_of_rows h (row := 1307) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_17
  simp only [← c80463, ← c80464, k2, ← c80465] at e_1307_17
  have hr := e_1307_17
  simp only [e_1307_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1307 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12044)) := by
  have c80466 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80467 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80468 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80469 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80470 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80471 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1307_18 := arithEq_of_rows h (row := 1307) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_18
  simp only [← c80466, ← c80467, ← c80468] at e_1307_18
  have e_1307_19 := arithEq_of_rows h (row := 1307) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1307_19
  simp only [← c80469, ← c80470, k2, ← c80471] at e_1307_19
  have hr := e_1307_19
  simp only [e_1307_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12045)) := by
  have c80472 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80473 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80474 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80475 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80476 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80477 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_0 := arithEq_of_rows h (row := 1308) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_0
  simp only [← c80472, ← c80473, ← c80474] at e_1308_0
  have e_1308_1 := arithEq_of_rows h (row := 1308) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_1
  simp only [← c80475, ← c80476, k2, ← c80477] at e_1308_1
  have hr := e_1308_1
  simp only [e_1308_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12046)) := by
  have c80478 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80479 := (publicBatchWrapper89_copies2514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80480 := (publicBatchWrapper89_copies2515 a h).1
  have c80481 := (publicBatchWrapper89_copies2515 a h).2.1
  have c80482 := (publicBatchWrapper89_copies2515 a h).2.2.1
  have c80483 := (publicBatchWrapper89_copies2515 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_2 := arithEq_of_rows h (row := 1308) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_2
  simp only [← c80478, ← c80479, ← c80480] at e_1308_2
  have e_1308_3 := arithEq_of_rows h (row := 1308) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_3
  simp only [← c80481, ← c80482, k2, ← c80483] at e_1308_3
  have hr := e_1308_3
  simp only [e_1308_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12047)) := by
  have c80484 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.1
  have c80485 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.1
  have c80486 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.1
  have c80487 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.1
  have c80488 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.1
  have c80489 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_4 := arithEq_of_rows h (row := 1308) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_4
  simp only [← c80484, ← c80485, ← c80486] at e_1308_4
  have e_1308_5 := arithEq_of_rows h (row := 1308) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_5
  simp only [← c80487, ← c80488, k2, ← c80489] at e_1308_5
  have hr := e_1308_5
  simp only [e_1308_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12048)) := by
  have c80490 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80491 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80492 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80493 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80494 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80495 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_6 := arithEq_of_rows h (row := 1308) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_6
  simp only [← c80490, ← c80491, ← c80492] at e_1308_6
  have e_1308_7 := arithEq_of_rows h (row := 1308) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_7
  simp only [← c80493, ← c80494, k2, ← c80495] at e_1308_7
  have hr := e_1308_7
  simp only [e_1308_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12049)) := by
  have c80496 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80497 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80498 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80499 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80500 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80501 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_8 := arithEq_of_rows h (row := 1308) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_8
  simp only [← c80496, ← c80497, ← c80498] at e_1308_8
  have e_1308_9 := arithEq_of_rows h (row := 1308) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_9
  simp only [← c80499, ← c80500, k2, ← c80501] at e_1308_9
  have hr := e_1308_9
  simp only [e_1308_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12050)) := by
  have c80502 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80503 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80504 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80505 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80506 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80507 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_10 := arithEq_of_rows h (row := 1308) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_10
  simp only [← c80502, ← c80503, ← c80504] at e_1308_10
  have e_1308_11 := arithEq_of_rows h (row := 1308) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_11
  simp only [← c80505, ← c80506, k2, ← c80507] at e_1308_11
  have hr := e_1308_11
  simp only [e_1308_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12051)) := by
  have c80508 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80509 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80510 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80511 := (publicBatchWrapper89_copies2515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80512 := (publicBatchWrapper89_copies2516 a h).1
  have c80513 := (publicBatchWrapper89_copies2516 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_12 := arithEq_of_rows h (row := 1308) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_12
  simp only [← c80508, ← c80509, ← c80510] at e_1308_12
  have e_1308_13 := arithEq_of_rows h (row := 1308) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_13
  simp only [← c80511, ← c80512, k2, ← c80513] at e_1308_13
  have hr := e_1308_13
  simp only [e_1308_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12052)) := by
  have c80514 := (publicBatchWrapper89_copies2516 a h).2.2.1
  have c80515 := (publicBatchWrapper89_copies2516 a h).2.2.2.1
  have c80516 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.1
  have c80517 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.1
  have c80518 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.1
  have c80519 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_14 := arithEq_of_rows h (row := 1308) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_14
  simp only [← c80514, ← c80515, ← c80516] at e_1308_14
  have e_1308_15 := arithEq_of_rows h (row := 1308) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_15
  simp only [← c80517, ← c80518, k2, ← c80519] at e_1308_15
  have hr := e_1308_15
  simp only [e_1308_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12053)) := by
  have c80520 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.1
  have c80521 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.1
  have c80522 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80523 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80524 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80525 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_16 := arithEq_of_rows h (row := 1308) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_16
  simp only [← c80520, ← c80521, ← c80522] at e_1308_16
  have e_1308_17 := arithEq_of_rows h (row := 1308) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_17
  simp only [← c80523, ← c80524, k2, ← c80525] at e_1308_17
  have hr := e_1308_17
  simp only [e_1308_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1308 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12054)) := by
  have c80526 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80527 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80528 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80529 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80530 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80531 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1308_18 := arithEq_of_rows h (row := 1308) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_18
  simp only [← c80526, ← c80527, ← c80528] at e_1308_18
  have e_1308_19 := arithEq_of_rows h (row := 1308) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1308_19
  simp only [← c80529, ← c80530, k2, ← c80531] at e_1308_19
  have hr := e_1308_19
  simp only [e_1308_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12055)) := by
  have c80532 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80533 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80534 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80535 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80536 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80537 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_0 := arithEq_of_rows h (row := 1309) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_0
  simp only [← c80532, ← c80533, ← c80534] at e_1309_0
  have e_1309_1 := arithEq_of_rows h (row := 1309) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_1
  simp only [← c80535, ← c80536, k2, ← c80537] at e_1309_1
  have hr := e_1309_1
  simp only [e_1309_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12056)) := by
  have c80538 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80539 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80540 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80541 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80542 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80543 := (publicBatchWrapper89_copies2516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_2 := arithEq_of_rows h (row := 1309) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_2
  simp only [← c80538, ← c80539, ← c80540] at e_1309_2
  have e_1309_3 := arithEq_of_rows h (row := 1309) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_3
  simp only [← c80541, ← c80542, k2, ← c80543] at e_1309_3
  have hr := e_1309_3
  simp only [e_1309_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12057)) := by
  have c80544 := (publicBatchWrapper89_copies2517 a h).1
  have c80545 := (publicBatchWrapper89_copies2517 a h).2.1
  have c80546 := (publicBatchWrapper89_copies2517 a h).2.2.1
  have c80547 := (publicBatchWrapper89_copies2517 a h).2.2.2.1
  have c80548 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.1
  have c80549 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_4 := arithEq_of_rows h (row := 1309) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_4
  simp only [← c80544, ← c80545, ← c80546] at e_1309_4
  have e_1309_5 := arithEq_of_rows h (row := 1309) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_5
  simp only [← c80547, ← c80548, k2, ← c80549] at e_1309_5
  have hr := e_1309_5
  simp only [e_1309_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12058)) := by
  have c80550 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.1
  have c80551 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.1
  have c80552 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.1
  have c80553 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.1
  have c80554 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80555 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_6 := arithEq_of_rows h (row := 1309) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_6
  simp only [← c80550, ← c80551, ← c80552] at e_1309_6
  have e_1309_7 := arithEq_of_rows h (row := 1309) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_7
  simp only [← c80553, ← c80554, k2, ← c80555] at e_1309_7
  have hr := e_1309_7
  simp only [e_1309_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12059)) := by
  have c80556 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80557 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80558 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80559 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80560 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80561 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_8 := arithEq_of_rows h (row := 1309) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_8
  simp only [← c80556, ← c80557, ← c80558] at e_1309_8
  have e_1309_9 := arithEq_of_rows h (row := 1309) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_9
  simp only [← c80559, ← c80560, k2, ← c80561] at e_1309_9
  have hr := e_1309_9
  simp only [e_1309_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12060)) := by
  have c80562 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80563 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80564 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80565 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80566 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80567 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_10 := arithEq_of_rows h (row := 1309) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_10
  simp only [← c80562, ← c80563, ← c80564] at e_1309_10
  have e_1309_11 := arithEq_of_rows h (row := 1309) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_11
  simp only [← c80565, ← c80566, k2, ← c80567] at e_1309_11
  have hr := e_1309_11
  simp only [e_1309_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12061)) := by
  have c80568 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80569 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80570 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80571 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80572 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80573 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_12 := arithEq_of_rows h (row := 1309) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_12
  simp only [← c80568, ← c80569, ← c80570] at e_1309_12
  have e_1309_13 := arithEq_of_rows h (row := 1309) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_13
  simp only [← c80571, ← c80572, k2, ← c80573] at e_1309_13
  have hr := e_1309_13
  simp only [e_1309_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12062)) := by
  have c80574 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80575 := (publicBatchWrapper89_copies2517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80576 := (publicBatchWrapper89_copies2518 a h).1
  have c80577 := (publicBatchWrapper89_copies2518 a h).2.1
  have c80578 := (publicBatchWrapper89_copies2518 a h).2.2.1
  have c80579 := (publicBatchWrapper89_copies2518 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_14 := arithEq_of_rows h (row := 1309) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_14
  simp only [← c80574, ← c80575, ← c80576] at e_1309_14
  have e_1309_15 := arithEq_of_rows h (row := 1309) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_15
  simp only [← c80577, ← c80578, k2, ← c80579] at e_1309_15
  have hr := e_1309_15
  simp only [e_1309_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12063)) := by
  have c80580 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.1
  have c80581 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.1
  have c80582 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.1
  have c80583 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.1
  have c80584 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.1
  have c80585 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_16 := arithEq_of_rows h (row := 1309) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_16
  simp only [← c80580, ← c80581, ← c80582] at e_1309_16
  have e_1309_17 := arithEq_of_rows h (row := 1309) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_17
  simp only [← c80583, ← c80584, k2, ← c80585] at e_1309_17
  have hr := e_1309_17
  simp only [e_1309_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1309 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12064)) := by
  have c80586 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80587 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80588 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80589 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80590 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80591 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1309_18 := arithEq_of_rows h (row := 1309) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_18
  simp only [← c80586, ← c80587, ← c80588] at e_1309_18
  have e_1309_19 := arithEq_of_rows h (row := 1309) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1309_19
  simp only [← c80589, ← c80590, k2, ← c80591] at e_1309_19
  have hr := e_1309_19
  simp only [e_1309_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12065)) := by
  have c80592 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80593 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80594 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80595 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80596 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80597 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_0 := arithEq_of_rows h (row := 1310) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_0
  simp only [← c80592, ← c80593, ← c80594] at e_1310_0
  have e_1310_1 := arithEq_of_rows h (row := 1310) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_1
  simp only [← c80595, ← c80596, k2, ← c80597] at e_1310_1
  have hr := e_1310_1
  simp only [e_1310_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12066)) := by
  have c80598 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80599 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80600 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80601 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80602 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80603 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_2 := arithEq_of_rows h (row := 1310) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_2
  simp only [← c80598, ← c80599, ← c80600] at e_1310_2
  have e_1310_3 := arithEq_of_rows h (row := 1310) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_3
  simp only [← c80601, ← c80602, k2, ← c80603] at e_1310_3
  have hr := e_1310_3
  simp only [e_1310_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12067)) := by
  have c80604 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80605 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80606 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80607 := (publicBatchWrapper89_copies2518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80608 := (publicBatchWrapper89_copies2519 a h).1
  have c80609 := (publicBatchWrapper89_copies2519 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_4 := arithEq_of_rows h (row := 1310) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_4
  simp only [← c80604, ← c80605, ← c80606] at e_1310_4
  have e_1310_5 := arithEq_of_rows h (row := 1310) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_5
  simp only [← c80607, ← c80608, k2, ← c80609] at e_1310_5
  have hr := e_1310_5
  simp only [e_1310_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12068)) := by
  have c80610 := (publicBatchWrapper89_copies2519 a h).2.2.1
  have c80611 := (publicBatchWrapper89_copies2519 a h).2.2.2.1
  have c80612 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.1
  have c80613 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.1
  have c80614 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.1
  have c80615 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_6 := arithEq_of_rows h (row := 1310) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_6
  simp only [← c80610, ← c80611, ← c80612] at e_1310_6
  have e_1310_7 := arithEq_of_rows h (row := 1310) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_7
  simp only [← c80613, ← c80614, k2, ← c80615] at e_1310_7
  have hr := e_1310_7
  simp only [e_1310_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12069)) := by
  have c80616 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.1
  have c80617 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.1
  have c80618 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80619 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80620 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80621 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_8 := arithEq_of_rows h (row := 1310) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_8
  simp only [← c80616, ← c80617, ← c80618] at e_1310_8
  have e_1310_9 := arithEq_of_rows h (row := 1310) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_9
  simp only [← c80619, ← c80620, k2, ← c80621] at e_1310_9
  have hr := e_1310_9
  simp only [e_1310_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12070)) := by
  have c80622 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80623 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80624 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80625 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80626 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80627 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_10 := arithEq_of_rows h (row := 1310) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_10
  simp only [← c80622, ← c80623, ← c80624] at e_1310_10
  have e_1310_11 := arithEq_of_rows h (row := 1310) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_11
  simp only [← c80625, ← c80626, k2, ← c80627] at e_1310_11
  have hr := e_1310_11
  simp only [e_1310_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12071)) := by
  have c80628 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80629 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80630 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80631 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80632 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80633 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_12 := arithEq_of_rows h (row := 1310) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_12
  simp only [← c80628, ← c80629, ← c80630] at e_1310_12
  have e_1310_13 := arithEq_of_rows h (row := 1310) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_13
  simp only [← c80631, ← c80632, k2, ← c80633] at e_1310_13
  have hr := e_1310_13
  simp only [e_1310_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12072)) := by
  have c80634 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80635 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80636 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80637 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80638 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80639 := (publicBatchWrapper89_copies2519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_14 := arithEq_of_rows h (row := 1310) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_14
  simp only [← c80634, ← c80635, ← c80636] at e_1310_14
  have e_1310_15 := arithEq_of_rows h (row := 1310) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_15
  simp only [← c80637, ← c80638, k2, ← c80639] at e_1310_15
  have hr := e_1310_15
  simp only [e_1310_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12073)) := by
  have c80640 := (publicBatchWrapper89_copies2520 a h).1
  have c80641 := (publicBatchWrapper89_copies2520 a h).2.1
  have c80642 := (publicBatchWrapper89_copies2520 a h).2.2.1
  have c80643 := (publicBatchWrapper89_copies2520 a h).2.2.2.1
  have c80644 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.1
  have c80645 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_16 := arithEq_of_rows h (row := 1310) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_16
  simp only [← c80640, ← c80641, ← c80642] at e_1310_16
  have e_1310_17 := arithEq_of_rows h (row := 1310) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_17
  simp only [← c80643, ← c80644, k2, ← c80645] at e_1310_17
  have hr := e_1310_17
  simp only [e_1310_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1310 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12074)) := by
  have c80646 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.1
  have c80647 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.1
  have c80648 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.1
  have c80649 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.1
  have c80650 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80651 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1310_18 := arithEq_of_rows h (row := 1310) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_18
  simp only [← c80646, ← c80647, ← c80648] at e_1310_18
  have e_1310_19 := arithEq_of_rows h (row := 1310) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1310_19
  simp only [← c80649, ← c80650, k2, ← c80651] at e_1310_19
  have hr := e_1310_19
  simp only [e_1310_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12075)) := by
  have c80652 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80653 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80654 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80655 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80656 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80657 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_0 := arithEq_of_rows h (row := 1311) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_0
  simp only [← c80652, ← c80653, ← c80654] at e_1311_0
  have e_1311_1 := arithEq_of_rows h (row := 1311) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_1
  simp only [← c80655, ← c80656, k2, ← c80657] at e_1311_1
  have hr := e_1311_1
  simp only [e_1311_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12076)) := by
  have c80658 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80659 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80660 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80661 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80662 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80663 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_2 := arithEq_of_rows h (row := 1311) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_2
  simp only [← c80658, ← c80659, ← c80660] at e_1311_2
  have e_1311_3 := arithEq_of_rows h (row := 1311) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_3
  simp only [← c80661, ← c80662, k2, ← c80663] at e_1311_3
  have hr := e_1311_3
  simp only [e_1311_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12077)) := by
  have c80664 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80665 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80666 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80667 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80668 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80669 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_4 := arithEq_of_rows h (row := 1311) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_4
  simp only [← c80664, ← c80665, ← c80666] at e_1311_4
  have e_1311_5 := arithEq_of_rows h (row := 1311) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_5
  simp only [← c80667, ← c80668, k2, ← c80669] at e_1311_5
  have hr := e_1311_5
  simp only [e_1311_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12078)) := by
  have c80670 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80671 := (publicBatchWrapper89_copies2520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80672 := (publicBatchWrapper89_copies2521 a h).1
  have c80673 := (publicBatchWrapper89_copies2521 a h).2.1
  have c80674 := (publicBatchWrapper89_copies2521 a h).2.2.1
  have c80675 := (publicBatchWrapper89_copies2521 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_6 := arithEq_of_rows h (row := 1311) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_6
  simp only [← c80670, ← c80671, ← c80672] at e_1311_6
  have e_1311_7 := arithEq_of_rows h (row := 1311) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_7
  simp only [← c80673, ← c80674, k2, ← c80675] at e_1311_7
  have hr := e_1311_7
  simp only [e_1311_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12079)) := by
  have c80676 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.1
  have c80677 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.1
  have c80678 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.1
  have c80679 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.1
  have c80680 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.1
  have c80681 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_8 := arithEq_of_rows h (row := 1311) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_8
  simp only [← c80676, ← c80677, ← c80678] at e_1311_8
  have e_1311_9 := arithEq_of_rows h (row := 1311) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_9
  simp only [← c80679, ← c80680, k2, ← c80681] at e_1311_9
  have hr := e_1311_9
  simp only [e_1311_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12080)) := by
  have c80682 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80683 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80684 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80685 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80686 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80687 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_10 := arithEq_of_rows h (row := 1311) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_10
  simp only [← c80682, ← c80683, ← c80684] at e_1311_10
  have e_1311_11 := arithEq_of_rows h (row := 1311) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_11
  simp only [← c80685, ← c80686, k2, ← c80687] at e_1311_11
  have hr := e_1311_11
  simp only [e_1311_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12081)) := by
  have c80688 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80689 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80690 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80691 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80692 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80693 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_12 := arithEq_of_rows h (row := 1311) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_12
  simp only [← c80688, ← c80689, ← c80690] at e_1311_12
  have e_1311_13 := arithEq_of_rows h (row := 1311) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_13
  simp only [← c80691, ← c80692, k2, ← c80693] at e_1311_13
  have hr := e_1311_13
  simp only [e_1311_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12082)) := by
  have c80694 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80695 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80696 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80697 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80698 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80699 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_14 := arithEq_of_rows h (row := 1311) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_14
  simp only [← c80694, ← c80695, ← c80696] at e_1311_14
  have e_1311_15 := arithEq_of_rows h (row := 1311) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_15
  simp only [← c80697, ← c80698, k2, ← c80699] at e_1311_15
  have hr := e_1311_15
  simp only [e_1311_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12083)) := by
  have c80700 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80701 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80702 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80703 := (publicBatchWrapper89_copies2521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80704 := (publicBatchWrapper89_copies2522 a h).1
  have c80705 := (publicBatchWrapper89_copies2522 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_16 := arithEq_of_rows h (row := 1311) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_16
  simp only [← c80700, ← c80701, ← c80702] at e_1311_16
  have e_1311_17 := arithEq_of_rows h (row := 1311) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_17
  simp only [← c80703, ← c80704, k2, ← c80705] at e_1311_17
  have hr := e_1311_17
  simp only [e_1311_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1311 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12084)) := by
  have c80706 := (publicBatchWrapper89_copies2522 a h).2.2.1
  have c80707 := (publicBatchWrapper89_copies2522 a h).2.2.2.1
  have c80708 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.1
  have c80709 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.1
  have c80710 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.1
  have c80711 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1311_18 := arithEq_of_rows h (row := 1311) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_18
  simp only [← c80706, ← c80707, ← c80708] at e_1311_18
  have e_1311_19 := arithEq_of_rows h (row := 1311) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1311_19
  simp only [← c80709, ← c80710, k2, ← c80711] at e_1311_19
  have hr := e_1311_19
  simp only [e_1311_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12085)) := by
  have c80712 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.1
  have c80713 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.1
  have c80714 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80715 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80716 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80717 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_0 := arithEq_of_rows h (row := 1312) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_0
  simp only [← c80712, ← c80713, ← c80714] at e_1312_0
  have e_1312_1 := arithEq_of_rows h (row := 1312) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_1
  simp only [← c80715, ← c80716, k2, ← c80717] at e_1312_1
  have hr := e_1312_1
  simp only [e_1312_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12086)) := by
  have c80718 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80719 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80720 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80721 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80722 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80723 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_2 := arithEq_of_rows h (row := 1312) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_2
  simp only [← c80718, ← c80719, ← c80720] at e_1312_2
  have e_1312_3 := arithEq_of_rows h (row := 1312) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_3
  simp only [← c80721, ← c80722, k2, ← c80723] at e_1312_3
  have hr := e_1312_3
  simp only [e_1312_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12087)) := by
  have c80724 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80725 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80726 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80727 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80728 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80729 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_4 := arithEq_of_rows h (row := 1312) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_4
  simp only [← c80724, ← c80725, ← c80726] at e_1312_4
  have e_1312_5 := arithEq_of_rows h (row := 1312) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_5
  simp only [← c80727, ← c80728, k2, ← c80729] at e_1312_5
  have hr := e_1312_5
  simp only [e_1312_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12088)) := by
  have c80730 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80731 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80732 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80733 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80734 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80735 := (publicBatchWrapper89_copies2522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_6 := arithEq_of_rows h (row := 1312) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_6
  simp only [← c80730, ← c80731, ← c80732] at e_1312_6
  have e_1312_7 := arithEq_of_rows h (row := 1312) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_7
  simp only [← c80733, ← c80734, k2, ← c80735] at e_1312_7
  have hr := e_1312_7
  simp only [e_1312_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12089)) := by
  have c80736 := (publicBatchWrapper89_copies2523 a h).1
  have c80737 := (publicBatchWrapper89_copies2523 a h).2.1
  have c80738 := (publicBatchWrapper89_copies2523 a h).2.2.1
  have c80739 := (publicBatchWrapper89_copies2523 a h).2.2.2.1
  have c80740 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.1
  have c80741 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_8 := arithEq_of_rows h (row := 1312) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_8
  simp only [← c80736, ← c80737, ← c80738] at e_1312_8
  have e_1312_9 := arithEq_of_rows h (row := 1312) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_9
  simp only [← c80739, ← c80740, k2, ← c80741] at e_1312_9
  have hr := e_1312_9
  simp only [e_1312_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12090)) := by
  have c80742 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.1
  have c80743 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.1
  have c80744 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.1
  have c80745 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.1
  have c80746 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80747 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_10 := arithEq_of_rows h (row := 1312) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_10
  simp only [← c80742, ← c80743, ← c80744] at e_1312_10
  have e_1312_11 := arithEq_of_rows h (row := 1312) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_11
  simp only [← c80745, ← c80746, k2, ← c80747] at e_1312_11
  have hr := e_1312_11
  simp only [e_1312_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12091)) := by
  have c80748 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80749 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80750 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80751 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80752 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80753 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_12 := arithEq_of_rows h (row := 1312) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_12
  simp only [← c80748, ← c80749, ← c80750] at e_1312_12
  have e_1312_13 := arithEq_of_rows h (row := 1312) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_13
  simp only [← c80751, ← c80752, k2, ← c80753] at e_1312_13
  have hr := e_1312_13
  simp only [e_1312_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12092)) := by
  have c80754 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80755 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80756 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80757 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80758 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80759 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_14 := arithEq_of_rows h (row := 1312) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_14
  simp only [← c80754, ← c80755, ← c80756] at e_1312_14
  have e_1312_15 := arithEq_of_rows h (row := 1312) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_15
  simp only [← c80757, ← c80758, k2, ← c80759] at e_1312_15
  have hr := e_1312_15
  simp only [e_1312_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12093)) := by
  have c80760 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80761 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80762 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80763 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80764 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80765 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_16 := arithEq_of_rows h (row := 1312) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_16
  simp only [← c80760, ← c80761, ← c80762] at e_1312_16
  have e_1312_17 := arithEq_of_rows h (row := 1312) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_17
  simp only [← c80763, ← c80764, k2, ← c80765] at e_1312_17
  have hr := e_1312_17
  simp only [e_1312_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1312 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12094)) := by
  have c80766 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80767 := (publicBatchWrapper89_copies2523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80768 := (publicBatchWrapper89_copies2524 a h).1
  have c80769 := (publicBatchWrapper89_copies2524 a h).2.1
  have c80770 := (publicBatchWrapper89_copies2524 a h).2.2.1
  have c80771 := (publicBatchWrapper89_copies2524 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1312_18 := arithEq_of_rows h (row := 1312) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_18
  simp only [← c80766, ← c80767, ← c80768] at e_1312_18
  have e_1312_19 := arithEq_of_rows h (row := 1312) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1312_19
  simp only [← c80769, ← c80770, k2, ← c80771] at e_1312_19
  have hr := e_1312_19
  simp only [e_1312_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12095)) := by
  have c80772 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.1
  have c80773 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.1
  have c80774 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.1
  have c80775 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.1
  have c80776 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.1
  have c80777 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_0 := arithEq_of_rows h (row := 1313) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_0
  simp only [← c80772, ← c80773, ← c80774] at e_1313_0
  have e_1313_1 := arithEq_of_rows h (row := 1313) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_1
  simp only [← c80775, ← c80776, k2, ← c80777] at e_1313_1
  have hr := e_1313_1
  simp only [e_1313_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12096)) := by
  have c80778 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80779 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80780 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80781 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80782 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80783 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_2 := arithEq_of_rows h (row := 1313) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_2
  simp only [← c80778, ← c80779, ← c80780] at e_1313_2
  have e_1313_3 := arithEq_of_rows h (row := 1313) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_3
  simp only [← c80781, ← c80782, k2, ← c80783] at e_1313_3
  have hr := e_1313_3
  simp only [e_1313_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12097)) := by
  have c80784 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80785 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80786 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80787 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80788 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80789 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_4 := arithEq_of_rows h (row := 1313) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_4
  simp only [← c80784, ← c80785, ← c80786] at e_1313_4
  have e_1313_5 := arithEq_of_rows h (row := 1313) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_5
  simp only [← c80787, ← c80788, k2, ← c80789] at e_1313_5
  have hr := e_1313_5
  simp only [e_1313_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12098)) := by
  have c80790 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80791 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80792 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80793 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80794 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80795 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_6 := arithEq_of_rows h (row := 1313) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_6
  simp only [← c80790, ← c80791, ← c80792] at e_1313_6
  have e_1313_7 := arithEq_of_rows h (row := 1313) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_7
  simp only [← c80793, ← c80794, k2, ← c80795] at e_1313_7
  have hr := e_1313_7
  simp only [e_1313_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12099)) := by
  have c80796 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80797 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80798 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80799 := (publicBatchWrapper89_copies2524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80800 := (publicBatchWrapper89_copies2525 a h).1
  have c80801 := (publicBatchWrapper89_copies2525 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_8 := arithEq_of_rows h (row := 1313) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_8
  simp only [← c80796, ← c80797, ← c80798] at e_1313_8
  have e_1313_9 := arithEq_of_rows h (row := 1313) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_9
  simp only [← c80799, ← c80800, k2, ← c80801] at e_1313_9
  have hr := e_1313_9
  simp only [e_1313_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12100)) := by
  have c80802 := (publicBatchWrapper89_copies2525 a h).2.2.1
  have c80803 := (publicBatchWrapper89_copies2525 a h).2.2.2.1
  have c80804 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.1
  have c80805 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.1
  have c80806 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.1
  have c80807 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_10 := arithEq_of_rows h (row := 1313) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_10
  simp only [← c80802, ← c80803, ← c80804] at e_1313_10
  have e_1313_11 := arithEq_of_rows h (row := 1313) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_11
  simp only [← c80805, ← c80806, k2, ← c80807] at e_1313_11
  have hr := e_1313_11
  simp only [e_1313_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12101)) := by
  have c80808 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.1
  have c80809 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.1
  have c80810 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80811 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80812 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80813 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_12 := arithEq_of_rows h (row := 1313) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_12
  simp only [← c80808, ← c80809, ← c80810] at e_1313_12
  have e_1313_13 := arithEq_of_rows h (row := 1313) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_13
  simp only [← c80811, ← c80812, k2, ← c80813] at e_1313_13
  have hr := e_1313_13
  simp only [e_1313_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12102)) := by
  have c80814 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80815 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80816 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80817 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80818 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80819 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_14 := arithEq_of_rows h (row := 1313) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_14
  simp only [← c80814, ← c80815, ← c80816] at e_1313_14
  have e_1313_15 := arithEq_of_rows h (row := 1313) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_15
  simp only [← c80817, ← c80818, k2, ← c80819] at e_1313_15
  have hr := e_1313_15
  simp only [e_1313_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12103)) := by
  have c80820 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80821 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80822 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80823 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80824 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80825 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_16 := arithEq_of_rows h (row := 1313) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_16
  simp only [← c80820, ← c80821, ← c80822] at e_1313_16
  have e_1313_17 := arithEq_of_rows h (row := 1313) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_17
  simp only [← c80823, ← c80824, k2, ← c80825] at e_1313_17
  have hr := e_1313_17
  simp only [e_1313_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1313 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12104)) := by
  have c80826 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80827 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80828 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80829 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80830 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80831 := (publicBatchWrapper89_copies2525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1313_18 := arithEq_of_rows h (row := 1313) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_18
  simp only [← c80826, ← c80827, ← c80828] at e_1313_18
  have e_1313_19 := arithEq_of_rows h (row := 1313) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1313_19
  simp only [← c80829, ← c80830, k2, ← c80831] at e_1313_19
  have hr := e_1313_19
  simp only [e_1313_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12105)) := by
  have c80832 := (publicBatchWrapper89_copies2526 a h).1
  have c80833 := (publicBatchWrapper89_copies2526 a h).2.1
  have c80834 := (publicBatchWrapper89_copies2526 a h).2.2.1
  have c80835 := (publicBatchWrapper89_copies2526 a h).2.2.2.1
  have c80836 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.1
  have c80837 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_0 := arithEq_of_rows h (row := 1314) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_0
  simp only [← c80832, ← c80833, ← c80834] at e_1314_0
  have e_1314_1 := arithEq_of_rows h (row := 1314) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_1
  simp only [← c80835, ← c80836, k2, ← c80837] at e_1314_1
  have hr := e_1314_1
  simp only [e_1314_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12106)) := by
  have c80838 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.1
  have c80839 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.1
  have c80840 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.1
  have c80841 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.1
  have c80842 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80843 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_2 := arithEq_of_rows h (row := 1314) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_2
  simp only [← c80838, ← c80839, ← c80840] at e_1314_2
  have e_1314_3 := arithEq_of_rows h (row := 1314) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_3
  simp only [← c80841, ← c80842, k2, ← c80843] at e_1314_3
  have hr := e_1314_3
  simp only [e_1314_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12107)) := by
  have c80844 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80845 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80846 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80847 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80848 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80849 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_4 := arithEq_of_rows h (row := 1314) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_4
  simp only [← c80844, ← c80845, ← c80846] at e_1314_4
  have e_1314_5 := arithEq_of_rows h (row := 1314) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_5
  simp only [← c80847, ← c80848, k2, ← c80849] at e_1314_5
  have hr := e_1314_5
  simp only [e_1314_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12108)) := by
  have c80850 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80851 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80852 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80853 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80854 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80855 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_6 := arithEq_of_rows h (row := 1314) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_6
  simp only [← c80850, ← c80851, ← c80852] at e_1314_6
  have e_1314_7 := arithEq_of_rows h (row := 1314) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_7
  simp only [← c80853, ← c80854, k2, ← c80855] at e_1314_7
  have hr := e_1314_7
  simp only [e_1314_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12109)) := by
  have c80856 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80857 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80858 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80859 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80860 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80861 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_8 := arithEq_of_rows h (row := 1314) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_8
  simp only [← c80856, ← c80857, ← c80858] at e_1314_8
  have e_1314_9 := arithEq_of_rows h (row := 1314) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_9
  simp only [← c80859, ← c80860, k2, ← c80861] at e_1314_9
  have hr := e_1314_9
  simp only [e_1314_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12110)) := by
  have c80862 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80863 := (publicBatchWrapper89_copies2526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80864 := (publicBatchWrapper89_copies2527 a h).1
  have c80865 := (publicBatchWrapper89_copies2527 a h).2.1
  have c80866 := (publicBatchWrapper89_copies2527 a h).2.2.1
  have c80867 := (publicBatchWrapper89_copies2527 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_10 := arithEq_of_rows h (row := 1314) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_10
  simp only [← c80862, ← c80863, ← c80864] at e_1314_10
  have e_1314_11 := arithEq_of_rows h (row := 1314) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_11
  simp only [← c80865, ← c80866, k2, ← c80867] at e_1314_11
  have hr := e_1314_11
  simp only [e_1314_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12111)) := by
  have c80868 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.1
  have c80869 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.1
  have c80870 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.1
  have c80871 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.1
  have c80872 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.1
  have c80873 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_12 := arithEq_of_rows h (row := 1314) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_12
  simp only [← c80868, ← c80869, ← c80870] at e_1314_12
  have e_1314_13 := arithEq_of_rows h (row := 1314) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_13
  simp only [← c80871, ← c80872, k2, ← c80873] at e_1314_13
  have hr := e_1314_13
  simp only [e_1314_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12112)) := by
  have c80874 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80875 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80876 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80877 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80878 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80879 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_14 := arithEq_of_rows h (row := 1314) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_14
  simp only [← c80874, ← c80875, ← c80876] at e_1314_14
  have e_1314_15 := arithEq_of_rows h (row := 1314) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_15
  simp only [← c80877, ← c80878, k2, ← c80879] at e_1314_15
  have hr := e_1314_15
  simp only [e_1314_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12113)) := by
  have c80880 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80881 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80882 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80883 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80884 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80885 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_16 := arithEq_of_rows h (row := 1314) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_16
  simp only [← c80880, ← c80881, ← c80882] at e_1314_16
  have e_1314_17 := arithEq_of_rows h (row := 1314) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_17
  simp only [← c80883, ← c80884, k2, ← c80885] at e_1314_17
  have hr := e_1314_17
  simp only [e_1314_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1314 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12114)) := by
  have c80886 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80887 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80888 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80889 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80890 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80891 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1314_18 := arithEq_of_rows h (row := 1314) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_18
  simp only [← c80886, ← c80887, ← c80888] at e_1314_18
  have e_1314_19 := arithEq_of_rows h (row := 1314) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1314_19
  simp only [← c80889, ← c80890, k2, ← c80891] at e_1314_19
  have hr := e_1314_19
  simp only [e_1314_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12115)) := by
  have c80892 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80893 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80894 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80895 := (publicBatchWrapper89_copies2527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80896 := (publicBatchWrapper89_copies2528 a h).1
  have c80897 := (publicBatchWrapper89_copies2528 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_0 := arithEq_of_rows h (row := 1315) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_0
  simp only [← c80892, ← c80893, ← c80894] at e_1315_0
  have e_1315_1 := arithEq_of_rows h (row := 1315) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_1
  simp only [← c80895, ← c80896, k2, ← c80897] at e_1315_1
  have hr := e_1315_1
  simp only [e_1315_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12116)) := by
  have c80898 := (publicBatchWrapper89_copies2528 a h).2.2.1
  have c80899 := (publicBatchWrapper89_copies2528 a h).2.2.2.1
  have c80900 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.1
  have c80901 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.1
  have c80902 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.1
  have c80903 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_2 := arithEq_of_rows h (row := 1315) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_2
  simp only [← c80898, ← c80899, ← c80900] at e_1315_2
  have e_1315_3 := arithEq_of_rows h (row := 1315) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_3
  simp only [← c80901, ← c80902, k2, ← c80903] at e_1315_3
  have hr := e_1315_3
  simp only [e_1315_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12117)) := by
  have c80904 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.1
  have c80905 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.1
  have c80906 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80907 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80908 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80909 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_4 := arithEq_of_rows h (row := 1315) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_4
  simp only [← c80904, ← c80905, ← c80906] at e_1315_4
  have e_1315_5 := arithEq_of_rows h (row := 1315) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_5
  simp only [← c80907, ← c80908, k2, ← c80909] at e_1315_5
  have hr := e_1315_5
  simp only [e_1315_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12118)) := by
  have c80910 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80911 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80912 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80913 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80914 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80915 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_6 := arithEq_of_rows h (row := 1315) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_6
  simp only [← c80910, ← c80911, ← c80912] at e_1315_6
  have e_1315_7 := arithEq_of_rows h (row := 1315) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_7
  simp only [← c80913, ← c80914, k2, ← c80915] at e_1315_7
  have hr := e_1315_7
  simp only [e_1315_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12119)) := by
  have c80916 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80917 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80918 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80919 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80920 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80921 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_8 := arithEq_of_rows h (row := 1315) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_8
  simp only [← c80916, ← c80917, ← c80918] at e_1315_8
  have e_1315_9 := arithEq_of_rows h (row := 1315) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_9
  simp only [← c80919, ← c80920, k2, ← c80921] at e_1315_9
  have hr := e_1315_9
  simp only [e_1315_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12120)) := by
  have c80922 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80923 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80924 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80925 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80926 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80927 := (publicBatchWrapper89_copies2528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_10 := arithEq_of_rows h (row := 1315) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_10
  simp only [← c80922, ← c80923, ← c80924] at e_1315_10
  have e_1315_11 := arithEq_of_rows h (row := 1315) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_11
  simp only [← c80925, ← c80926, k2, ← c80927] at e_1315_11
  have hr := e_1315_11
  simp only [e_1315_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12121)) := by
  have c80928 := (publicBatchWrapper89_copies2529 a h).1
  have c80929 := (publicBatchWrapper89_copies2529 a h).2.1
  have c80930 := (publicBatchWrapper89_copies2529 a h).2.2.1
  have c80931 := (publicBatchWrapper89_copies2529 a h).2.2.2.1
  have c80932 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.1
  have c80933 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_12 := arithEq_of_rows h (row := 1315) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_12
  simp only [← c80928, ← c80929, ← c80930] at e_1315_12
  have e_1315_13 := arithEq_of_rows h (row := 1315) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_13
  simp only [← c80931, ← c80932, k2, ← c80933] at e_1315_13
  have hr := e_1315_13
  simp only [e_1315_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12122)) := by
  have c80934 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.1
  have c80935 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.1
  have c80936 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.1
  have c80937 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.1
  have c80938 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80939 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_14 := arithEq_of_rows h (row := 1315) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_14
  simp only [← c80934, ← c80935, ← c80936] at e_1315_14
  have e_1315_15 := arithEq_of_rows h (row := 1315) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_15
  simp only [← c80937, ← c80938, k2, ← c80939] at e_1315_15
  have hr := e_1315_15
  simp only [e_1315_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12123)) := by
  have c80940 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80941 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80942 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80943 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80944 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80945 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_16 := arithEq_of_rows h (row := 1315) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_16
  simp only [← c80940, ← c80941, ← c80942] at e_1315_16
  have e_1315_17 := arithEq_of_rows h (row := 1315) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_17
  simp only [← c80943, ← c80944, k2, ← c80945] at e_1315_17
  have hr := e_1315_17
  simp only [e_1315_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1315 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12124)) := by
  have c80946 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80947 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80948 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80949 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80950 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80951 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1315_18 := arithEq_of_rows h (row := 1315) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_18
  simp only [← c80946, ← c80947, ← c80948] at e_1315_18
  have e_1315_19 := arithEq_of_rows h (row := 1315) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1315_19
  simp only [← c80949, ← c80950, k2, ← c80951] at e_1315_19
  have hr := e_1315_19
  simp only [e_1315_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12125)) := by
  have c80952 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80953 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80954 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80955 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80956 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80957 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_0 := arithEq_of_rows h (row := 1316) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_0
  simp only [← c80952, ← c80953, ← c80954] at e_1316_0
  have e_1316_1 := arithEq_of_rows h (row := 1316) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_1
  simp only [← c80955, ← c80956, k2, ← c80957] at e_1316_1
  have hr := e_1316_1
  simp only [e_1316_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12126)) := by
  have c80958 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80959 := (publicBatchWrapper89_copies2529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80960 := (publicBatchWrapper89_copies2530 a h).1
  have c80961 := (publicBatchWrapper89_copies2530 a h).2.1
  have c80962 := (publicBatchWrapper89_copies2530 a h).2.2.1
  have c80963 := (publicBatchWrapper89_copies2530 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_2 := arithEq_of_rows h (row := 1316) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_2
  simp only [← c80958, ← c80959, ← c80960] at e_1316_2
  have e_1316_3 := arithEq_of_rows h (row := 1316) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_3
  simp only [← c80961, ← c80962, k2, ← c80963] at e_1316_3
  have hr := e_1316_3
  simp only [e_1316_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12127)) := by
  have c80964 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.1
  have c80965 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.1
  have c80966 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.1
  have c80967 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.1
  have c80968 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.1
  have c80969 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_4 := arithEq_of_rows h (row := 1316) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_4
  simp only [← c80964, ← c80965, ← c80966] at e_1316_4
  have e_1316_5 := arithEq_of_rows h (row := 1316) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_5
  simp only [← c80967, ← c80968, k2, ← c80969] at e_1316_5
  have hr := e_1316_5
  simp only [e_1316_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12128)) := by
  have c80970 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c80971 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c80972 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80973 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80974 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80975 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_6 := arithEq_of_rows h (row := 1316) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_6
  simp only [← c80970, ← c80971, ← c80972] at e_1316_6
  have e_1316_7 := arithEq_of_rows h (row := 1316) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_7
  simp only [← c80973, ← c80974, k2, ← c80975] at e_1316_7
  have hr := e_1316_7
  simp only [e_1316_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12129)) := by
  have c80976 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80977 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80978 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80979 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80980 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80981 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_8 := arithEq_of_rows h (row := 1316) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_8
  simp only [← c80976, ← c80977, ← c80978] at e_1316_8
  have e_1316_9 := arithEq_of_rows h (row := 1316) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_9
  simp only [← c80979, ← c80980, k2, ← c80981] at e_1316_9
  have hr := e_1316_9
  simp only [e_1316_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12130)) := by
  have c80982 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80983 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80984 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80985 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80986 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80987 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_10 := arithEq_of_rows h (row := 1316) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_10
  simp only [← c80982, ← c80983, ← c80984] at e_1316_10
  have e_1316_11 := arithEq_of_rows h (row := 1316) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_11
  simp only [← c80985, ← c80986, k2, ← c80987] at e_1316_11
  have hr := e_1316_11
  simp only [e_1316_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12131)) := by
  have c80988 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80989 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80990 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c80991 := (publicBatchWrapper89_copies2530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c80992 := (publicBatchWrapper89_copies2531 a h).1
  have c80993 := (publicBatchWrapper89_copies2531 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_12 := arithEq_of_rows h (row := 1316) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_12
  simp only [← c80988, ← c80989, ← c80990] at e_1316_12
  have e_1316_13 := arithEq_of_rows h (row := 1316) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_13
  simp only [← c80991, ← c80992, k2, ← c80993] at e_1316_13
  have hr := e_1316_13
  simp only [e_1316_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12132)) := by
  have c80994 := (publicBatchWrapper89_copies2531 a h).2.2.1
  have c80995 := (publicBatchWrapper89_copies2531 a h).2.2.2.1
  have c80996 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.1
  have c80997 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.1
  have c80998 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.1
  have c80999 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_14 := arithEq_of_rows h (row := 1316) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_14
  simp only [← c80994, ← c80995, ← c80996] at e_1316_14
  have e_1316_15 := arithEq_of_rows h (row := 1316) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_15
  simp only [← c80997, ← c80998, k2, ← c80999] at e_1316_15
  have hr := e_1316_15
  simp only [e_1316_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12133)) := by
  have c81000 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.1
  have c81001 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.1
  have c81002 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81003 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81004 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81005 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_16 := arithEq_of_rows h (row := 1316) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_16
  simp only [← c81000, ← c81001, ← c81002] at e_1316_16
  have e_1316_17 := arithEq_of_rows h (row := 1316) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_17
  simp only [← c81003, ← c81004, k2, ← c81005] at e_1316_17
  have hr := e_1316_17
  simp only [e_1316_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1316 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12134)) := by
  have c81006 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81007 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81008 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81009 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81010 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81011 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1316_18 := arithEq_of_rows h (row := 1316) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_18
  simp only [← c81006, ← c81007, ← c81008] at e_1316_18
  have e_1316_19 := arithEq_of_rows h (row := 1316) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1316_19
  simp only [← c81009, ← c81010, k2, ← c81011] at e_1316_19
  have hr := e_1316_19
  simp only [e_1316_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12135)) := by
  have c81012 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81013 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81014 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81015 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81016 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81017 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_0 := arithEq_of_rows h (row := 1317) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_0
  simp only [← c81012, ← c81013, ← c81014] at e_1317_0
  have e_1317_1 := arithEq_of_rows h (row := 1317) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_1
  simp only [← c81015, ← c81016, k2, ← c81017] at e_1317_1
  have hr := e_1317_1
  simp only [e_1317_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12136)) := by
  have c81018 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81019 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81020 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81021 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81022 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81023 := (publicBatchWrapper89_copies2531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_2 := arithEq_of_rows h (row := 1317) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_2
  simp only [← c81018, ← c81019, ← c81020] at e_1317_2
  have e_1317_3 := arithEq_of_rows h (row := 1317) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_3
  simp only [← c81021, ← c81022, k2, ← c81023] at e_1317_3
  have hr := e_1317_3
  simp only [e_1317_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12137)) := by
  have c81024 := (publicBatchWrapper89_copies2532 a h).1
  have c81025 := (publicBatchWrapper89_copies2532 a h).2.1
  have c81026 := (publicBatchWrapper89_copies2532 a h).2.2.1
  have c81027 := (publicBatchWrapper89_copies2532 a h).2.2.2.1
  have c81028 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.1
  have c81029 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_4 := arithEq_of_rows h (row := 1317) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_4
  simp only [← c81024, ← c81025, ← c81026] at e_1317_4
  have e_1317_5 := arithEq_of_rows h (row := 1317) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_5
  simp only [← c81027, ← c81028, k2, ← c81029] at e_1317_5
  have hr := e_1317_5
  simp only [e_1317_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12138)) := by
  have c81030 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.1
  have c81031 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.1
  have c81032 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.1
  have c81033 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.1
  have c81034 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81035 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_6 := arithEq_of_rows h (row := 1317) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_6
  simp only [← c81030, ← c81031, ← c81032] at e_1317_6
  have e_1317_7 := arithEq_of_rows h (row := 1317) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_7
  simp only [← c81033, ← c81034, k2, ← c81035] at e_1317_7
  have hr := e_1317_7
  simp only [e_1317_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12139)) := by
  have c81036 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81037 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81038 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81039 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81040 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81041 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_8 := arithEq_of_rows h (row := 1317) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_8
  simp only [← c81036, ← c81037, ← c81038] at e_1317_8
  have e_1317_9 := arithEq_of_rows h (row := 1317) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_9
  simp only [← c81039, ← c81040, k2, ← c81041] at e_1317_9
  have hr := e_1317_9
  simp only [e_1317_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12140)) := by
  have c81042 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81043 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81044 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81045 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81046 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81047 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_10 := arithEq_of_rows h (row := 1317) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_10
  simp only [← c81042, ← c81043, ← c81044] at e_1317_10
  have e_1317_11 := arithEq_of_rows h (row := 1317) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_11
  simp only [← c81045, ← c81046, k2, ← c81047] at e_1317_11
  have hr := e_1317_11
  simp only [e_1317_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12141)) := by
  have c81048 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81049 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81050 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81051 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81052 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81053 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_12 := arithEq_of_rows h (row := 1317) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_12
  simp only [← c81048, ← c81049, ← c81050] at e_1317_12
  have e_1317_13 := arithEq_of_rows h (row := 1317) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_13
  simp only [← c81051, ← c81052, k2, ← c81053] at e_1317_13
  have hr := e_1317_13
  simp only [e_1317_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12142)) := by
  have c81054 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81055 := (publicBatchWrapper89_copies2532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81056 := (publicBatchWrapper89_copies2533 a h).1
  have c81057 := (publicBatchWrapper89_copies2533 a h).2.1
  have c81058 := (publicBatchWrapper89_copies2533 a h).2.2.1
  have c81059 := (publicBatchWrapper89_copies2533 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_14 := arithEq_of_rows h (row := 1317) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_14
  simp only [← c81054, ← c81055, ← c81056] at e_1317_14
  have e_1317_15 := arithEq_of_rows h (row := 1317) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_15
  simp only [← c81057, ← c81058, k2, ← c81059] at e_1317_15
  have hr := e_1317_15
  simp only [e_1317_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12143)) := by
  have c81060 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.1
  have c81061 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.1
  have c81062 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.1
  have c81063 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.1
  have c81064 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.1
  have c81065 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_16 := arithEq_of_rows h (row := 1317) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_16
  simp only [← c81060, ← c81061, ← c81062] at e_1317_16
  have e_1317_17 := arithEq_of_rows h (row := 1317) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_17
  simp only [← c81063, ← c81064, k2, ← c81065] at e_1317_17
  have hr := e_1317_17
  simp only [e_1317_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1317 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12144)) := by
  have c81066 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81067 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81068 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81069 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81070 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81071 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1317_18 := arithEq_of_rows h (row := 1317) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_18
  simp only [← c81066, ← c81067, ← c81068] at e_1317_18
  have e_1317_19 := arithEq_of_rows h (row := 1317) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1317_19
  simp only [← c81069, ← c81070, k2, ← c81071] at e_1317_19
  have hr := e_1317_19
  simp only [e_1317_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12145)) := by
  have c81072 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81073 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81074 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81075 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81076 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81077 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_0 := arithEq_of_rows h (row := 1318) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_0
  simp only [← c81072, ← c81073, ← c81074] at e_1318_0
  have e_1318_1 := arithEq_of_rows h (row := 1318) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_1
  simp only [← c81075, ← c81076, k2, ← c81077] at e_1318_1
  have hr := e_1318_1
  simp only [e_1318_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12146)) := by
  have c81078 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81079 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81080 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81081 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81082 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81083 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_2 := arithEq_of_rows h (row := 1318) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_2
  simp only [← c81078, ← c81079, ← c81080] at e_1318_2
  have e_1318_3 := arithEq_of_rows h (row := 1318) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_3
  simp only [← c81081, ← c81082, k2, ← c81083] at e_1318_3
  have hr := e_1318_3
  simp only [e_1318_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12147)) := by
  have c81084 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81085 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81086 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81087 := (publicBatchWrapper89_copies2533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81088 := (publicBatchWrapper89_copies2534 a h).1
  have c81089 := (publicBatchWrapper89_copies2534 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_4 := arithEq_of_rows h (row := 1318) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_4
  simp only [← c81084, ← c81085, ← c81086] at e_1318_4
  have e_1318_5 := arithEq_of_rows h (row := 1318) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_5
  simp only [← c81087, ← c81088, k2, ← c81089] at e_1318_5
  have hr := e_1318_5
  simp only [e_1318_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12148)) := by
  have c81090 := (publicBatchWrapper89_copies2534 a h).2.2.1
  have c81091 := (publicBatchWrapper89_copies2534 a h).2.2.2.1
  have c81092 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.1
  have c81093 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.1
  have c81094 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.1
  have c81095 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_6 := arithEq_of_rows h (row := 1318) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_6
  simp only [← c81090, ← c81091, ← c81092] at e_1318_6
  have e_1318_7 := arithEq_of_rows h (row := 1318) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_7
  simp only [← c81093, ← c81094, k2, ← c81095] at e_1318_7
  have hr := e_1318_7
  simp only [e_1318_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12149)) := by
  have c81096 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.1
  have c81097 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.1
  have c81098 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81099 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81100 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81101 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_8 := arithEq_of_rows h (row := 1318) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_8
  simp only [← c81096, ← c81097, ← c81098] at e_1318_8
  have e_1318_9 := arithEq_of_rows h (row := 1318) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_9
  simp only [← c81099, ← c81100, k2, ← c81101] at e_1318_9
  have hr := e_1318_9
  simp only [e_1318_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12150)) := by
  have c81102 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81103 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81104 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81105 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81106 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81107 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_10 := arithEq_of_rows h (row := 1318) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_10
  simp only [← c81102, ← c81103, ← c81104] at e_1318_10
  have e_1318_11 := arithEq_of_rows h (row := 1318) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_11
  simp only [← c81105, ← c81106, k2, ← c81107] at e_1318_11
  have hr := e_1318_11
  simp only [e_1318_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12151)) := by
  have c81108 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81109 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81110 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81111 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81112 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81113 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_12 := arithEq_of_rows h (row := 1318) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_12
  simp only [← c81108, ← c81109, ← c81110] at e_1318_12
  have e_1318_13 := arithEq_of_rows h (row := 1318) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_13
  simp only [← c81111, ← c81112, k2, ← c81113] at e_1318_13
  have hr := e_1318_13
  simp only [e_1318_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12152)) := by
  have c81114 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81115 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81116 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81117 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81118 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81119 := (publicBatchWrapper89_copies2534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_14 := arithEq_of_rows h (row := 1318) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_14
  simp only [← c81114, ← c81115, ← c81116] at e_1318_14
  have e_1318_15 := arithEq_of_rows h (row := 1318) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_15
  simp only [← c81117, ← c81118, k2, ← c81119] at e_1318_15
  have hr := e_1318_15
  simp only [e_1318_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12153)) := by
  have c81120 := (publicBatchWrapper89_copies2535 a h).1
  have c81121 := (publicBatchWrapper89_copies2535 a h).2.1
  have c81122 := (publicBatchWrapper89_copies2535 a h).2.2.1
  have c81123 := (publicBatchWrapper89_copies2535 a h).2.2.2.1
  have c81124 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.1
  have c81125 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_16 := arithEq_of_rows h (row := 1318) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_16
  simp only [← c81120, ← c81121, ← c81122] at e_1318_16
  have e_1318_17 := arithEq_of_rows h (row := 1318) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_17
  simp only [← c81123, ← c81124, k2, ← c81125] at e_1318_17
  have hr := e_1318_17
  simp only [e_1318_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1318 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12154)) := by
  have c81126 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.1
  have c81127 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.1
  have c81128 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.1
  have c81129 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.1
  have c81130 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81131 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1318_18 := arithEq_of_rows h (row := 1318) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_18
  simp only [← c81126, ← c81127, ← c81128] at e_1318_18
  have e_1318_19 := arithEq_of_rows h (row := 1318) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1318_19
  simp only [← c81129, ← c81130, k2, ← c81131] at e_1318_19
  have hr := e_1318_19
  simp only [e_1318_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12155)) := by
  have c81132 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81133 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81134 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81135 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81136 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81137 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_0 := arithEq_of_rows h (row := 1319) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_0
  simp only [← c81132, ← c81133, ← c81134] at e_1319_0
  have e_1319_1 := arithEq_of_rows h (row := 1319) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_1
  simp only [← c81135, ← c81136, k2, ← c81137] at e_1319_1
  have hr := e_1319_1
  simp only [e_1319_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12156)) := by
  have c81138 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81139 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81140 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81141 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81142 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81143 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_2 := arithEq_of_rows h (row := 1319) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_2
  simp only [← c81138, ← c81139, ← c81140] at e_1319_2
  have e_1319_3 := arithEq_of_rows h (row := 1319) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_3
  simp only [← c81141, ← c81142, k2, ← c81143] at e_1319_3
  have hr := e_1319_3
  simp only [e_1319_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12157)) := by
  have c81144 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81145 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81146 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81147 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81148 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81149 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_4 := arithEq_of_rows h (row := 1319) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_4
  simp only [← c81144, ← c81145, ← c81146] at e_1319_4
  have e_1319_5 := arithEq_of_rows h (row := 1319) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_5
  simp only [← c81147, ← c81148, k2, ← c81149] at e_1319_5
  have hr := e_1319_5
  simp only [e_1319_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12158)) := by
  have c81150 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81151 := (publicBatchWrapper89_copies2535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81152 := (publicBatchWrapper89_copies2536 a h).1
  have c81153 := (publicBatchWrapper89_copies2536 a h).2.1
  have c81154 := (publicBatchWrapper89_copies2536 a h).2.2.1
  have c81155 := (publicBatchWrapper89_copies2536 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_6 := arithEq_of_rows h (row := 1319) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_6
  simp only [← c81150, ← c81151, ← c81152] at e_1319_6
  have e_1319_7 := arithEq_of_rows h (row := 1319) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_7
  simp only [← c81153, ← c81154, k2, ← c81155] at e_1319_7
  have hr := e_1319_7
  simp only [e_1319_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12159)) := by
  have c81156 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.1
  have c81157 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.1
  have c81158 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.1
  have c81159 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.1
  have c81160 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.1
  have c81161 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_8 := arithEq_of_rows h (row := 1319) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_8
  simp only [← c81156, ← c81157, ← c81158] at e_1319_8
  have e_1319_9 := arithEq_of_rows h (row := 1319) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_9
  simp only [← c81159, ← c81160, k2, ← c81161] at e_1319_9
  have hr := e_1319_9
  simp only [e_1319_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12160)) := by
  have c81162 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81163 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81164 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81165 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81166 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81167 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_10 := arithEq_of_rows h (row := 1319) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_10
  simp only [← c81162, ← c81163, ← c81164] at e_1319_10
  have e_1319_11 := arithEq_of_rows h (row := 1319) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_11
  simp only [← c81165, ← c81166, k2, ← c81167] at e_1319_11
  have hr := e_1319_11
  simp only [e_1319_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12161)) := by
  have c81168 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81169 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81170 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81171 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81172 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81173 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_12 := arithEq_of_rows h (row := 1319) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_12
  simp only [← c81168, ← c81169, ← c81170] at e_1319_12
  have e_1319_13 := arithEq_of_rows h (row := 1319) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_13
  simp only [← c81171, ← c81172, k2, ← c81173] at e_1319_13
  have hr := e_1319_13
  simp only [e_1319_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12162)) := by
  have c81174 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81175 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81176 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81177 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81178 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81179 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_14 := arithEq_of_rows h (row := 1319) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_14
  simp only [← c81174, ← c81175, ← c81176] at e_1319_14
  have e_1319_15 := arithEq_of_rows h (row := 1319) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_15
  simp only [← c81177, ← c81178, k2, ← c81179] at e_1319_15
  have hr := e_1319_15
  simp only [e_1319_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12163)) := by
  have c81180 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81181 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81182 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81183 := (publicBatchWrapper89_copies2536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81184 := (publicBatchWrapper89_copies2537 a h).1
  have c81185 := (publicBatchWrapper89_copies2537 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_16 := arithEq_of_rows h (row := 1319) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_16
  simp only [← c81180, ← c81181, ← c81182] at e_1319_16
  have e_1319_17 := arithEq_of_rows h (row := 1319) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_17
  simp only [← c81183, ← c81184, k2, ← c81185] at e_1319_17
  have hr := e_1319_17
  simp only [e_1319_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1319 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12164)) := by
  have c81186 := (publicBatchWrapper89_copies2537 a h).2.2.1
  have c81187 := (publicBatchWrapper89_copies2537 a h).2.2.2.1
  have c81188 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.1
  have c81189 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.1
  have c81190 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.1
  have c81191 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1319_18 := arithEq_of_rows h (row := 1319) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_18
  simp only [← c81186, ← c81187, ← c81188] at e_1319_18
  have e_1319_19 := arithEq_of_rows h (row := 1319) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1319_19
  simp only [← c81189, ← c81190, k2, ← c81191] at e_1319_19
  have hr := e_1319_19
  simp only [e_1319_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12165)) := by
  have c81192 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.1
  have c81193 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.1
  have c81194 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81195 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81196 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81197 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_0 := arithEq_of_rows h (row := 1320) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_0
  simp only [← c81192, ← c81193, ← c81194] at e_1320_0
  have e_1320_1 := arithEq_of_rows h (row := 1320) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_1
  simp only [← c81195, ← c81196, k2, ← c81197] at e_1320_1
  have hr := e_1320_1
  simp only [e_1320_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12166)) := by
  have c81198 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81199 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81200 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81201 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81202 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81203 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_2 := arithEq_of_rows h (row := 1320) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_2
  simp only [← c81198, ← c81199, ← c81200] at e_1320_2
  have e_1320_3 := arithEq_of_rows h (row := 1320) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_3
  simp only [← c81201, ← c81202, k2, ← c81203] at e_1320_3
  have hr := e_1320_3
  simp only [e_1320_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12167)) := by
  have c81204 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81205 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81206 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81207 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81208 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81209 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_4 := arithEq_of_rows h (row := 1320) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_4
  simp only [← c81204, ← c81205, ← c81206] at e_1320_4
  have e_1320_5 := arithEq_of_rows h (row := 1320) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_5
  simp only [← c81207, ← c81208, k2, ← c81209] at e_1320_5
  have hr := e_1320_5
  simp only [e_1320_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12168)) := by
  have c81210 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81211 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81212 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81213 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81214 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81215 := (publicBatchWrapper89_copies2537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_6 := arithEq_of_rows h (row := 1320) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_6
  simp only [← c81210, ← c81211, ← c81212] at e_1320_6
  have e_1320_7 := arithEq_of_rows h (row := 1320) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_7
  simp only [← c81213, ← c81214, k2, ← c81215] at e_1320_7
  have hr := e_1320_7
  simp only [e_1320_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12169)) := by
  have c81216 := (publicBatchWrapper89_copies2538 a h).1
  have c81217 := (publicBatchWrapper89_copies2538 a h).2.1
  have c81218 := (publicBatchWrapper89_copies2538 a h).2.2.1
  have c81219 := (publicBatchWrapper89_copies2538 a h).2.2.2.1
  have c81220 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.1
  have c81221 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_8 := arithEq_of_rows h (row := 1320) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_8
  simp only [← c81216, ← c81217, ← c81218] at e_1320_8
  have e_1320_9 := arithEq_of_rows h (row := 1320) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_9
  simp only [← c81219, ← c81220, k2, ← c81221] at e_1320_9
  have hr := e_1320_9
  simp only [e_1320_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12210)) := by
  have c81222 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.1
  have c81223 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.1
  have c81224 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.1
  have c81225 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.1
  have c81226 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81227 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_10 := arithEq_of_rows h (row := 1320) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_10
  simp only [← c81222, ← c81223, ← c81224] at e_1320_10
  have e_1320_11 := arithEq_of_rows h (row := 1320) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_11
  simp only [← c81225, ← c81226, k2, ← c81227] at e_1320_11
  have hr := e_1320_11
  simp only [e_1320_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12211)) := by
  have c81228 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81229 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81230 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81231 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81232 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81233 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_12 := arithEq_of_rows h (row := 1320) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_12
  simp only [← c81228, ← c81229, ← c81230] at e_1320_12
  have e_1320_13 := arithEq_of_rows h (row := 1320) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_13
  simp only [← c81231, ← c81232, k2, ← c81233] at e_1320_13
  have hr := e_1320_13
  simp only [e_1320_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12212)) := by
  have c81234 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81235 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81236 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81237 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81238 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81239 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_14 := arithEq_of_rows h (row := 1320) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_14
  simp only [← c81234, ← c81235, ← c81236] at e_1320_14
  have e_1320_15 := arithEq_of_rows h (row := 1320) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_15
  simp only [← c81237, ← c81238, k2, ← c81239] at e_1320_15
  have hr := e_1320_15
  simp only [e_1320_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12213)) := by
  have c81240 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81241 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81242 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81243 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81244 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81245 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_16 := arithEq_of_rows h (row := 1320) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_16
  simp only [← c81240, ← c81241, ← c81242] at e_1320_16
  have e_1320_17 := arithEq_of_rows h (row := 1320) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_17
  simp only [← c81243, ← c81244, k2, ← c81245] at e_1320_17
  have hr := e_1320_17
  simp only [e_1320_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1320 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12214)) := by
  have c81246 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81247 := (publicBatchWrapper89_copies2538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81248 := (publicBatchWrapper89_copies2539 a h).1
  have c81249 := (publicBatchWrapper89_copies2539 a h).2.1
  have c81250 := (publicBatchWrapper89_copies2539 a h).2.2.1
  have c81251 := (publicBatchWrapper89_copies2539 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1320_18 := arithEq_of_rows h (row := 1320) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_18
  simp only [← c81246, ← c81247, ← c81248] at e_1320_18
  have e_1320_19 := arithEq_of_rows h (row := 1320) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1320_19
  simp only [← c81249, ← c81250, k2, ← c81251] at e_1320_19
  have hr := e_1320_19
  simp only [e_1320_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12215)) := by
  have c81252 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.1
  have c81253 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.1
  have c81254 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.1
  have c81255 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.1
  have c81256 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.1
  have c81257 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_0 := arithEq_of_rows h (row := 1321) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_0
  simp only [← c81252, ← c81253, ← c81254] at e_1321_0
  have e_1321_1 := arithEq_of_rows h (row := 1321) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_1
  simp only [← c81255, ← c81256, k2, ← c81257] at e_1321_1
  have hr := e_1321_1
  simp only [e_1321_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12216)) := by
  have c81258 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81259 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81260 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81261 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81262 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81263 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_2 := arithEq_of_rows h (row := 1321) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_2
  simp only [← c81258, ← c81259, ← c81260] at e_1321_2
  have e_1321_3 := arithEq_of_rows h (row := 1321) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_3
  simp only [← c81261, ← c81262, k2, ← c81263] at e_1321_3
  have hr := e_1321_3
  simp only [e_1321_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12217)) := by
  have c81264 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81265 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81266 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81267 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81268 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81269 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_4 := arithEq_of_rows h (row := 1321) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_4
  simp only [← c81264, ← c81265, ← c81266] at e_1321_4
  have e_1321_5 := arithEq_of_rows h (row := 1321) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_5
  simp only [← c81267, ← c81268, k2, ← c81269] at e_1321_5
  have hr := e_1321_5
  simp only [e_1321_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12218)) := by
  have c81270 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81271 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81272 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81273 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81274 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81275 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_6 := arithEq_of_rows h (row := 1321) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_6
  simp only [← c81270, ← c81271, ← c81272] at e_1321_6
  have e_1321_7 := arithEq_of_rows h (row := 1321) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_7
  simp only [← c81273, ← c81274, k2, ← c81275] at e_1321_7
  have hr := e_1321_7
  simp only [e_1321_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12219)) := by
  have c81276 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81277 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81278 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81279 := (publicBatchWrapper89_copies2539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81280 := (publicBatchWrapper89_copies2540 a h).1
  have c81281 := (publicBatchWrapper89_copies2540 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_8 := arithEq_of_rows h (row := 1321) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_8
  simp only [← c81276, ← c81277, ← c81278] at e_1321_8
  have e_1321_9 := arithEq_of_rows h (row := 1321) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_9
  simp only [← c81279, ← c81280, k2, ← c81281] at e_1321_9
  have hr := e_1321_9
  simp only [e_1321_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12220)) := by
  have c81282 := (publicBatchWrapper89_copies2540 a h).2.2.1
  have c81283 := (publicBatchWrapper89_copies2540 a h).2.2.2.1
  have c81284 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.1
  have c81285 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.1
  have c81286 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.1
  have c81287 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_10 := arithEq_of_rows h (row := 1321) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_10
  simp only [← c81282, ← c81283, ← c81284] at e_1321_10
  have e_1321_11 := arithEq_of_rows h (row := 1321) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_11
  simp only [← c81285, ← c81286, k2, ← c81287] at e_1321_11
  have hr := e_1321_11
  simp only [e_1321_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12221)) := by
  have c81288 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.1
  have c81289 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.1
  have c81290 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81291 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81292 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81293 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_12 := arithEq_of_rows h (row := 1321) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_12
  simp only [← c81288, ← c81289, ← c81290] at e_1321_12
  have e_1321_13 := arithEq_of_rows h (row := 1321) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_13
  simp only [← c81291, ← c81292, k2, ← c81293] at e_1321_13
  have hr := e_1321_13
  simp only [e_1321_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12222)) := by
  have c81294 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81295 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81296 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81297 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81298 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81299 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_14 := arithEq_of_rows h (row := 1321) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_14
  simp only [← c81294, ← c81295, ← c81296] at e_1321_14
  have e_1321_15 := arithEq_of_rows h (row := 1321) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_15
  simp only [← c81297, ← c81298, k2, ← c81299] at e_1321_15
  have hr := e_1321_15
  simp only [e_1321_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12223)) := by
  have c81300 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81301 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81302 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81303 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81304 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81305 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_16 := arithEq_of_rows h (row := 1321) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_16
  simp only [← c81300, ← c81301, ← c81302] at e_1321_16
  have e_1321_17 := arithEq_of_rows h (row := 1321) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_17
  simp only [← c81303, ← c81304, k2, ← c81305] at e_1321_17
  have hr := e_1321_17
  simp only [e_1321_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1321 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12224)) := by
  have c81306 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81307 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81308 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81309 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81310 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81311 := (publicBatchWrapper89_copies2540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1321_18 := arithEq_of_rows h (row := 1321) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_18
  simp only [← c81306, ← c81307, ← c81308] at e_1321_18
  have e_1321_19 := arithEq_of_rows h (row := 1321) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1321_19
  simp only [← c81309, ← c81310, k2, ← c81311] at e_1321_19
  have hr := e_1321_19
  simp only [e_1321_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12225)) := by
  have c81312 := (publicBatchWrapper89_copies2541 a h).1
  have c81313 := (publicBatchWrapper89_copies2541 a h).2.1
  have c81314 := (publicBatchWrapper89_copies2541 a h).2.2.1
  have c81315 := (publicBatchWrapper89_copies2541 a h).2.2.2.1
  have c81316 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.1
  have c81317 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_0 := arithEq_of_rows h (row := 1322) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_0
  simp only [← c81312, ← c81313, ← c81314] at e_1322_0
  have e_1322_1 := arithEq_of_rows h (row := 1322) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_1
  simp only [← c81315, ← c81316, k2, ← c81317] at e_1322_1
  have hr := e_1322_1
  simp only [e_1322_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12226)) := by
  have c81318 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.1
  have c81319 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.1
  have c81320 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.1
  have c81321 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.1
  have c81322 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81323 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_2 := arithEq_of_rows h (row := 1322) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_2
  simp only [← c81318, ← c81319, ← c81320] at e_1322_2
  have e_1322_3 := arithEq_of_rows h (row := 1322) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_3
  simp only [← c81321, ← c81322, k2, ← c81323] at e_1322_3
  have hr := e_1322_3
  simp only [e_1322_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12227)) := by
  have c81324 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81325 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81326 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81327 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81328 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81329 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_4 := arithEq_of_rows h (row := 1322) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_4
  simp only [← c81324, ← c81325, ← c81326] at e_1322_4
  have e_1322_5 := arithEq_of_rows h (row := 1322) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_5
  simp only [← c81327, ← c81328, k2, ← c81329] at e_1322_5
  have hr := e_1322_5
  simp only [e_1322_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12228)) := by
  have c81330 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81331 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81332 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81333 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81334 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81335 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_6 := arithEq_of_rows h (row := 1322) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_6
  simp only [← c81330, ← c81331, ← c81332] at e_1322_6
  have e_1322_7 := arithEq_of_rows h (row := 1322) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_7
  simp only [← c81333, ← c81334, k2, ← c81335] at e_1322_7
  have hr := e_1322_7
  simp only [e_1322_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12229)) := by
  have c81336 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81337 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81338 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81339 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81340 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81341 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_8 := arithEq_of_rows h (row := 1322) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_8
  simp only [← c81336, ← c81337, ← c81338] at e_1322_8
  have e_1322_9 := arithEq_of_rows h (row := 1322) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_9
  simp only [← c81339, ← c81340, k2, ← c81341] at e_1322_9
  have hr := e_1322_9
  simp only [e_1322_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12230)) := by
  have c81342 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81343 := (publicBatchWrapper89_copies2541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81344 := (publicBatchWrapper89_copies2542 a h).1
  have c81345 := (publicBatchWrapper89_copies2542 a h).2.1
  have c81346 := (publicBatchWrapper89_copies2542 a h).2.2.1
  have c81347 := (publicBatchWrapper89_copies2542 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_10 := arithEq_of_rows h (row := 1322) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_10
  simp only [← c81342, ← c81343, ← c81344] at e_1322_10
  have e_1322_11 := arithEq_of_rows h (row := 1322) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_11
  simp only [← c81345, ← c81346, k2, ← c81347] at e_1322_11
  have hr := e_1322_11
  simp only [e_1322_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12231)) := by
  have c81348 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.1
  have c81349 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.1
  have c81350 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.1
  have c81351 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.1
  have c81352 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.1
  have c81353 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_12 := arithEq_of_rows h (row := 1322) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_12
  simp only [← c81348, ← c81349, ← c81350] at e_1322_12
  have e_1322_13 := arithEq_of_rows h (row := 1322) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_13
  simp only [← c81351, ← c81352, k2, ← c81353] at e_1322_13
  have hr := e_1322_13
  simp only [e_1322_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12232)) := by
  have c81354 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81355 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81356 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81357 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81358 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81359 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_14 := arithEq_of_rows h (row := 1322) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_14
  simp only [← c81354, ← c81355, ← c81356] at e_1322_14
  have e_1322_15 := arithEq_of_rows h (row := 1322) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_15
  simp only [← c81357, ← c81358, k2, ← c81359] at e_1322_15
  have hr := e_1322_15
  simp only [e_1322_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12233)) := by
  have c81360 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81361 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81362 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81363 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81364 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81365 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_16 := arithEq_of_rows h (row := 1322) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_16
  simp only [← c81360, ← c81361, ← c81362] at e_1322_16
  have e_1322_17 := arithEq_of_rows h (row := 1322) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_17
  simp only [← c81363, ← c81364, k2, ← c81365] at e_1322_17
  have hr := e_1322_17
  simp only [e_1322_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1322 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12234)) := by
  have c81366 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81367 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81368 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81369 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81370 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81371 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1322_18 := arithEq_of_rows h (row := 1322) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_18
  simp only [← c81366, ← c81367, ← c81368] at e_1322_18
  have e_1322_19 := arithEq_of_rows h (row := 1322) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1322_19
  simp only [← c81369, ← c81370, k2, ← c81371] at e_1322_19
  have hr := e_1322_19
  simp only [e_1322_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12235)) := by
  have c81372 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81373 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81374 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81375 := (publicBatchWrapper89_copies2542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81376 := (publicBatchWrapper89_copies2543 a h).1
  have c81377 := (publicBatchWrapper89_copies2543 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_0 := arithEq_of_rows h (row := 1323) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_0
  simp only [← c81372, ← c81373, ← c81374] at e_1323_0
  have e_1323_1 := arithEq_of_rows h (row := 1323) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_1
  simp only [← c81375, ← c81376, k2, ← c81377] at e_1323_1
  have hr := e_1323_1
  simp only [e_1323_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12236)) := by
  have c81378 := (publicBatchWrapper89_copies2543 a h).2.2.1
  have c81379 := (publicBatchWrapper89_copies2543 a h).2.2.2.1
  have c81380 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.1
  have c81381 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.1
  have c81382 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.1
  have c81383 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_2 := arithEq_of_rows h (row := 1323) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_2
  simp only [← c81378, ← c81379, ← c81380] at e_1323_2
  have e_1323_3 := arithEq_of_rows h (row := 1323) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_3
  simp only [← c81381, ← c81382, k2, ← c81383] at e_1323_3
  have hr := e_1323_3
  simp only [e_1323_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12237)) := by
  have c81384 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.1
  have c81385 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.1
  have c81386 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81387 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81388 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81389 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_4 := arithEq_of_rows h (row := 1323) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_4
  simp only [← c81384, ← c81385, ← c81386] at e_1323_4
  have e_1323_5 := arithEq_of_rows h (row := 1323) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_5
  simp only [← c81387, ← c81388, k2, ← c81389] at e_1323_5
  have hr := e_1323_5
  simp only [e_1323_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12238)) := by
  have c81390 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81391 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81392 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81393 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81394 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81395 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_6 := arithEq_of_rows h (row := 1323) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_6
  simp only [← c81390, ← c81391, ← c81392] at e_1323_6
  have e_1323_7 := arithEq_of_rows h (row := 1323) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_7
  simp only [← c81393, ← c81394, k2, ← c81395] at e_1323_7
  have hr := e_1323_7
  simp only [e_1323_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12239)) := by
  have c81396 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81397 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81398 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81399 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81400 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81401 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_8 := arithEq_of_rows h (row := 1323) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_8
  simp only [← c81396, ← c81397, ← c81398] at e_1323_8
  have e_1323_9 := arithEq_of_rows h (row := 1323) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_9
  simp only [← c81399, ← c81400, k2, ← c81401] at e_1323_9
  have hr := e_1323_9
  simp only [e_1323_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12240)) := by
  have c81402 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81403 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81404 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81405 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81406 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81407 := (publicBatchWrapper89_copies2543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_10 := arithEq_of_rows h (row := 1323) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_10
  simp only [← c81402, ← c81403, ← c81404] at e_1323_10
  have e_1323_11 := arithEq_of_rows h (row := 1323) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_11
  simp only [← c81405, ← c81406, k2, ← c81407] at e_1323_11
  have hr := e_1323_11
  simp only [e_1323_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12241)) := by
  have c81408 := (publicBatchWrapper89_copies2544 a h).1
  have c81409 := (publicBatchWrapper89_copies2544 a h).2.1
  have c81410 := (publicBatchWrapper89_copies2544 a h).2.2.1
  have c81411 := (publicBatchWrapper89_copies2544 a h).2.2.2.1
  have c81412 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.1
  have c81413 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_12 := arithEq_of_rows h (row := 1323) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_12
  simp only [← c81408, ← c81409, ← c81410] at e_1323_12
  have e_1323_13 := arithEq_of_rows h (row := 1323) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_13
  simp only [← c81411, ← c81412, k2, ← c81413] at e_1323_13
  have hr := e_1323_13
  simp only [e_1323_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12242)) := by
  have c81414 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.1
  have c81415 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.1
  have c81416 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.1
  have c81417 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.1
  have c81418 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81419 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_14 := arithEq_of_rows h (row := 1323) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_14
  simp only [← c81414, ← c81415, ← c81416] at e_1323_14
  have e_1323_15 := arithEq_of_rows h (row := 1323) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_15
  simp only [← c81417, ← c81418, k2, ← c81419] at e_1323_15
  have hr := e_1323_15
  simp only [e_1323_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12243)) := by
  have c81420 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81421 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81422 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81423 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81424 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81425 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_16 := arithEq_of_rows h (row := 1323) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_16
  simp only [← c81420, ← c81421, ← c81422] at e_1323_16
  have e_1323_17 := arithEq_of_rows h (row := 1323) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_17
  simp only [← c81423, ← c81424, k2, ← c81425] at e_1323_17
  have hr := e_1323_17
  simp only [e_1323_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1323 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12244)) := by
  have c81426 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81427 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81428 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81429 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81430 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81431 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1323_18 := arithEq_of_rows h (row := 1323) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_18
  simp only [← c81426, ← c81427, ← c81428] at e_1323_18
  have e_1323_19 := arithEq_of_rows h (row := 1323) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1323_19
  simp only [← c81429, ← c81430, k2, ← c81431] at e_1323_19
  have hr := e_1323_19
  simp only [e_1323_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12245)) := by
  have c81432 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81433 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81434 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81435 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81436 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81437 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_0 := arithEq_of_rows h (row := 1324) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_0
  simp only [← c81432, ← c81433, ← c81434] at e_1324_0
  have e_1324_1 := arithEq_of_rows h (row := 1324) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_1
  simp only [← c81435, ← c81436, k2, ← c81437] at e_1324_1
  have hr := e_1324_1
  simp only [e_1324_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12246)) := by
  have c81438 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81439 := (publicBatchWrapper89_copies2544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81440 := (publicBatchWrapper89_copies2545 a h).1
  have c81441 := (publicBatchWrapper89_copies2545 a h).2.1
  have c81442 := (publicBatchWrapper89_copies2545 a h).2.2.1
  have c81443 := (publicBatchWrapper89_copies2545 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_2 := arithEq_of_rows h (row := 1324) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_2
  simp only [← c81438, ← c81439, ← c81440] at e_1324_2
  have e_1324_3 := arithEq_of_rows h (row := 1324) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_3
  simp only [← c81441, ← c81442, k2, ← c81443] at e_1324_3
  have hr := e_1324_3
  simp only [e_1324_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12247)) := by
  have c81444 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.1
  have c81445 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.1
  have c81446 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.1
  have c81447 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.1
  have c81448 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.1
  have c81449 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_4 := arithEq_of_rows h (row := 1324) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_4
  simp only [← c81444, ← c81445, ← c81446] at e_1324_4
  have e_1324_5 := arithEq_of_rows h (row := 1324) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_5
  simp only [← c81447, ← c81448, k2, ← c81449] at e_1324_5
  have hr := e_1324_5
  simp only [e_1324_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12248)) := by
  have c81450 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81451 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81452 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81453 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81454 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81455 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_6 := arithEq_of_rows h (row := 1324) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_6
  simp only [← c81450, ← c81451, ← c81452] at e_1324_6
  have e_1324_7 := arithEq_of_rows h (row := 1324) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_7
  simp only [← c81453, ← c81454, k2, ← c81455] at e_1324_7
  have hr := e_1324_7
  simp only [e_1324_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12249)) := by
  have c81456 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81457 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81458 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81459 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81460 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81461 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_8 := arithEq_of_rows h (row := 1324) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_8
  simp only [← c81456, ← c81457, ← c81458] at e_1324_8
  have e_1324_9 := arithEq_of_rows h (row := 1324) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_9
  simp only [← c81459, ← c81460, k2, ← c81461] at e_1324_9
  have hr := e_1324_9
  simp only [e_1324_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12250)) := by
  have c81462 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81463 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81464 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81465 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81466 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81467 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_10 := arithEq_of_rows h (row := 1324) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_10
  simp only [← c81462, ← c81463, ← c81464] at e_1324_10
  have e_1324_11 := arithEq_of_rows h (row := 1324) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_11
  simp only [← c81465, ← c81466, k2, ← c81467] at e_1324_11
  have hr := e_1324_11
  simp only [e_1324_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12251)) := by
  have c81468 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81469 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81470 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81471 := (publicBatchWrapper89_copies2545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81472 := (publicBatchWrapper89_copies2546 a h).1
  have c81473 := (publicBatchWrapper89_copies2546 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_12 := arithEq_of_rows h (row := 1324) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_12
  simp only [← c81468, ← c81469, ← c81470] at e_1324_12
  have e_1324_13 := arithEq_of_rows h (row := 1324) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_13
  simp only [← c81471, ← c81472, k2, ← c81473] at e_1324_13
  have hr := e_1324_13
  simp only [e_1324_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12252)) := by
  have c81474 := (publicBatchWrapper89_copies2546 a h).2.2.1
  have c81475 := (publicBatchWrapper89_copies2546 a h).2.2.2.1
  have c81476 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.1
  have c81477 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.1
  have c81478 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.1
  have c81479 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_14 := arithEq_of_rows h (row := 1324) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_14
  simp only [← c81474, ← c81475, ← c81476] at e_1324_14
  have e_1324_15 := arithEq_of_rows h (row := 1324) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_15
  simp only [← c81477, ← c81478, k2, ← c81479] at e_1324_15
  have hr := e_1324_15
  simp only [e_1324_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12253)) := by
  have c81480 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.1
  have c81481 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.1
  have c81482 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81483 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81484 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81485 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_16 := arithEq_of_rows h (row := 1324) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_16
  simp only [← c81480, ← c81481, ← c81482] at e_1324_16
  have e_1324_17 := arithEq_of_rows h (row := 1324) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_17
  simp only [← c81483, ← c81484, k2, ← c81485] at e_1324_17
  have hr := e_1324_17
  simp only [e_1324_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1324 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12254)) := by
  have c81486 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81487 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81488 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81489 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81490 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81491 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1324_18 := arithEq_of_rows h (row := 1324) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_18
  simp only [← c81486, ← c81487, ← c81488] at e_1324_18
  have e_1324_19 := arithEq_of_rows h (row := 1324) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1324_19
  simp only [← c81489, ← c81490, k2, ← c81491] at e_1324_19
  have hr := e_1324_19
  simp only [e_1324_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12255)) := by
  have c81492 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81493 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81494 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81495 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81496 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81497 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_0 := arithEq_of_rows h (row := 1325) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_0
  simp only [← c81492, ← c81493, ← c81494] at e_1325_0
  have e_1325_1 := arithEq_of_rows h (row := 1325) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_1
  simp only [← c81495, ← c81496, k2, ← c81497] at e_1325_1
  have hr := e_1325_1
  simp only [e_1325_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12256)) := by
  have c81498 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81499 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81500 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81501 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81502 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81503 := (publicBatchWrapper89_copies2546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_2 := arithEq_of_rows h (row := 1325) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_2
  simp only [← c81498, ← c81499, ← c81500] at e_1325_2
  have e_1325_3 := arithEq_of_rows h (row := 1325) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_3
  simp only [← c81501, ← c81502, k2, ← c81503] at e_1325_3
  have hr := e_1325_3
  simp only [e_1325_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12257)) := by
  have c81504 := (publicBatchWrapper89_copies2547 a h).1
  have c81505 := (publicBatchWrapper89_copies2547 a h).2.1
  have c81506 := (publicBatchWrapper89_copies2547 a h).2.2.1
  have c81507 := (publicBatchWrapper89_copies2547 a h).2.2.2.1
  have c81508 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.1
  have c81509 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_4 := arithEq_of_rows h (row := 1325) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_4
  simp only [← c81504, ← c81505, ← c81506] at e_1325_4
  have e_1325_5 := arithEq_of_rows h (row := 1325) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_5
  simp only [← c81507, ← c81508, k2, ← c81509] at e_1325_5
  have hr := e_1325_5
  simp only [e_1325_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12258)) := by
  have c81510 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.1
  have c81511 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.1
  have c81512 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.1
  have c81513 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.1
  have c81514 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81515 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_6 := arithEq_of_rows h (row := 1325) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_6
  simp only [← c81510, ← c81511, ← c81512] at e_1325_6
  have e_1325_7 := arithEq_of_rows h (row := 1325) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_7
  simp only [← c81513, ← c81514, k2, ← c81515] at e_1325_7
  have hr := e_1325_7
  simp only [e_1325_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12259)) := by
  have c81516 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81517 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81518 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81519 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81520 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81521 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_8 := arithEq_of_rows h (row := 1325) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_8
  simp only [← c81516, ← c81517, ← c81518] at e_1325_8
  have e_1325_9 := arithEq_of_rows h (row := 1325) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_9
  simp only [← c81519, ← c81520, k2, ← c81521] at e_1325_9
  have hr := e_1325_9
  simp only [e_1325_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12260)) := by
  have c81522 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81523 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81524 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81525 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81526 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81527 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_10 := arithEq_of_rows h (row := 1325) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_10
  simp only [← c81522, ← c81523, ← c81524] at e_1325_10
  have e_1325_11 := arithEq_of_rows h (row := 1325) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_11
  simp only [← c81525, ← c81526, k2, ← c81527] at e_1325_11
  have hr := e_1325_11
  simp only [e_1325_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12261)) := by
  have c81528 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81529 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81530 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81531 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81532 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81533 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_12 := arithEq_of_rows h (row := 1325) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_12
  simp only [← c81528, ← c81529, ← c81530] at e_1325_12
  have e_1325_13 := arithEq_of_rows h (row := 1325) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_13
  simp only [← c81531, ← c81532, k2, ← c81533] at e_1325_13
  have hr := e_1325_13
  simp only [e_1325_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12262)) := by
  have c81534 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81535 := (publicBatchWrapper89_copies2547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81536 := (publicBatchWrapper89_copies2548 a h).1
  have c81537 := (publicBatchWrapper89_copies2548 a h).2.1
  have c81538 := (publicBatchWrapper89_copies2548 a h).2.2.1
  have c81539 := (publicBatchWrapper89_copies2548 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_14 := arithEq_of_rows h (row := 1325) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_14
  simp only [← c81534, ← c81535, ← c81536] at e_1325_14
  have e_1325_15 := arithEq_of_rows h (row := 1325) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_15
  simp only [← c81537, ← c81538, k2, ← c81539] at e_1325_15
  have hr := e_1325_15
  simp only [e_1325_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12263)) := by
  have c81540 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.1
  have c81541 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.1
  have c81542 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.1
  have c81543 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.1
  have c81544 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.1
  have c81545 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_16 := arithEq_of_rows h (row := 1325) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_16
  simp only [← c81540, ← c81541, ← c81542] at e_1325_16
  have e_1325_17 := arithEq_of_rows h (row := 1325) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_17
  simp only [← c81543, ← c81544, k2, ← c81545] at e_1325_17
  have hr := e_1325_17
  simp only [e_1325_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1325 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12264)) := by
  have c81546 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81547 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81548 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81549 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81550 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81551 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1325_18 := arithEq_of_rows h (row := 1325) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_18
  simp only [← c81546, ← c81547, ← c81548] at e_1325_18
  have e_1325_19 := arithEq_of_rows h (row := 1325) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1325_19
  simp only [← c81549, ← c81550, k2, ← c81551] at e_1325_19
  have hr := e_1325_19
  simp only [e_1325_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12265)) := by
  have c81552 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81553 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81554 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81555 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81556 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81557 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_0 := arithEq_of_rows h (row := 1326) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_0
  simp only [← c81552, ← c81553, ← c81554] at e_1326_0
  have e_1326_1 := arithEq_of_rows h (row := 1326) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_1
  simp only [← c81555, ← c81556, k2, ← c81557] at e_1326_1
  have hr := e_1326_1
  simp only [e_1326_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12266)) := by
  have c81558 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81559 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81560 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81561 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81562 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81563 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_2 := arithEq_of_rows h (row := 1326) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_2
  simp only [← c81558, ← c81559, ← c81560] at e_1326_2
  have e_1326_3 := arithEq_of_rows h (row := 1326) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_3
  simp only [← c81561, ← c81562, k2, ← c81563] at e_1326_3
  have hr := e_1326_3
  simp only [e_1326_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12267)) := by
  have c81564 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81565 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81566 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81567 := (publicBatchWrapper89_copies2548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81568 := (publicBatchWrapper89_copies2549 a h).1
  have c81569 := (publicBatchWrapper89_copies2549 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_4 := arithEq_of_rows h (row := 1326) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_4
  simp only [← c81564, ← c81565, ← c81566] at e_1326_4
  have e_1326_5 := arithEq_of_rows h (row := 1326) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_5
  simp only [← c81567, ← c81568, k2, ← c81569] at e_1326_5
  have hr := e_1326_5
  simp only [e_1326_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12268)) := by
  have c81570 := (publicBatchWrapper89_copies2549 a h).2.2.1
  have c81571 := (publicBatchWrapper89_copies2549 a h).2.2.2.1
  have c81572 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.1
  have c81573 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.1
  have c81574 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.1
  have c81575 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_6 := arithEq_of_rows h (row := 1326) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_6
  simp only [← c81570, ← c81571, ← c81572] at e_1326_6
  have e_1326_7 := arithEq_of_rows h (row := 1326) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_7
  simp only [← c81573, ← c81574, k2, ← c81575] at e_1326_7
  have hr := e_1326_7
  simp only [e_1326_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12269)) := by
  have c81576 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.1
  have c81577 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.1
  have c81578 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81579 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81580 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81581 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_8 := arithEq_of_rows h (row := 1326) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_8
  simp only [← c81576, ← c81577, ← c81578] at e_1326_8
  have e_1326_9 := arithEq_of_rows h (row := 1326) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_9
  simp only [← c81579, ← c81580, k2, ← c81581] at e_1326_9
  have hr := e_1326_9
  simp only [e_1326_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12270)) := by
  have c81582 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81583 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81584 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81585 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81586 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81587 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_10 := arithEq_of_rows h (row := 1326) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_10
  simp only [← c81582, ← c81583, ← c81584] at e_1326_10
  have e_1326_11 := arithEq_of_rows h (row := 1326) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_11
  simp only [← c81585, ← c81586, k2, ← c81587] at e_1326_11
  have hr := e_1326_11
  simp only [e_1326_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12271)) := by
  have c81588 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81589 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81590 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81591 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81592 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81593 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_12 := arithEq_of_rows h (row := 1326) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_12
  simp only [← c81588, ← c81589, ← c81590] at e_1326_12
  have e_1326_13 := arithEq_of_rows h (row := 1326) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_13
  simp only [← c81591, ← c81592, k2, ← c81593] at e_1326_13
  have hr := e_1326_13
  simp only [e_1326_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12272)) := by
  have c81594 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81595 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81596 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81597 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81598 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81599 := (publicBatchWrapper89_copies2549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_14 := arithEq_of_rows h (row := 1326) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_14
  simp only [← c81594, ← c81595, ← c81596] at e_1326_14
  have e_1326_15 := arithEq_of_rows h (row := 1326) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_15
  simp only [← c81597, ← c81598, k2, ← c81599] at e_1326_15
  have hr := e_1326_15
  simp only [e_1326_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12273)) := by
  have c81600 := (publicBatchWrapper89_copies2550 a h).1
  have c81601 := (publicBatchWrapper89_copies2550 a h).2.1
  have c81602 := (publicBatchWrapper89_copies2550 a h).2.2.1
  have c81603 := (publicBatchWrapper89_copies2550 a h).2.2.2.1
  have c81604 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.1
  have c81605 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_16 := arithEq_of_rows h (row := 1326) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_16
  simp only [← c81600, ← c81601, ← c81602] at e_1326_16
  have e_1326_17 := arithEq_of_rows h (row := 1326) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_17
  simp only [← c81603, ← c81604, k2, ← c81605] at e_1326_17
  have hr := e_1326_17
  simp only [e_1326_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1326 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12274)) := by
  have c81606 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.1
  have c81607 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.1
  have c81608 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.1
  have c81609 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.1
  have c81610 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81611 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1326_18 := arithEq_of_rows h (row := 1326) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_18
  simp only [← c81606, ← c81607, ← c81608] at e_1326_18
  have e_1326_19 := arithEq_of_rows h (row := 1326) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1326_19
  simp only [← c81609, ← c81610, k2, ← c81611] at e_1326_19
  have hr := e_1326_19
  simp only [e_1326_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12275)) := by
  have c81612 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81613 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81614 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81615 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81616 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81617 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_0 := arithEq_of_rows h (row := 1327) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_0
  simp only [← c81612, ← c81613, ← c81614] at e_1327_0
  have e_1327_1 := arithEq_of_rows h (row := 1327) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_1
  simp only [← c81615, ← c81616, k2, ← c81617] at e_1327_1
  have hr := e_1327_1
  simp only [e_1327_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12276)) := by
  have c81618 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81619 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81620 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81621 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81622 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81623 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_2 := arithEq_of_rows h (row := 1327) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_2
  simp only [← c81618, ← c81619, ← c81620] at e_1327_2
  have e_1327_3 := arithEq_of_rows h (row := 1327) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_3
  simp only [← c81621, ← c81622, k2, ← c81623] at e_1327_3
  have hr := e_1327_3
  simp only [e_1327_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12277)) := by
  have c81624 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81625 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81626 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81627 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81628 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81629 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_4 := arithEq_of_rows h (row := 1327) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_4
  simp only [← c81624, ← c81625, ← c81626] at e_1327_4
  have e_1327_5 := arithEq_of_rows h (row := 1327) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_5
  simp only [← c81627, ← c81628, k2, ← c81629] at e_1327_5
  have hr := e_1327_5
  simp only [e_1327_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12278)) := by
  have c81630 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81631 := (publicBatchWrapper89_copies2550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81632 := (publicBatchWrapper89_copies2551 a h).1
  have c81633 := (publicBatchWrapper89_copies2551 a h).2.1
  have c81634 := (publicBatchWrapper89_copies2551 a h).2.2.1
  have c81635 := (publicBatchWrapper89_copies2551 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_6 := arithEq_of_rows h (row := 1327) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_6
  simp only [← c81630, ← c81631, ← c81632] at e_1327_6
  have e_1327_7 := arithEq_of_rows h (row := 1327) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_7
  simp only [← c81633, ← c81634, k2, ← c81635] at e_1327_7
  have hr := e_1327_7
  simp only [e_1327_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12279)) := by
  have c81636 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.1
  have c81637 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.1
  have c81638 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.1
  have c81639 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.1
  have c81640 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.1
  have c81641 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_8 := arithEq_of_rows h (row := 1327) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_8
  simp only [← c81636, ← c81637, ← c81638] at e_1327_8
  have e_1327_9 := arithEq_of_rows h (row := 1327) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_9
  simp only [← c81639, ← c81640, k2, ← c81641] at e_1327_9
  have hr := e_1327_9
  simp only [e_1327_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12280)) := by
  have c81642 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81643 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81644 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81645 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81646 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81647 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_10 := arithEq_of_rows h (row := 1327) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_10
  simp only [← c81642, ← c81643, ← c81644] at e_1327_10
  have e_1327_11 := arithEq_of_rows h (row := 1327) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_11
  simp only [← c81645, ← c81646, k2, ← c81647] at e_1327_11
  have hr := e_1327_11
  simp only [e_1327_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12281)) := by
  have c81648 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81649 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81650 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81651 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81652 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81653 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_12 := arithEq_of_rows h (row := 1327) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_12
  simp only [← c81648, ← c81649, ← c81650] at e_1327_12
  have e_1327_13 := arithEq_of_rows h (row := 1327) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_13
  simp only [← c81651, ← c81652, k2, ← c81653] at e_1327_13
  have hr := e_1327_13
  simp only [e_1327_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12282)) := by
  have c81654 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81655 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81656 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81657 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81658 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81659 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_14 := arithEq_of_rows h (row := 1327) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_14
  simp only [← c81654, ← c81655, ← c81656] at e_1327_14
  have e_1327_15 := arithEq_of_rows h (row := 1327) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_15
  simp only [← c81657, ← c81658, k2, ← c81659] at e_1327_15
  have hr := e_1327_15
  simp only [e_1327_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12283)) := by
  have c81660 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81661 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81662 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81663 := (publicBatchWrapper89_copies2551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81664 := (publicBatchWrapper89_copies2552 a h).1
  have c81665 := (publicBatchWrapper89_copies2552 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_16 := arithEq_of_rows h (row := 1327) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_16
  simp only [← c81660, ← c81661, ← c81662] at e_1327_16
  have e_1327_17 := arithEq_of_rows h (row := 1327) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_17
  simp only [← c81663, ← c81664, k2, ← c81665] at e_1327_17
  have hr := e_1327_17
  simp only [e_1327_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1327 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12284)) := by
  have c81666 := (publicBatchWrapper89_copies2552 a h).2.2.1
  have c81667 := (publicBatchWrapper89_copies2552 a h).2.2.2.1
  have c81668 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.1
  have c81669 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.1
  have c81670 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.1
  have c81671 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1327_18 := arithEq_of_rows h (row := 1327) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_18
  simp only [← c81666, ← c81667, ← c81668] at e_1327_18
  have e_1327_19 := arithEq_of_rows h (row := 1327) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1327_19
  simp only [← c81669, ← c81670, k2, ← c81671] at e_1327_19
  have hr := e_1327_19
  simp only [e_1327_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12285)) := by
  have c81672 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.1
  have c81673 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.1
  have c81674 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81675 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81676 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81677 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_0 := arithEq_of_rows h (row := 1328) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_0
  simp only [← c81672, ← c81673, ← c81674] at e_1328_0
  have e_1328_1 := arithEq_of_rows h (row := 1328) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_1
  simp only [← c81675, ← c81676, k2, ← c81677] at e_1328_1
  have hr := e_1328_1
  simp only [e_1328_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12286)) := by
  have c81678 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81679 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81680 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81681 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81682 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81683 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_2 := arithEq_of_rows h (row := 1328) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_2
  simp only [← c81678, ← c81679, ← c81680] at e_1328_2
  have e_1328_3 := arithEq_of_rows h (row := 1328) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_3
  simp only [← c81681, ← c81682, k2, ← c81683] at e_1328_3
  have hr := e_1328_3
  simp only [e_1328_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12287)) := by
  have c81684 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81685 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81686 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81687 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81688 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81689 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_4 := arithEq_of_rows h (row := 1328) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_4
  simp only [← c81684, ← c81685, ← c81686] at e_1328_4
  have e_1328_5 := arithEq_of_rows h (row := 1328) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_5
  simp only [← c81687, ← c81688, k2, ← c81689] at e_1328_5
  have hr := e_1328_5
  simp only [e_1328_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12288)) := by
  have c81690 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81691 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81692 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81693 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81694 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81695 := (publicBatchWrapper89_copies2552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_6 := arithEq_of_rows h (row := 1328) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_6
  simp only [← c81690, ← c81691, ← c81692] at e_1328_6
  have e_1328_7 := arithEq_of_rows h (row := 1328) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_7
  simp only [← c81693, ← c81694, k2, ← c81695] at e_1328_7
  have hr := e_1328_7
  simp only [e_1328_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12289)) := by
  have c81696 := (publicBatchWrapper89_copies2553 a h).1
  have c81697 := (publicBatchWrapper89_copies2553 a h).2.1
  have c81698 := (publicBatchWrapper89_copies2553 a h).2.2.1
  have c81699 := (publicBatchWrapper89_copies2553 a h).2.2.2.1
  have c81700 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.1
  have c81701 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_8 := arithEq_of_rows h (row := 1328) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_8
  simp only [← c81696, ← c81697, ← c81698] at e_1328_8
  have e_1328_9 := arithEq_of_rows h (row := 1328) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_9
  simp only [← c81699, ← c81700, k2, ← c81701] at e_1328_9
  have hr := e_1328_9
  simp only [e_1328_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12290)) := by
  have c81702 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.1
  have c81703 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.1
  have c81704 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.1
  have c81705 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.1
  have c81706 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81707 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_10 := arithEq_of_rows h (row := 1328) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_10
  simp only [← c81702, ← c81703, ← c81704] at e_1328_10
  have e_1328_11 := arithEq_of_rows h (row := 1328) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_11
  simp only [← c81705, ← c81706, k2, ← c81707] at e_1328_11
  have hr := e_1328_11
  simp only [e_1328_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12291)) := by
  have c81708 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81709 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81710 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81711 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81712 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81713 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_12 := arithEq_of_rows h (row := 1328) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_12
  simp only [← c81708, ← c81709, ← c81710] at e_1328_12
  have e_1328_13 := arithEq_of_rows h (row := 1328) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_13
  simp only [← c81711, ← c81712, k2, ← c81713] at e_1328_13
  have hr := e_1328_13
  simp only [e_1328_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12292)) := by
  have c81714 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81715 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81716 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81717 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81718 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81719 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_14 := arithEq_of_rows h (row := 1328) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_14
  simp only [← c81714, ← c81715, ← c81716] at e_1328_14
  have e_1328_15 := arithEq_of_rows h (row := 1328) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_15
  simp only [← c81717, ← c81718, k2, ← c81719] at e_1328_15
  have hr := e_1328_15
  simp only [e_1328_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12293)) := by
  have c81720 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81721 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81722 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81723 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81724 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81725 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_16 := arithEq_of_rows h (row := 1328) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_16
  simp only [← c81720, ← c81721, ← c81722] at e_1328_16
  have e_1328_17 := arithEq_of_rows h (row := 1328) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_17
  simp only [← c81723, ← c81724, k2, ← c81725] at e_1328_17
  have hr := e_1328_17
  simp only [e_1328_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1328 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12294)) := by
  have c81726 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81727 := (publicBatchWrapper89_copies2553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81728 := (publicBatchWrapper89_copies2554 a h).1
  have c81729 := (publicBatchWrapper89_copies2554 a h).2.1
  have c81730 := (publicBatchWrapper89_copies2554 a h).2.2.1
  have c81731 := (publicBatchWrapper89_copies2554 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1328_18 := arithEq_of_rows h (row := 1328) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_18
  simp only [← c81726, ← c81727, ← c81728] at e_1328_18
  have e_1328_19 := arithEq_of_rows h (row := 1328) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1328_19
  simp only [← c81729, ← c81730, k2, ← c81731] at e_1328_19
  have hr := e_1328_19
  simp only [e_1328_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12295)) := by
  have c81732 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.1
  have c81733 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.1
  have c81734 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.1
  have c81735 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.1
  have c81736 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.1
  have c81737 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_0 := arithEq_of_rows h (row := 1329) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_0
  simp only [← c81732, ← c81733, ← c81734] at e_1329_0
  have e_1329_1 := arithEq_of_rows h (row := 1329) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_1
  simp only [← c81735, ← c81736, k2, ← c81737] at e_1329_1
  have hr := e_1329_1
  simp only [e_1329_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12296)) := by
  have c81738 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81739 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81740 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81741 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81742 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81743 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_2 := arithEq_of_rows h (row := 1329) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_2
  simp only [← c81738, ← c81739, ← c81740] at e_1329_2
  have e_1329_3 := arithEq_of_rows h (row := 1329) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_3
  simp only [← c81741, ← c81742, k2, ← c81743] at e_1329_3
  have hr := e_1329_3
  simp only [e_1329_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12297)) := by
  have c81744 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81745 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81746 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81747 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81748 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81749 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_4 := arithEq_of_rows h (row := 1329) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_4
  simp only [← c81744, ← c81745, ← c81746] at e_1329_4
  have e_1329_5 := arithEq_of_rows h (row := 1329) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_5
  simp only [← c81747, ← c81748, k2, ← c81749] at e_1329_5
  have hr := e_1329_5
  simp only [e_1329_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12298)) := by
  have c81750 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81751 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81752 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81753 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81754 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81755 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_6 := arithEq_of_rows h (row := 1329) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_6
  simp only [← c81750, ← c81751, ← c81752] at e_1329_6
  have e_1329_7 := arithEq_of_rows h (row := 1329) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_7
  simp only [← c81753, ← c81754, k2, ← c81755] at e_1329_7
  have hr := e_1329_7
  simp only [e_1329_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12299)) := by
  have c81756 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81757 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81758 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81759 := (publicBatchWrapper89_copies2554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81760 := (publicBatchWrapper89_copies2555 a h).1
  have c81761 := (publicBatchWrapper89_copies2555 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_8 := arithEq_of_rows h (row := 1329) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_8
  simp only [← c81756, ← c81757, ← c81758] at e_1329_8
  have e_1329_9 := arithEq_of_rows h (row := 1329) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_9
  simp only [← c81759, ← c81760, k2, ← c81761] at e_1329_9
  have hr := e_1329_9
  simp only [e_1329_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12300)) := by
  have c81762 := (publicBatchWrapper89_copies2555 a h).2.2.1
  have c81763 := (publicBatchWrapper89_copies2555 a h).2.2.2.1
  have c81764 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.1
  have c81765 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.1
  have c81766 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.1
  have c81767 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_10 := arithEq_of_rows h (row := 1329) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_10
  simp only [← c81762, ← c81763, ← c81764] at e_1329_10
  have e_1329_11 := arithEq_of_rows h (row := 1329) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_11
  simp only [← c81765, ← c81766, k2, ← c81767] at e_1329_11
  have hr := e_1329_11
  simp only [e_1329_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12301)) := by
  have c81768 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.1
  have c81769 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.1
  have c81770 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81771 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81772 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81773 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_12 := arithEq_of_rows h (row := 1329) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_12
  simp only [← c81768, ← c81769, ← c81770] at e_1329_12
  have e_1329_13 := arithEq_of_rows h (row := 1329) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_13
  simp only [← c81771, ← c81772, k2, ← c81773] at e_1329_13
  have hr := e_1329_13
  simp only [e_1329_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12302)) := by
  have c81774 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81775 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81776 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81777 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81778 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81779 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_14 := arithEq_of_rows h (row := 1329) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_14
  simp only [← c81774, ← c81775, ← c81776] at e_1329_14
  have e_1329_15 := arithEq_of_rows h (row := 1329) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_15
  simp only [← c81777, ← c81778, k2, ← c81779] at e_1329_15
  have hr := e_1329_15
  simp only [e_1329_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12303)) := by
  have c81780 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81781 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81782 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81783 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81784 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81785 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_16 := arithEq_of_rows h (row := 1329) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_16
  simp only [← c81780, ← c81781, ← c81782] at e_1329_16
  have e_1329_17 := arithEq_of_rows h (row := 1329) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_17
  simp only [← c81783, ← c81784, k2, ← c81785] at e_1329_17
  have hr := e_1329_17
  simp only [e_1329_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1329 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12304)) := by
  have c81786 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81787 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81788 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81789 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81790 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81791 := (publicBatchWrapper89_copies2555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1329_18 := arithEq_of_rows h (row := 1329) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_18
  simp only [← c81786, ← c81787, ← c81788] at e_1329_18
  have e_1329_19 := arithEq_of_rows h (row := 1329) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1329_19
  simp only [← c81789, ← c81790, k2, ← c81791] at e_1329_19
  have hr := e_1329_19
  simp only [e_1329_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12305)) := by
  have c81792 := (publicBatchWrapper89_copies2556 a h).1
  have c81793 := (publicBatchWrapper89_copies2556 a h).2.1
  have c81794 := (publicBatchWrapper89_copies2556 a h).2.2.1
  have c81795 := (publicBatchWrapper89_copies2556 a h).2.2.2.1
  have c81796 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.1
  have c81797 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_0 := arithEq_of_rows h (row := 1330) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_0
  simp only [← c81792, ← c81793, ← c81794] at e_1330_0
  have e_1330_1 := arithEq_of_rows h (row := 1330) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_1
  simp only [← c81795, ← c81796, k2, ← c81797] at e_1330_1
  have hr := e_1330_1
  simp only [e_1330_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12306)) := by
  have c81798 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.1
  have c81799 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.1
  have c81800 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.1
  have c81801 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.1
  have c81802 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81803 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_2 := arithEq_of_rows h (row := 1330) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_2
  simp only [← c81798, ← c81799, ← c81800] at e_1330_2
  have e_1330_3 := arithEq_of_rows h (row := 1330) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_3
  simp only [← c81801, ← c81802, k2, ← c81803] at e_1330_3
  have hr := e_1330_3
  simp only [e_1330_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12307)) := by
  have c81804 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81805 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81806 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81807 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81808 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81809 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_4 := arithEq_of_rows h (row := 1330) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_4
  simp only [← c81804, ← c81805, ← c81806] at e_1330_4
  have e_1330_5 := arithEq_of_rows h (row := 1330) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_5
  simp only [← c81807, ← c81808, k2, ← c81809] at e_1330_5
  have hr := e_1330_5
  simp only [e_1330_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12308)) := by
  have c81810 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81811 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81812 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81813 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81814 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81815 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_6 := arithEq_of_rows h (row := 1330) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_6
  simp only [← c81810, ← c81811, ← c81812] at e_1330_6
  have e_1330_7 := arithEq_of_rows h (row := 1330) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_7
  simp only [← c81813, ← c81814, k2, ← c81815] at e_1330_7
  have hr := e_1330_7
  simp only [e_1330_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12309)) := by
  have c81816 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81817 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81818 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81819 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81820 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81821 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_8 := arithEq_of_rows h (row := 1330) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_8
  simp only [← c81816, ← c81817, ← c81818] at e_1330_8
  have e_1330_9 := arithEq_of_rows h (row := 1330) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_9
  simp only [← c81819, ← c81820, k2, ← c81821] at e_1330_9
  have hr := e_1330_9
  simp only [e_1330_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12310)) := by
  have c81822 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81823 := (publicBatchWrapper89_copies2556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81824 := (publicBatchWrapper89_copies2557 a h).1
  have c81825 := (publicBatchWrapper89_copies2557 a h).2.1
  have c81826 := (publicBatchWrapper89_copies2557 a h).2.2.1
  have c81827 := (publicBatchWrapper89_copies2557 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_10 := arithEq_of_rows h (row := 1330) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_10
  simp only [← c81822, ← c81823, ← c81824] at e_1330_10
  have e_1330_11 := arithEq_of_rows h (row := 1330) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_11
  simp only [← c81825, ← c81826, k2, ← c81827] at e_1330_11
  have hr := e_1330_11
  simp only [e_1330_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12311)) := by
  have c81828 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.1
  have c81829 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.1
  have c81830 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.1
  have c81831 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.1
  have c81832 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.1
  have c81833 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_12 := arithEq_of_rows h (row := 1330) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_12
  simp only [← c81828, ← c81829, ← c81830] at e_1330_12
  have e_1330_13 := arithEq_of_rows h (row := 1330) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_13
  simp only [← c81831, ← c81832, k2, ← c81833] at e_1330_13
  have hr := e_1330_13
  simp only [e_1330_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12312)) := by
  have c81834 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81835 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81836 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81837 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81838 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81839 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_14 := arithEq_of_rows h (row := 1330) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_14
  simp only [← c81834, ← c81835, ← c81836] at e_1330_14
  have e_1330_15 := arithEq_of_rows h (row := 1330) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_15
  simp only [← c81837, ← c81838, k2, ← c81839] at e_1330_15
  have hr := e_1330_15
  simp only [e_1330_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12313)) := by
  have c81840 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81841 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81842 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81843 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81844 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81845 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_16 := arithEq_of_rows h (row := 1330) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_16
  simp only [← c81840, ← c81841, ← c81842] at e_1330_16
  have e_1330_17 := arithEq_of_rows h (row := 1330) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_17
  simp only [← c81843, ← c81844, k2, ← c81845] at e_1330_17
  have hr := e_1330_17
  simp only [e_1330_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1330 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12314)) := by
  have c81846 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81847 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81848 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81849 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81850 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81851 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1330_18 := arithEq_of_rows h (row := 1330) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_18
  simp only [← c81846, ← c81847, ← c81848] at e_1330_18
  have e_1330_19 := arithEq_of_rows h (row := 1330) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1330_19
  simp only [← c81849, ← c81850, k2, ← c81851] at e_1330_19
  have hr := e_1330_19
  simp only [e_1330_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12315)) := by
  have c81852 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81853 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81854 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81855 := (publicBatchWrapper89_copies2557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81856 := (publicBatchWrapper89_copies2558 a h).1
  have c81857 := (publicBatchWrapper89_copies2558 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_0 := arithEq_of_rows h (row := 1331) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_0
  simp only [← c81852, ← c81853, ← c81854] at e_1331_0
  have e_1331_1 := arithEq_of_rows h (row := 1331) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_1
  simp only [← c81855, ← c81856, k2, ← c81857] at e_1331_1
  have hr := e_1331_1
  simp only [e_1331_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12316)) := by
  have c81858 := (publicBatchWrapper89_copies2558 a h).2.2.1
  have c81859 := (publicBatchWrapper89_copies2558 a h).2.2.2.1
  have c81860 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.1
  have c81861 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.1
  have c81862 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.1
  have c81863 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_2 := arithEq_of_rows h (row := 1331) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_2
  simp only [← c81858, ← c81859, ← c81860] at e_1331_2
  have e_1331_3 := arithEq_of_rows h (row := 1331) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_3
  simp only [← c81861, ← c81862, k2, ← c81863] at e_1331_3
  have hr := e_1331_3
  simp only [e_1331_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12317)) := by
  have c81864 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.1
  have c81865 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.1
  have c81866 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81867 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81868 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81869 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_4 := arithEq_of_rows h (row := 1331) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_4
  simp only [← c81864, ← c81865, ← c81866] at e_1331_4
  have e_1331_5 := arithEq_of_rows h (row := 1331) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_5
  simp only [← c81867, ← c81868, k2, ← c81869] at e_1331_5
  have hr := e_1331_5
  simp only [e_1331_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12318)) := by
  have c81870 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81871 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81872 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81873 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81874 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81875 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_6 := arithEq_of_rows h (row := 1331) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_6
  simp only [← c81870, ← c81871, ← c81872] at e_1331_6
  have e_1331_7 := arithEq_of_rows h (row := 1331) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_7
  simp only [← c81873, ← c81874, k2, ← c81875] at e_1331_7
  have hr := e_1331_7
  simp only [e_1331_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12319)) := by
  have c81876 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81877 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81878 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81879 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81880 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81881 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_8 := arithEq_of_rows h (row := 1331) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_8
  simp only [← c81876, ← c81877, ← c81878] at e_1331_8
  have e_1331_9 := arithEq_of_rows h (row := 1331) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_9
  simp only [← c81879, ← c81880, k2, ← c81881] at e_1331_9
  have hr := e_1331_9
  simp only [e_1331_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12320)) := by
  have c81882 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81883 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81884 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81885 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81886 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81887 := (publicBatchWrapper89_copies2558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_10 := arithEq_of_rows h (row := 1331) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_10
  simp only [← c81882, ← c81883, ← c81884] at e_1331_10
  have e_1331_11 := arithEq_of_rows h (row := 1331) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_11
  simp only [← c81885, ← c81886, k2, ← c81887] at e_1331_11
  have hr := e_1331_11
  simp only [e_1331_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12321)) := by
  have c81888 := (publicBatchWrapper89_copies2559 a h).1
  have c81889 := (publicBatchWrapper89_copies2559 a h).2.1
  have c81890 := (publicBatchWrapper89_copies2559 a h).2.2.1
  have c81891 := (publicBatchWrapper89_copies2559 a h).2.2.2.1
  have c81892 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.1
  have c81893 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_12 := arithEq_of_rows h (row := 1331) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_12
  simp only [← c81888, ← c81889, ← c81890] at e_1331_12
  have e_1331_13 := arithEq_of_rows h (row := 1331) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_13
  simp only [← c81891, ← c81892, k2, ← c81893] at e_1331_13
  have hr := e_1331_13
  simp only [e_1331_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12322)) := by
  have c81894 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.1
  have c81895 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.1
  have c81896 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.1
  have c81897 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.1
  have c81898 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81899 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_14 := arithEq_of_rows h (row := 1331) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_14
  simp only [← c81894, ← c81895, ← c81896] at e_1331_14
  have e_1331_15 := arithEq_of_rows h (row := 1331) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_15
  simp only [← c81897, ← c81898, k2, ← c81899] at e_1331_15
  have hr := e_1331_15
  simp only [e_1331_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12323)) := by
  have c81900 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81901 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81902 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81903 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81904 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81905 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_16 := arithEq_of_rows h (row := 1331) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_16
  simp only [← c81900, ← c81901, ← c81902] at e_1331_16
  have e_1331_17 := arithEq_of_rows h (row := 1331) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_17
  simp only [← c81903, ← c81904, k2, ← c81905] at e_1331_17
  have hr := e_1331_17
  simp only [e_1331_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1331 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12324)) := by
  have c81906 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81907 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81908 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81909 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81910 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81911 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1331_18 := arithEq_of_rows h (row := 1331) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_18
  simp only [← c81906, ← c81907, ← c81908] at e_1331_18
  have e_1331_19 := arithEq_of_rows h (row := 1331) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1331_19
  simp only [← c81909, ← c81910, k2, ← c81911] at e_1331_19
  have hr := e_1331_19
  simp only [e_1331_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12325)) := by
  have c81912 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81913 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81914 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81915 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81916 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81917 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_0 := arithEq_of_rows h (row := 1332) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_0
  simp only [← c81912, ← c81913, ← c81914] at e_1332_0
  have e_1332_1 := arithEq_of_rows h (row := 1332) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_1
  simp only [← c81915, ← c81916, k2, ← c81917] at e_1332_1
  have hr := e_1332_1
  simp only [e_1332_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12326)) := by
  have c81918 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81919 := (publicBatchWrapper89_copies2559 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81920 := (publicBatchWrapper89_copies2560 a h).1
  have c81921 := (publicBatchWrapper89_copies2560 a h).2.1
  have c81922 := (publicBatchWrapper89_copies2560 a h).2.2.1
  have c81923 := (publicBatchWrapper89_copies2560 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_2 := arithEq_of_rows h (row := 1332) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_2
  simp only [← c81918, ← c81919, ← c81920] at e_1332_2
  have e_1332_3 := arithEq_of_rows h (row := 1332) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_3
  simp only [← c81921, ← c81922, k2, ← c81923] at e_1332_3
  have hr := e_1332_3
  simp only [e_1332_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12327)) := by
  have c81924 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.1
  have c81925 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.1
  have c81926 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.1
  have c81927 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.1
  have c81928 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.1
  have c81929 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_4 := arithEq_of_rows h (row := 1332) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_4
  simp only [← c81924, ← c81925, ← c81926] at e_1332_4
  have e_1332_5 := arithEq_of_rows h (row := 1332) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_5
  simp only [← c81927, ← c81928, k2, ← c81929] at e_1332_5
  have hr := e_1332_5
  simp only [e_1332_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12328)) := by
  have c81930 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81931 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81932 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81933 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81934 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81935 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_6 := arithEq_of_rows h (row := 1332) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_6
  simp only [← c81930, ← c81931, ← c81932] at e_1332_6
  have e_1332_7 := arithEq_of_rows h (row := 1332) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_7
  simp only [← c81933, ← c81934, k2, ← c81935] at e_1332_7
  have hr := e_1332_7
  simp only [e_1332_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12329)) := by
  have c81936 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81937 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81938 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81939 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81940 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81941 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_8 := arithEq_of_rows h (row := 1332) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_8
  simp only [← c81936, ← c81937, ← c81938] at e_1332_8
  have e_1332_9 := arithEq_of_rows h (row := 1332) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_9
  simp only [← c81939, ← c81940, k2, ← c81941] at e_1332_9
  have hr := e_1332_9
  simp only [e_1332_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12330)) := by
  have c81942 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81943 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81944 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81945 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81946 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81947 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_10 := arithEq_of_rows h (row := 1332) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_10
  simp only [← c81942, ← c81943, ← c81944] at e_1332_10
  have e_1332_11 := arithEq_of_rows h (row := 1332) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_11
  simp only [← c81945, ← c81946, k2, ← c81947] at e_1332_11
  have hr := e_1332_11
  simp only [e_1332_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12331)) := by
  have c81948 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81949 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81950 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81951 := (publicBatchWrapper89_copies2560 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c81952 := (publicBatchWrapper89_copies2561 a h).1
  have c81953 := (publicBatchWrapper89_copies2561 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_12 := arithEq_of_rows h (row := 1332) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_12
  simp only [← c81948, ← c81949, ← c81950] at e_1332_12
  have e_1332_13 := arithEq_of_rows h (row := 1332) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_13
  simp only [← c81951, ← c81952, k2, ← c81953] at e_1332_13
  have hr := e_1332_13
  simp only [e_1332_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12332)) := by
  have c81954 := (publicBatchWrapper89_copies2561 a h).2.2.1
  have c81955 := (publicBatchWrapper89_copies2561 a h).2.2.2.1
  have c81956 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.1
  have c81957 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.1
  have c81958 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.1
  have c81959 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_14 := arithEq_of_rows h (row := 1332) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_14
  simp only [← c81954, ← c81955, ← c81956] at e_1332_14
  have e_1332_15 := arithEq_of_rows h (row := 1332) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_15
  simp only [← c81957, ← c81958, k2, ← c81959] at e_1332_15
  have hr := e_1332_15
  simp only [e_1332_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12333)) := by
  have c81960 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.1
  have c81961 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.1
  have c81962 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81963 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c81964 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81965 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_16 := arithEq_of_rows h (row := 1332) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_16
  simp only [← c81960, ← c81961, ← c81962] at e_1332_16
  have e_1332_17 := arithEq_of_rows h (row := 1332) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_17
  simp only [← c81963, ← c81964, k2, ← c81965] at e_1332_17
  have hr := e_1332_17
  simp only [e_1332_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1332 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12334)) := by
  have c81966 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81967 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81968 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81969 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81970 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81971 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1332_18 := arithEq_of_rows h (row := 1332) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_18
  simp only [← c81966, ← c81967, ← c81968] at e_1332_18
  have e_1332_19 := arithEq_of_rows h (row := 1332) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1332_19
  simp only [← c81969, ← c81970, k2, ← c81971] at e_1332_19
  have hr := e_1332_19
  simp only [e_1332_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12335)) := by
  have c81972 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81973 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81974 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81975 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81976 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81977 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_0 := arithEq_of_rows h (row := 1333) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_0
  simp only [← c81972, ← c81973, ← c81974] at e_1333_0
  have e_1333_1 := arithEq_of_rows h (row := 1333) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_1
  simp only [← c81975, ← c81976, k2, ← c81977] at e_1333_1
  have hr := e_1333_1
  simp only [e_1333_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12336)) := by
  have c81978 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81979 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81980 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81981 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81982 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81983 := (publicBatchWrapper89_copies2561 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_2 := arithEq_of_rows h (row := 1333) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_2
  simp only [← c81978, ← c81979, ← c81980] at e_1333_2
  have e_1333_3 := arithEq_of_rows h (row := 1333) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_3
  simp only [← c81981, ← c81982, k2, ← c81983] at e_1333_3
  have hr := e_1333_3
  simp only [e_1333_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12337)) := by
  have c81984 := (publicBatchWrapper89_copies2562 a h).1
  have c81985 := (publicBatchWrapper89_copies2562 a h).2.1
  have c81986 := (publicBatchWrapper89_copies2562 a h).2.2.1
  have c81987 := (publicBatchWrapper89_copies2562 a h).2.2.2.1
  have c81988 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.1
  have c81989 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_4 := arithEq_of_rows h (row := 1333) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_4
  simp only [← c81984, ← c81985, ← c81986] at e_1333_4
  have e_1333_5 := arithEq_of_rows h (row := 1333) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_5
  simp only [← c81987, ← c81988, k2, ← c81989] at e_1333_5
  have hr := e_1333_5
  simp only [e_1333_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12338)) := by
  have c81990 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.1
  have c81991 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.1
  have c81992 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.1
  have c81993 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.1
  have c81994 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.1
  have c81995 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_6 := arithEq_of_rows h (row := 1333) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_6
  simp only [← c81990, ← c81991, ← c81992] at e_1333_6
  have e_1333_7 := arithEq_of_rows h (row := 1333) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_7
  simp only [← c81993, ← c81994, k2, ← c81995] at e_1333_7
  have hr := e_1333_7
  simp only [e_1333_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12339)) := by
  have c81996 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81997 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81998 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c81999 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82000 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82001 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_8 := arithEq_of_rows h (row := 1333) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_8
  simp only [← c81996, ← c81997, ← c81998] at e_1333_8
  have e_1333_9 := arithEq_of_rows h (row := 1333) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_9
  simp only [← c81999, ← c82000, k2, ← c82001] at e_1333_9
  have hr := e_1333_9
  simp only [e_1333_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12340)) := by
  have c82002 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82003 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82004 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82005 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82006 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82007 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_10 := arithEq_of_rows h (row := 1333) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_10
  simp only [← c82002, ← c82003, ← c82004] at e_1333_10
  have e_1333_11 := arithEq_of_rows h (row := 1333) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_11
  simp only [← c82005, ← c82006, k2, ← c82007] at e_1333_11
  have hr := e_1333_11
  simp only [e_1333_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12341)) := by
  have c82008 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82009 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82010 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82011 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82012 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82013 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_12 := arithEq_of_rows h (row := 1333) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_12
  simp only [← c82008, ← c82009, ← c82010] at e_1333_12
  have e_1333_13 := arithEq_of_rows h (row := 1333) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_13
  simp only [← c82011, ← c82012, k2, ← c82013] at e_1333_13
  have hr := e_1333_13
  simp only [e_1333_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12342)) := by
  have c82014 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82015 := (publicBatchWrapper89_copies2562 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82016 := (publicBatchWrapper89_copies2563 a h).1
  have c82017 := (publicBatchWrapper89_copies2563 a h).2.1
  have c82018 := (publicBatchWrapper89_copies2563 a h).2.2.1
  have c82019 := (publicBatchWrapper89_copies2563 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_14 := arithEq_of_rows h (row := 1333) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_14
  simp only [← c82014, ← c82015, ← c82016] at e_1333_14
  have e_1333_15 := arithEq_of_rows h (row := 1333) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_15
  simp only [← c82017, ← c82018, k2, ← c82019] at e_1333_15
  have hr := e_1333_15
  simp only [e_1333_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12343)) := by
  have c82020 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.1
  have c82021 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.1
  have c82022 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.1
  have c82023 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.1
  have c82024 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.1
  have c82025 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_16 := arithEq_of_rows h (row := 1333) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_16
  simp only [← c82020, ← c82021, ← c82022] at e_1333_16
  have e_1333_17 := arithEq_of_rows h (row := 1333) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_17
  simp only [← c82023, ← c82024, k2, ← c82025] at e_1333_17
  have hr := e_1333_17
  simp only [e_1333_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1333 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12344)) := by
  have c82026 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82027 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82028 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82029 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82030 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82031 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1333_18 := arithEq_of_rows h (row := 1333) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_18
  simp only [← c82026, ← c82027, ← c82028] at e_1333_18
  have e_1333_19 := arithEq_of_rows h (row := 1333) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1333_19
  simp only [← c82029, ← c82030, k2, ← c82031] at e_1333_19
  have hr := e_1333_19
  simp only [e_1333_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12345)) := by
  have c82032 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82033 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82034 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82035 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82036 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82037 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_0 := arithEq_of_rows h (row := 1334) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_0
  simp only [← c82032, ← c82033, ← c82034] at e_1334_0
  have e_1334_1 := arithEq_of_rows h (row := 1334) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_1
  simp only [← c82035, ← c82036, k2, ← c82037] at e_1334_1
  have hr := e_1334_1
  simp only [e_1334_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12346)) := by
  have c82038 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82039 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82040 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82041 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82042 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82043 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_2 := arithEq_of_rows h (row := 1334) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_2
  simp only [← c82038, ← c82039, ← c82040] at e_1334_2
  have e_1334_3 := arithEq_of_rows h (row := 1334) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_3
  simp only [← c82041, ← c82042, k2, ← c82043] at e_1334_3
  have hr := e_1334_3
  simp only [e_1334_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12347)) := by
  have c82044 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82045 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82046 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82047 := (publicBatchWrapper89_copies2563 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82048 := (publicBatchWrapper89_copies2564 a h).1
  have c82049 := (publicBatchWrapper89_copies2564 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_4 := arithEq_of_rows h (row := 1334) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_4
  simp only [← c82044, ← c82045, ← c82046] at e_1334_4
  have e_1334_5 := arithEq_of_rows h (row := 1334) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_5
  simp only [← c82047, ← c82048, k2, ← c82049] at e_1334_5
  have hr := e_1334_5
  simp only [e_1334_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12348)) := by
  have c82050 := (publicBatchWrapper89_copies2564 a h).2.2.1
  have c82051 := (publicBatchWrapper89_copies2564 a h).2.2.2.1
  have c82052 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.1
  have c82053 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.1
  have c82054 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.1
  have c82055 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_6 := arithEq_of_rows h (row := 1334) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_6
  simp only [← c82050, ← c82051, ← c82052] at e_1334_6
  have e_1334_7 := arithEq_of_rows h (row := 1334) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_7
  simp only [← c82053, ← c82054, k2, ← c82055] at e_1334_7
  have hr := e_1334_7
  simp only [e_1334_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12349)) := by
  have c82056 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.1
  have c82057 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.1
  have c82058 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82059 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82060 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82061 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_8 := arithEq_of_rows h (row := 1334) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_8
  simp only [← c82056, ← c82057, ← c82058] at e_1334_8
  have e_1334_9 := arithEq_of_rows h (row := 1334) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_9
  simp only [← c82059, ← c82060, k2, ← c82061] at e_1334_9
  have hr := e_1334_9
  simp only [e_1334_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12350)) := by
  have c82062 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82063 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82064 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82065 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82066 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82067 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_10 := arithEq_of_rows h (row := 1334) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_10
  simp only [← c82062, ← c82063, ← c82064] at e_1334_10
  have e_1334_11 := arithEq_of_rows h (row := 1334) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_11
  simp only [← c82065, ← c82066, k2, ← c82067] at e_1334_11
  have hr := e_1334_11
  simp only [e_1334_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12351)) := by
  have c82068 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82069 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82070 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82071 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82072 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82073 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_12 := arithEq_of_rows h (row := 1334) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_12
  simp only [← c82068, ← c82069, ← c82070] at e_1334_12
  have e_1334_13 := arithEq_of_rows h (row := 1334) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_13
  simp only [← c82071, ← c82072, k2, ← c82073] at e_1334_13
  have hr := e_1334_13
  simp only [e_1334_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12352)) := by
  have c82074 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82075 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82076 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82077 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82078 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82079 := (publicBatchWrapper89_copies2564 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_14 := arithEq_of_rows h (row := 1334) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_14
  simp only [← c82074, ← c82075, ← c82076] at e_1334_14
  have e_1334_15 := arithEq_of_rows h (row := 1334) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_15
  simp only [← c82077, ← c82078, k2, ← c82079] at e_1334_15
  have hr := e_1334_15
  simp only [e_1334_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12353)) := by
  have c82080 := (publicBatchWrapper89_copies2565 a h).1
  have c82081 := (publicBatchWrapper89_copies2565 a h).2.1
  have c82082 := (publicBatchWrapper89_copies2565 a h).2.2.1
  have c82083 := (publicBatchWrapper89_copies2565 a h).2.2.2.1
  have c82084 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.1
  have c82085 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_16 := arithEq_of_rows h (row := 1334) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_16
  simp only [← c82080, ← c82081, ← c82082] at e_1334_16
  have e_1334_17 := arithEq_of_rows h (row := 1334) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_17
  simp only [← c82083, ← c82084, k2, ← c82085] at e_1334_17
  have hr := e_1334_17
  simp only [e_1334_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1334 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12354)) := by
  have c82086 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.1
  have c82087 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.1
  have c82088 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.1
  have c82089 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.1
  have c82090 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82091 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1334_18 := arithEq_of_rows h (row := 1334) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_18
  simp only [← c82086, ← c82087, ← c82088] at e_1334_18
  have e_1334_19 := arithEq_of_rows h (row := 1334) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1334_19
  simp only [← c82089, ← c82090, k2, ← c82091] at e_1334_19
  have hr := e_1334_19
  simp only [e_1334_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12355)) := by
  have c82092 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82093 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82094 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82095 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82096 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82097 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_0 := arithEq_of_rows h (row := 1335) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_0
  simp only [← c82092, ← c82093, ← c82094] at e_1335_0
  have e_1335_1 := arithEq_of_rows h (row := 1335) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_1
  simp only [← c82095, ← c82096, k2, ← c82097] at e_1335_1
  have hr := e_1335_1
  simp only [e_1335_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12356)) := by
  have c82098 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82099 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82100 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82101 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82102 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82103 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_2 := arithEq_of_rows h (row := 1335) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_2
  simp only [← c82098, ← c82099, ← c82100] at e_1335_2
  have e_1335_3 := arithEq_of_rows h (row := 1335) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_3
  simp only [← c82101, ← c82102, k2, ← c82103] at e_1335_3
  have hr := e_1335_3
  simp only [e_1335_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12357)) := by
  have c82104 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82105 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82106 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82107 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82108 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82109 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_4 := arithEq_of_rows h (row := 1335) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_4
  simp only [← c82104, ← c82105, ← c82106] at e_1335_4
  have e_1335_5 := arithEq_of_rows h (row := 1335) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_5
  simp only [← c82107, ← c82108, k2, ← c82109] at e_1335_5
  have hr := e_1335_5
  simp only [e_1335_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12358)) := by
  have c82110 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82111 := (publicBatchWrapper89_copies2565 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82112 := (publicBatchWrapper89_copies2566 a h).1
  have c82113 := (publicBatchWrapper89_copies2566 a h).2.1
  have c82114 := (publicBatchWrapper89_copies2566 a h).2.2.1
  have c82115 := (publicBatchWrapper89_copies2566 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_6 := arithEq_of_rows h (row := 1335) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_6
  simp only [← c82110, ← c82111, ← c82112] at e_1335_6
  have e_1335_7 := arithEq_of_rows h (row := 1335) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_7
  simp only [← c82113, ← c82114, k2, ← c82115] at e_1335_7
  have hr := e_1335_7
  simp only [e_1335_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12359)) := by
  have c82116 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.1
  have c82117 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.1
  have c82118 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.1
  have c82119 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.1
  have c82120 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.1
  have c82121 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_8 := arithEq_of_rows h (row := 1335) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_8
  simp only [← c82116, ← c82117, ← c82118] at e_1335_8
  have e_1335_9 := arithEq_of_rows h (row := 1335) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_9
  simp only [← c82119, ← c82120, k2, ← c82121] at e_1335_9
  have hr := e_1335_9
  simp only [e_1335_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12360)) := by
  have c82122 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82123 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82124 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82125 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82126 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82127 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_10 := arithEq_of_rows h (row := 1335) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_10
  simp only [← c82122, ← c82123, ← c82124] at e_1335_10
  have e_1335_11 := arithEq_of_rows h (row := 1335) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_11
  simp only [← c82125, ← c82126, k2, ← c82127] at e_1335_11
  have hr := e_1335_11
  simp only [e_1335_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12361)) := by
  have c82128 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82129 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82130 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82131 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82132 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82133 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_12 := arithEq_of_rows h (row := 1335) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_12
  simp only [← c82128, ← c82129, ← c82130] at e_1335_12
  have e_1335_13 := arithEq_of_rows h (row := 1335) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_13
  simp only [← c82131, ← c82132, k2, ← c82133] at e_1335_13
  have hr := e_1335_13
  simp only [e_1335_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12362)) := by
  have c82134 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82135 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82136 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82137 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82138 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82139 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_14 := arithEq_of_rows h (row := 1335) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_14
  simp only [← c82134, ← c82135, ← c82136] at e_1335_14
  have e_1335_15 := arithEq_of_rows h (row := 1335) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_15
  simp only [← c82137, ← c82138, k2, ← c82139] at e_1335_15
  have hr := e_1335_15
  simp only [e_1335_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12363)) := by
  have c82140 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82141 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82142 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82143 := (publicBatchWrapper89_copies2566 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82144 := (publicBatchWrapper89_copies2567 a h).1
  have c82145 := (publicBatchWrapper89_copies2567 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_16 := arithEq_of_rows h (row := 1335) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_16
  simp only [← c82140, ← c82141, ← c82142] at e_1335_16
  have e_1335_17 := arithEq_of_rows h (row := 1335) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_17
  simp only [← c82143, ← c82144, k2, ← c82145] at e_1335_17
  have hr := e_1335_17
  simp only [e_1335_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1335 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12364)) := by
  have c82146 := (publicBatchWrapper89_copies2567 a h).2.2.1
  have c82147 := (publicBatchWrapper89_copies2567 a h).2.2.2.1
  have c82148 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.1
  have c82149 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.1
  have c82150 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.1
  have c82151 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1335_18 := arithEq_of_rows h (row := 1335) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_18
  simp only [← c82146, ← c82147, ← c82148] at e_1335_18
  have e_1335_19 := arithEq_of_rows h (row := 1335) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1335_19
  simp only [← c82149, ← c82150, k2, ← c82151] at e_1335_19
  have hr := e_1335_19
  simp only [e_1335_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12365)) := by
  have c82152 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.1
  have c82153 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.1
  have c82154 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82155 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82156 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82157 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_0 := arithEq_of_rows h (row := 1336) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_0
  simp only [← c82152, ← c82153, ← c82154] at e_1336_0
  have e_1336_1 := arithEq_of_rows h (row := 1336) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_1
  simp only [← c82155, ← c82156, k2, ← c82157] at e_1336_1
  have hr := e_1336_1
  simp only [e_1336_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12366)) := by
  have c82158 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82159 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82160 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82161 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82162 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82163 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_2 := arithEq_of_rows h (row := 1336) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_2
  simp only [← c82158, ← c82159, ← c82160] at e_1336_2
  have e_1336_3 := arithEq_of_rows h (row := 1336) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_3
  simp only [← c82161, ← c82162, k2, ← c82163] at e_1336_3
  have hr := e_1336_3
  simp only [e_1336_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12367)) := by
  have c82164 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82165 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82166 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82167 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82168 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82169 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_4 := arithEq_of_rows h (row := 1336) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_4
  simp only [← c82164, ← c82165, ← c82166] at e_1336_4
  have e_1336_5 := arithEq_of_rows h (row := 1336) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_5
  simp only [← c82167, ← c82168, k2, ← c82169] at e_1336_5
  have hr := e_1336_5
  simp only [e_1336_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12368)) := by
  have c82170 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82171 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82172 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82173 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82174 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82175 := (publicBatchWrapper89_copies2567 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_6 := arithEq_of_rows h (row := 1336) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_6
  simp only [← c82170, ← c82171, ← c82172] at e_1336_6
  have e_1336_7 := arithEq_of_rows h (row := 1336) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_7
  simp only [← c82173, ← c82174, k2, ← c82175] at e_1336_7
  have hr := e_1336_7
  simp only [e_1336_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12369)) := by
  have c82176 := (publicBatchWrapper89_copies2568 a h).1
  have c82177 := (publicBatchWrapper89_copies2568 a h).2.1
  have c82178 := (publicBatchWrapper89_copies2568 a h).2.2.1
  have c82179 := (publicBatchWrapper89_copies2568 a h).2.2.2.1
  have c82180 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.1
  have c82181 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_8 := arithEq_of_rows h (row := 1336) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_8
  simp only [← c82176, ← c82177, ← c82178] at e_1336_8
  have e_1336_9 := arithEq_of_rows h (row := 1336) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_9
  simp only [← c82179, ← c82180, k2, ← c82181] at e_1336_9
  have hr := e_1336_9
  simp only [e_1336_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12410)) := by
  have c82182 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.1
  have c82183 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.1
  have c82184 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.1
  have c82185 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.1
  have c82186 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82187 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_10 := arithEq_of_rows h (row := 1336) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_10
  simp only [← c82182, ← c82183, ← c82184] at e_1336_10
  have e_1336_11 := arithEq_of_rows h (row := 1336) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_11
  simp only [← c82185, ← c82186, k2, ← c82187] at e_1336_11
  have hr := e_1336_11
  simp only [e_1336_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12411)) := by
  have c82188 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82189 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82190 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82191 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82192 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82193 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_12 := arithEq_of_rows h (row := 1336) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_12
  simp only [← c82188, ← c82189, ← c82190] at e_1336_12
  have e_1336_13 := arithEq_of_rows h (row := 1336) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_13
  simp only [← c82191, ← c82192, k2, ← c82193] at e_1336_13
  have hr := e_1336_13
  simp only [e_1336_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12412)) := by
  have c82194 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82195 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82196 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82197 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82198 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82199 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_14 := arithEq_of_rows h (row := 1336) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_14
  simp only [← c82194, ← c82195, ← c82196] at e_1336_14
  have e_1336_15 := arithEq_of_rows h (row := 1336) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_15
  simp only [← c82197, ← c82198, k2, ← c82199] at e_1336_15
  have hr := e_1336_15
  simp only [e_1336_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12413)) := by
  have c82200 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82201 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82202 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82203 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82204 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82205 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_16 := arithEq_of_rows h (row := 1336) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_16
  simp only [← c82200, ← c82201, ← c82202] at e_1336_16
  have e_1336_17 := arithEq_of_rows h (row := 1336) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_17
  simp only [← c82203, ← c82204, k2, ← c82205] at e_1336_17
  have hr := e_1336_17
  simp only [e_1336_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1336 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12414)) := by
  have c82206 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82207 := (publicBatchWrapper89_copies2568 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82208 := (publicBatchWrapper89_copies2569 a h).1
  have c82209 := (publicBatchWrapper89_copies2569 a h).2.1
  have c82210 := (publicBatchWrapper89_copies2569 a h).2.2.1
  have c82211 := (publicBatchWrapper89_copies2569 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1336_18 := arithEq_of_rows h (row := 1336) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_18
  simp only [← c82206, ← c82207, ← c82208] at e_1336_18
  have e_1336_19 := arithEq_of_rows h (row := 1336) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1336_19
  simp only [← c82209, ← c82210, k2, ← c82211] at e_1336_19
  have hr := e_1336_19
  simp only [e_1336_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12415)) := by
  have c82212 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.1
  have c82213 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.1
  have c82214 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.1
  have c82215 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.1
  have c82216 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.1
  have c82217 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_0 := arithEq_of_rows h (row := 1337) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_0
  simp only [← c82212, ← c82213, ← c82214] at e_1337_0
  have e_1337_1 := arithEq_of_rows h (row := 1337) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_1
  simp only [← c82215, ← c82216, k2, ← c82217] at e_1337_1
  have hr := e_1337_1
  simp only [e_1337_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12416)) := by
  have c82218 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82219 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82220 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82221 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82222 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82223 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_2 := arithEq_of_rows h (row := 1337) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_2
  simp only [← c82218, ← c82219, ← c82220] at e_1337_2
  have e_1337_3 := arithEq_of_rows h (row := 1337) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_3
  simp only [← c82221, ← c82222, k2, ← c82223] at e_1337_3
  have hr := e_1337_3
  simp only [e_1337_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12417)) := by
  have c82224 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82225 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82226 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82227 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82228 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82229 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_4 := arithEq_of_rows h (row := 1337) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_4
  simp only [← c82224, ← c82225, ← c82226] at e_1337_4
  have e_1337_5 := arithEq_of_rows h (row := 1337) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_5
  simp only [← c82227, ← c82228, k2, ← c82229] at e_1337_5
  have hr := e_1337_5
  simp only [e_1337_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12418)) := by
  have c82230 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82231 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82232 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82233 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82234 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82235 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_6 := arithEq_of_rows h (row := 1337) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_6
  simp only [← c82230, ← c82231, ← c82232] at e_1337_6
  have e_1337_7 := arithEq_of_rows h (row := 1337) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_7
  simp only [← c82233, ← c82234, k2, ← c82235] at e_1337_7
  have hr := e_1337_7
  simp only [e_1337_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12419)) := by
  have c82236 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82237 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82238 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82239 := (publicBatchWrapper89_copies2569 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82240 := (publicBatchWrapper89_copies2570 a h).1
  have c82241 := (publicBatchWrapper89_copies2570 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_8 := arithEq_of_rows h (row := 1337) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_8
  simp only [← c82236, ← c82237, ← c82238] at e_1337_8
  have e_1337_9 := arithEq_of_rows h (row := 1337) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_9
  simp only [← c82239, ← c82240, k2, ← c82241] at e_1337_9
  have hr := e_1337_9
  simp only [e_1337_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12420)) := by
  have c82242 := (publicBatchWrapper89_copies2570 a h).2.2.1
  have c82243 := (publicBatchWrapper89_copies2570 a h).2.2.2.1
  have c82244 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.1
  have c82245 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.1
  have c82246 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.1
  have c82247 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_10 := arithEq_of_rows h (row := 1337) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_10
  simp only [← c82242, ← c82243, ← c82244] at e_1337_10
  have e_1337_11 := arithEq_of_rows h (row := 1337) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_11
  simp only [← c82245, ← c82246, k2, ← c82247] at e_1337_11
  have hr := e_1337_11
  simp only [e_1337_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12421)) := by
  have c82248 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.1
  have c82249 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.1
  have c82250 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82251 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82252 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82253 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_12 := arithEq_of_rows h (row := 1337) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_12
  simp only [← c82248, ← c82249, ← c82250] at e_1337_12
  have e_1337_13 := arithEq_of_rows h (row := 1337) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_13
  simp only [← c82251, ← c82252, k2, ← c82253] at e_1337_13
  have hr := e_1337_13
  simp only [e_1337_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12422)) := by
  have c82254 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82255 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82256 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82257 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82258 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82259 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_14 := arithEq_of_rows h (row := 1337) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_14
  simp only [← c82254, ← c82255, ← c82256] at e_1337_14
  have e_1337_15 := arithEq_of_rows h (row := 1337) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_15
  simp only [← c82257, ← c82258, k2, ← c82259] at e_1337_15
  have hr := e_1337_15
  simp only [e_1337_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12423)) := by
  have c82260 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82261 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82262 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82263 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82264 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82265 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_16 := arithEq_of_rows h (row := 1337) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_16
  simp only [← c82260, ← c82261, ← c82262] at e_1337_16
  have e_1337_17 := arithEq_of_rows h (row := 1337) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_17
  simp only [← c82263, ← c82264, k2, ← c82265] at e_1337_17
  have hr := e_1337_17
  simp only [e_1337_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1337 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12424)) := by
  have c82266 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82267 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82268 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82269 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82270 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82271 := (publicBatchWrapper89_copies2570 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1337_18 := arithEq_of_rows h (row := 1337) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_18
  simp only [← c82266, ← c82267, ← c82268] at e_1337_18
  have e_1337_19 := arithEq_of_rows h (row := 1337) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1337_19
  simp only [← c82269, ← c82270, k2, ← c82271] at e_1337_19
  have hr := e_1337_19
  simp only [e_1337_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12425)) := by
  have c82272 := (publicBatchWrapper89_copies2571 a h).1
  have c82273 := (publicBatchWrapper89_copies2571 a h).2.1
  have c82274 := (publicBatchWrapper89_copies2571 a h).2.2.1
  have c82275 := (publicBatchWrapper89_copies2571 a h).2.2.2.1
  have c82276 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.1
  have c82277 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_0 := arithEq_of_rows h (row := 1338) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_0
  simp only [← c82272, ← c82273, ← c82274] at e_1338_0
  have e_1338_1 := arithEq_of_rows h (row := 1338) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_1
  simp only [← c82275, ← c82276, k2, ← c82277] at e_1338_1
  have hr := e_1338_1
  simp only [e_1338_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12426)) := by
  have c82278 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.1
  have c82279 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.1
  have c82280 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.1
  have c82281 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.1
  have c82282 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82283 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_2 := arithEq_of_rows h (row := 1338) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_2
  simp only [← c82278, ← c82279, ← c82280] at e_1338_2
  have e_1338_3 := arithEq_of_rows h (row := 1338) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_3
  simp only [← c82281, ← c82282, k2, ← c82283] at e_1338_3
  have hr := e_1338_3
  simp only [e_1338_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12427)) := by
  have c82284 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82285 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82286 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82287 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82288 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82289 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_4 := arithEq_of_rows h (row := 1338) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_4
  simp only [← c82284, ← c82285, ← c82286] at e_1338_4
  have e_1338_5 := arithEq_of_rows h (row := 1338) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_5
  simp only [← c82287, ← c82288, k2, ← c82289] at e_1338_5
  have hr := e_1338_5
  simp only [e_1338_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12428)) := by
  have c82290 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82291 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82292 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82293 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82294 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82295 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_6 := arithEq_of_rows h (row := 1338) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_6
  simp only [← c82290, ← c82291, ← c82292] at e_1338_6
  have e_1338_7 := arithEq_of_rows h (row := 1338) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_7
  simp only [← c82293, ← c82294, k2, ← c82295] at e_1338_7
  have hr := e_1338_7
  simp only [e_1338_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12429)) := by
  have c82296 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82297 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82298 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82299 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82300 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82301 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_8 := arithEq_of_rows h (row := 1338) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_8
  simp only [← c82296, ← c82297, ← c82298] at e_1338_8
  have e_1338_9 := arithEq_of_rows h (row := 1338) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_9
  simp only [← c82299, ← c82300, k2, ← c82301] at e_1338_9
  have hr := e_1338_9
  simp only [e_1338_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12430)) := by
  have c82302 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82303 := (publicBatchWrapper89_copies2571 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82304 := (publicBatchWrapper89_copies2572 a h).1
  have c82305 := (publicBatchWrapper89_copies2572 a h).2.1
  have c82306 := (publicBatchWrapper89_copies2572 a h).2.2.1
  have c82307 := (publicBatchWrapper89_copies2572 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_10 := arithEq_of_rows h (row := 1338) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_10
  simp only [← c82302, ← c82303, ← c82304] at e_1338_10
  have e_1338_11 := arithEq_of_rows h (row := 1338) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_11
  simp only [← c82305, ← c82306, k2, ← c82307] at e_1338_11
  have hr := e_1338_11
  simp only [e_1338_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12431)) := by
  have c82308 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.1
  have c82309 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.1
  have c82310 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.1
  have c82311 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.1
  have c82312 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.1
  have c82313 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_12 := arithEq_of_rows h (row := 1338) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_12
  simp only [← c82308, ← c82309, ← c82310] at e_1338_12
  have e_1338_13 := arithEq_of_rows h (row := 1338) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_13
  simp only [← c82311, ← c82312, k2, ← c82313] at e_1338_13
  have hr := e_1338_13
  simp only [e_1338_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12432)) := by
  have c82314 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82315 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82316 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82317 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82318 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82319 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_14 := arithEq_of_rows h (row := 1338) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_14
  simp only [← c82314, ← c82315, ← c82316] at e_1338_14
  have e_1338_15 := arithEq_of_rows h (row := 1338) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_15
  simp only [← c82317, ← c82318, k2, ← c82319] at e_1338_15
  have hr := e_1338_15
  simp only [e_1338_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12433)) := by
  have c82320 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82321 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82322 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82323 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82324 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82325 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_16 := arithEq_of_rows h (row := 1338) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_16
  simp only [← c82320, ← c82321, ← c82322] at e_1338_16
  have e_1338_17 := arithEq_of_rows h (row := 1338) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_17
  simp only [← c82323, ← c82324, k2, ← c82325] at e_1338_17
  have hr := e_1338_17
  simp only [e_1338_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1338 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12434)) := by
  have c82326 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82327 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82328 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82329 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82330 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82331 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1338_18 := arithEq_of_rows h (row := 1338) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_18
  simp only [← c82326, ← c82327, ← c82328] at e_1338_18
  have e_1338_19 := arithEq_of_rows h (row := 1338) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1338_19
  simp only [← c82329, ← c82330, k2, ← c82331] at e_1338_19
  have hr := e_1338_19
  simp only [e_1338_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12435)) := by
  have c82332 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82333 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82334 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82335 := (publicBatchWrapper89_copies2572 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82336 := (publicBatchWrapper89_copies2573 a h).1
  have c82337 := (publicBatchWrapper89_copies2573 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_0 := arithEq_of_rows h (row := 1339) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_0
  simp only [← c82332, ← c82333, ← c82334] at e_1339_0
  have e_1339_1 := arithEq_of_rows h (row := 1339) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_1
  simp only [← c82335, ← c82336, k2, ← c82337] at e_1339_1
  have hr := e_1339_1
  simp only [e_1339_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12436)) := by
  have c82338 := (publicBatchWrapper89_copies2573 a h).2.2.1
  have c82339 := (publicBatchWrapper89_copies2573 a h).2.2.2.1
  have c82340 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.1
  have c82341 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.1
  have c82342 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.1
  have c82343 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_2 := arithEq_of_rows h (row := 1339) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_2
  simp only [← c82338, ← c82339, ← c82340] at e_1339_2
  have e_1339_3 := arithEq_of_rows h (row := 1339) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_3
  simp only [← c82341, ← c82342, k2, ← c82343] at e_1339_3
  have hr := e_1339_3
  simp only [e_1339_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12437)) := by
  have c82344 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.1
  have c82345 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.1
  have c82346 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82347 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82348 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82349 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_4 := arithEq_of_rows h (row := 1339) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_4
  simp only [← c82344, ← c82345, ← c82346] at e_1339_4
  have e_1339_5 := arithEq_of_rows h (row := 1339) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_5
  simp only [← c82347, ← c82348, k2, ← c82349] at e_1339_5
  have hr := e_1339_5
  simp only [e_1339_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12438)) := by
  have c82350 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82351 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82352 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82353 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82354 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82355 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_6 := arithEq_of_rows h (row := 1339) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_6
  simp only [← c82350, ← c82351, ← c82352] at e_1339_6
  have e_1339_7 := arithEq_of_rows h (row := 1339) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_7
  simp only [← c82353, ← c82354, k2, ← c82355] at e_1339_7
  have hr := e_1339_7
  simp only [e_1339_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12439)) := by
  have c82356 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82357 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82358 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82359 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82360 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82361 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_8 := arithEq_of_rows h (row := 1339) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_8
  simp only [← c82356, ← c82357, ← c82358] at e_1339_8
  have e_1339_9 := arithEq_of_rows h (row := 1339) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_9
  simp only [← c82359, ← c82360, k2, ← c82361] at e_1339_9
  have hr := e_1339_9
  simp only [e_1339_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12440)) := by
  have c82362 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82363 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82364 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82365 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82366 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82367 := (publicBatchWrapper89_copies2573 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_10 := arithEq_of_rows h (row := 1339) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_10
  simp only [← c82362, ← c82363, ← c82364] at e_1339_10
  have e_1339_11 := arithEq_of_rows h (row := 1339) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_11
  simp only [← c82365, ← c82366, k2, ← c82367] at e_1339_11
  have hr := e_1339_11
  simp only [e_1339_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12441)) := by
  have c82368 := (publicBatchWrapper89_copies2574 a h).1
  have c82369 := (publicBatchWrapper89_copies2574 a h).2.1
  have c82370 := (publicBatchWrapper89_copies2574 a h).2.2.1
  have c82371 := (publicBatchWrapper89_copies2574 a h).2.2.2.1
  have c82372 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.1
  have c82373 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_12 := arithEq_of_rows h (row := 1339) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_12
  simp only [← c82368, ← c82369, ← c82370] at e_1339_12
  have e_1339_13 := arithEq_of_rows h (row := 1339) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_13
  simp only [← c82371, ← c82372, k2, ← c82373] at e_1339_13
  have hr := e_1339_13
  simp only [e_1339_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12442)) := by
  have c82374 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.1
  have c82375 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.1
  have c82376 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.1
  have c82377 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.1
  have c82378 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82379 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_14 := arithEq_of_rows h (row := 1339) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_14
  simp only [← c82374, ← c82375, ← c82376] at e_1339_14
  have e_1339_15 := arithEq_of_rows h (row := 1339) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_15
  simp only [← c82377, ← c82378, k2, ← c82379] at e_1339_15
  have hr := e_1339_15
  simp only [e_1339_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12443)) := by
  have c82380 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82381 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82382 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82383 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82384 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82385 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_16 := arithEq_of_rows h (row := 1339) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_16
  simp only [← c82380, ← c82381, ← c82382] at e_1339_16
  have e_1339_17 := arithEq_of_rows h (row := 1339) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_17
  simp only [← c82383, ← c82384, k2, ← c82385] at e_1339_17
  have hr := e_1339_17
  simp only [e_1339_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1339 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12444)) := by
  have c82386 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82387 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82388 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82389 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82390 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82391 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1339_18 := arithEq_of_rows h (row := 1339) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_18
  simp only [← c82386, ← c82387, ← c82388] at e_1339_18
  have e_1339_19 := arithEq_of_rows h (row := 1339) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1339_19
  simp only [← c82389, ← c82390, k2, ← c82391] at e_1339_19
  have hr := e_1339_19
  simp only [e_1339_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12445)) := by
  have c82392 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82393 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82394 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82395 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82396 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82397 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_0 := arithEq_of_rows h (row := 1340) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_0
  simp only [← c82392, ← c82393, ← c82394] at e_1340_0
  have e_1340_1 := arithEq_of_rows h (row := 1340) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_1
  simp only [← c82395, ← c82396, k2, ← c82397] at e_1340_1
  have hr := e_1340_1
  simp only [e_1340_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12446)) := by
  have c82398 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82399 := (publicBatchWrapper89_copies2574 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82400 := (publicBatchWrapper89_copies2575 a h).1
  have c82401 := (publicBatchWrapper89_copies2575 a h).2.1
  have c82402 := (publicBatchWrapper89_copies2575 a h).2.2.1
  have c82403 := (publicBatchWrapper89_copies2575 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_2 := arithEq_of_rows h (row := 1340) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_2
  simp only [← c82398, ← c82399, ← c82400] at e_1340_2
  have e_1340_3 := arithEq_of_rows h (row := 1340) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_3
  simp only [← c82401, ← c82402, k2, ← c82403] at e_1340_3
  have hr := e_1340_3
  simp only [e_1340_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12447)) := by
  have c82404 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.1
  have c82405 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.1
  have c82406 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.1
  have c82407 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.1
  have c82408 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.1
  have c82409 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_4 := arithEq_of_rows h (row := 1340) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_4
  simp only [← c82404, ← c82405, ← c82406] at e_1340_4
  have e_1340_5 := arithEq_of_rows h (row := 1340) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_5
  simp only [← c82407, ← c82408, k2, ← c82409] at e_1340_5
  have hr := e_1340_5
  simp only [e_1340_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12448)) := by
  have c82410 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82411 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82412 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82413 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82414 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82415 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_6 := arithEq_of_rows h (row := 1340) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_6
  simp only [← c82410, ← c82411, ← c82412] at e_1340_6
  have e_1340_7 := arithEq_of_rows h (row := 1340) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_7
  simp only [← c82413, ← c82414, k2, ← c82415] at e_1340_7
  have hr := e_1340_7
  simp only [e_1340_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12449)) := by
  have c82416 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82417 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82418 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82419 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82420 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82421 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_8 := arithEq_of_rows h (row := 1340) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_8
  simp only [← c82416, ← c82417, ← c82418] at e_1340_8
  have e_1340_9 := arithEq_of_rows h (row := 1340) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_9
  simp only [← c82419, ← c82420, k2, ← c82421] at e_1340_9
  have hr := e_1340_9
  simp only [e_1340_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12450)) := by
  have c82422 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82423 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82424 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82425 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82426 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82427 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_10 := arithEq_of_rows h (row := 1340) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_10
  simp only [← c82422, ← c82423, ← c82424] at e_1340_10
  have e_1340_11 := arithEq_of_rows h (row := 1340) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_11
  simp only [← c82425, ← c82426, k2, ← c82427] at e_1340_11
  have hr := e_1340_11
  simp only [e_1340_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12451)) := by
  have c82428 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82429 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82430 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82431 := (publicBatchWrapper89_copies2575 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82432 := (publicBatchWrapper89_copies2576 a h).1
  have c82433 := (publicBatchWrapper89_copies2576 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_12 := arithEq_of_rows h (row := 1340) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_12
  simp only [← c82428, ← c82429, ← c82430] at e_1340_12
  have e_1340_13 := arithEq_of_rows h (row := 1340) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_13
  simp only [← c82431, ← c82432, k2, ← c82433] at e_1340_13
  have hr := e_1340_13
  simp only [e_1340_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12452)) := by
  have c82434 := (publicBatchWrapper89_copies2576 a h).2.2.1
  have c82435 := (publicBatchWrapper89_copies2576 a h).2.2.2.1
  have c82436 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.1
  have c82437 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.1
  have c82438 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.1
  have c82439 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_14 := arithEq_of_rows h (row := 1340) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_14
  simp only [← c82434, ← c82435, ← c82436] at e_1340_14
  have e_1340_15 := arithEq_of_rows h (row := 1340) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_15
  simp only [← c82437, ← c82438, k2, ← c82439] at e_1340_15
  have hr := e_1340_15
  simp only [e_1340_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12453)) := by
  have c82440 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.1
  have c82441 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.1
  have c82442 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82443 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82444 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82445 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_16 := arithEq_of_rows h (row := 1340) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_16
  simp only [← c82440, ← c82441, ← c82442] at e_1340_16
  have e_1340_17 := arithEq_of_rows h (row := 1340) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_17
  simp only [← c82443, ← c82444, k2, ← c82445] at e_1340_17
  have hr := e_1340_17
  simp only [e_1340_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1340 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12454)) := by
  have c82446 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82447 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82448 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82449 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82450 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82451 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1340_18 := arithEq_of_rows h (row := 1340) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_18
  simp only [← c82446, ← c82447, ← c82448] at e_1340_18
  have e_1340_19 := arithEq_of_rows h (row := 1340) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1340_19
  simp only [← c82449, ← c82450, k2, ← c82451] at e_1340_19
  have hr := e_1340_19
  simp only [e_1340_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12455)) := by
  have c82452 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82453 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82454 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82455 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82456 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82457 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_0 := arithEq_of_rows h (row := 1341) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_0
  simp only [← c82452, ← c82453, ← c82454] at e_1341_0
  have e_1341_1 := arithEq_of_rows h (row := 1341) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_1
  simp only [← c82455, ← c82456, k2, ← c82457] at e_1341_1
  have hr := e_1341_1
  simp only [e_1341_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12456)) := by
  have c82458 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82459 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82460 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82461 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82462 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82463 := (publicBatchWrapper89_copies2576 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_2 := arithEq_of_rows h (row := 1341) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_2
  simp only [← c82458, ← c82459, ← c82460] at e_1341_2
  have e_1341_3 := arithEq_of_rows h (row := 1341) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_3
  simp only [← c82461, ← c82462, k2, ← c82463] at e_1341_3
  have hr := e_1341_3
  simp only [e_1341_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12457)) := by
  have c82464 := (publicBatchWrapper89_copies2577 a h).1
  have c82465 := (publicBatchWrapper89_copies2577 a h).2.1
  have c82466 := (publicBatchWrapper89_copies2577 a h).2.2.1
  have c82467 := (publicBatchWrapper89_copies2577 a h).2.2.2.1
  have c82468 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.1
  have c82469 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_4 := arithEq_of_rows h (row := 1341) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_4
  simp only [← c82464, ← c82465, ← c82466] at e_1341_4
  have e_1341_5 := arithEq_of_rows h (row := 1341) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_5
  simp only [← c82467, ← c82468, k2, ← c82469] at e_1341_5
  have hr := e_1341_5
  simp only [e_1341_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12458)) := by
  have c82470 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.1
  have c82471 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.1
  have c82472 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.1
  have c82473 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.1
  have c82474 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82475 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_6 := arithEq_of_rows h (row := 1341) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_6
  simp only [← c82470, ← c82471, ← c82472] at e_1341_6
  have e_1341_7 := arithEq_of_rows h (row := 1341) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_7
  simp only [← c82473, ← c82474, k2, ← c82475] at e_1341_7
  have hr := e_1341_7
  simp only [e_1341_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12459)) := by
  have c82476 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82477 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82478 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82479 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82480 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82481 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_8 := arithEq_of_rows h (row := 1341) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_8
  simp only [← c82476, ← c82477, ← c82478] at e_1341_8
  have e_1341_9 := arithEq_of_rows h (row := 1341) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_9
  simp only [← c82479, ← c82480, k2, ← c82481] at e_1341_9
  have hr := e_1341_9
  simp only [e_1341_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12460)) := by
  have c82482 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82483 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82484 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82485 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82486 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82487 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_10 := arithEq_of_rows h (row := 1341) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_10
  simp only [← c82482, ← c82483, ← c82484] at e_1341_10
  have e_1341_11 := arithEq_of_rows h (row := 1341) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_11
  simp only [← c82485, ← c82486, k2, ← c82487] at e_1341_11
  have hr := e_1341_11
  simp only [e_1341_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12461)) := by
  have c82488 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82489 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82490 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82491 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82492 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82493 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_12 := arithEq_of_rows h (row := 1341) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_12
  simp only [← c82488, ← c82489, ← c82490] at e_1341_12
  have e_1341_13 := arithEq_of_rows h (row := 1341) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_13
  simp only [← c82491, ← c82492, k2, ← c82493] at e_1341_13
  have hr := e_1341_13
  simp only [e_1341_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12462)) := by
  have c82494 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82495 := (publicBatchWrapper89_copies2577 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82496 := (publicBatchWrapper89_copies2578 a h).1
  have c82497 := (publicBatchWrapper89_copies2578 a h).2.1
  have c82498 := (publicBatchWrapper89_copies2578 a h).2.2.1
  have c82499 := (publicBatchWrapper89_copies2578 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_14 := arithEq_of_rows h (row := 1341) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_14
  simp only [← c82494, ← c82495, ← c82496] at e_1341_14
  have e_1341_15 := arithEq_of_rows h (row := 1341) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_15
  simp only [← c82497, ← c82498, k2, ← c82499] at e_1341_15
  have hr := e_1341_15
  simp only [e_1341_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f12999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12463)) := by
  have c82500 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.1
  have c82501 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.1
  have c82502 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.1
  have c82503 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.1
  have c82504 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.1
  have c82505 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_16 := arithEq_of_rows h (row := 1341) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_16
  simp only [← c82500, ← c82501, ← c82502] at e_1341_16
  have e_1341_17 := arithEq_of_rows h (row := 1341) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_17
  simp only [← c82503, ← c82504, k2, ← c82505] at e_1341_17
  have hr := e_1341_17
  simp only [e_1341_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1341 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12464)) := by
  have c82506 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82507 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82508 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82509 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82510 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82511 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1341_18 := arithEq_of_rows h (row := 1341) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_18
  simp only [← c82506, ← c82507, ← c82508] at e_1341_18
  have e_1341_19 := arithEq_of_rows h (row := 1341) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1341_19
  simp only [← c82509, ← c82510, k2, ← c82511] at e_1341_19
  have hr := e_1341_19
  simp only [e_1341_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12465)) := by
  have c82512 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82513 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82514 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82515 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82516 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82517 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_0 := arithEq_of_rows h (row := 1342) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_0
  simp only [← c82512, ← c82513, ← c82514] at e_1342_0
  have e_1342_1 := arithEq_of_rows h (row := 1342) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_1
  simp only [← c82515, ← c82516, k2, ← c82517] at e_1342_1
  have hr := e_1342_1
  simp only [e_1342_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12466)) := by
  have c82518 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82519 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82520 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82521 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82522 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82523 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_2 := arithEq_of_rows h (row := 1342) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_2
  simp only [← c82518, ← c82519, ← c82520] at e_1342_2
  have e_1342_3 := arithEq_of_rows h (row := 1342) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_3
  simp only [← c82521, ← c82522, k2, ← c82523] at e_1342_3
  have hr := e_1342_3
  simp only [e_1342_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12467)) := by
  have c82524 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82525 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82526 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82527 := (publicBatchWrapper89_copies2578 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82528 := (publicBatchWrapper89_copies2579 a h).1
  have c82529 := (publicBatchWrapper89_copies2579 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_4 := arithEq_of_rows h (row := 1342) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_4
  simp only [← c82524, ← c82525, ← c82526] at e_1342_4
  have e_1342_5 := arithEq_of_rows h (row := 1342) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_5
  simp only [← c82527, ← c82528, k2, ← c82529] at e_1342_5
  have hr := e_1342_5
  simp only [e_1342_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12468)) := by
  have c82530 := (publicBatchWrapper89_copies2579 a h).2.2.1
  have c82531 := (publicBatchWrapper89_copies2579 a h).2.2.2.1
  have c82532 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.1
  have c82533 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.1
  have c82534 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.1
  have c82535 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_6 := arithEq_of_rows h (row := 1342) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_6
  simp only [← c82530, ← c82531, ← c82532] at e_1342_6
  have e_1342_7 := arithEq_of_rows h (row := 1342) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_7
  simp only [← c82533, ← c82534, k2, ← c82535] at e_1342_7
  have hr := e_1342_7
  simp only [e_1342_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12469)) := by
  have c82536 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.1
  have c82537 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.1
  have c82538 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82539 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82540 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82541 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_8 := arithEq_of_rows h (row := 1342) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_8
  simp only [← c82536, ← c82537, ← c82538] at e_1342_8
  have e_1342_9 := arithEq_of_rows h (row := 1342) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_9
  simp only [← c82539, ← c82540, k2, ← c82541] at e_1342_9
  have hr := e_1342_9
  simp only [e_1342_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12470)) := by
  have c82542 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82543 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82544 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82545 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82546 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82547 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_10 := arithEq_of_rows h (row := 1342) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_10
  simp only [← c82542, ← c82543, ← c82544] at e_1342_10
  have e_1342_11 := arithEq_of_rows h (row := 1342) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_11
  simp only [← c82545, ← c82546, k2, ← c82547] at e_1342_11
  have hr := e_1342_11
  simp only [e_1342_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12471)) := by
  have c82548 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82549 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82550 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82551 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82552 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82553 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_12 := arithEq_of_rows h (row := 1342) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_12
  simp only [← c82548, ← c82549, ← c82550] at e_1342_12
  have e_1342_13 := arithEq_of_rows h (row := 1342) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_13
  simp only [← c82551, ← c82552, k2, ← c82553] at e_1342_13
  have hr := e_1342_13
  simp only [e_1342_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12472)) := by
  have c82554 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82555 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82556 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82557 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82558 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82559 := (publicBatchWrapper89_copies2579 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_14 := arithEq_of_rows h (row := 1342) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_14
  simp only [← c82554, ← c82555, ← c82556] at e_1342_14
  have e_1342_15 := arithEq_of_rows h (row := 1342) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_15
  simp only [← c82557, ← c82558, k2, ← c82559] at e_1342_15
  have hr := e_1342_15
  simp only [e_1342_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12473)) := by
  have c82560 := (publicBatchWrapper89_copies2580 a h).1
  have c82561 := (publicBatchWrapper89_copies2580 a h).2.1
  have c82562 := (publicBatchWrapper89_copies2580 a h).2.2.1
  have c82563 := (publicBatchWrapper89_copies2580 a h).2.2.2.1
  have c82564 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.1
  have c82565 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_16 := arithEq_of_rows h (row := 1342) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_16
  simp only [← c82560, ← c82561, ← c82562] at e_1342_16
  have e_1342_17 := arithEq_of_rows h (row := 1342) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_17
  simp only [← c82563, ← c82564, k2, ← c82565] at e_1342_17
  have hr := e_1342_17
  simp only [e_1342_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1342 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12474)) := by
  have c82566 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.1
  have c82567 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.1
  have c82568 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.1
  have c82569 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.1
  have c82570 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82571 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1342_18 := arithEq_of_rows h (row := 1342) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_18
  simp only [← c82566, ← c82567, ← c82568] at e_1342_18
  have e_1342_19 := arithEq_of_rows h (row := 1342) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1342_19
  simp only [← c82569, ← c82570, k2, ← c82571] at e_1342_19
  have hr := e_1342_19
  simp only [e_1342_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12475)) := by
  have c82572 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82573 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82574 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82575 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82576 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82577 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_0 := arithEq_of_rows h (row := 1343) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_0
  simp only [← c82572, ← c82573, ← c82574] at e_1343_0
  have e_1343_1 := arithEq_of_rows h (row := 1343) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_1
  simp only [← c82575, ← c82576, k2, ← c82577] at e_1343_1
  have hr := e_1343_1
  simp only [e_1343_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12476)) := by
  have c82578 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82579 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82580 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82581 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82582 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82583 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_2 := arithEq_of_rows h (row := 1343) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_2
  simp only [← c82578, ← c82579, ← c82580] at e_1343_2
  have e_1343_3 := arithEq_of_rows h (row := 1343) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_3
  simp only [← c82581, ← c82582, k2, ← c82583] at e_1343_3
  have hr := e_1343_3
  simp only [e_1343_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12477)) := by
  have c82584 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82585 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82586 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82587 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82588 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82589 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_4 := arithEq_of_rows h (row := 1343) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_4
  simp only [← c82584, ← c82585, ← c82586] at e_1343_4
  have e_1343_5 := arithEq_of_rows h (row := 1343) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_5
  simp only [← c82587, ← c82588, k2, ← c82589] at e_1343_5
  have hr := e_1343_5
  simp only [e_1343_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12478)) := by
  have c82590 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82591 := (publicBatchWrapper89_copies2580 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82592 := (publicBatchWrapper89_copies2581 a h).1
  have c82593 := (publicBatchWrapper89_copies2581 a h).2.1
  have c82594 := (publicBatchWrapper89_copies2581 a h).2.2.1
  have c82595 := (publicBatchWrapper89_copies2581 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_6 := arithEq_of_rows h (row := 1343) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_6
  simp only [← c82590, ← c82591, ← c82592] at e_1343_6
  have e_1343_7 := arithEq_of_rows h (row := 1343) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_7
  simp only [← c82593, ← c82594, k2, ← c82595] at e_1343_7
  have hr := e_1343_7
  simp only [e_1343_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12479)) := by
  have c82596 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.1
  have c82597 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.1
  have c82598 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.1
  have c82599 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.1
  have c82600 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.1
  have c82601 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_8 := arithEq_of_rows h (row := 1343) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_8
  simp only [← c82596, ← c82597, ← c82598] at e_1343_8
  have e_1343_9 := arithEq_of_rows h (row := 1343) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_9
  simp only [← c82599, ← c82600, k2, ← c82601] at e_1343_9
  have hr := e_1343_9
  simp only [e_1343_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12480)) := by
  have c82602 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82603 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82604 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82605 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82606 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82607 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_10 := arithEq_of_rows h (row := 1343) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_10
  simp only [← c82602, ← c82603, ← c82604] at e_1343_10
  have e_1343_11 := arithEq_of_rows h (row := 1343) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_11
  simp only [← c82605, ← c82606, k2, ← c82607] at e_1343_11
  have hr := e_1343_11
  simp only [e_1343_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12481)) := by
  have c82608 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82609 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82610 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82611 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82612 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82613 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_12 := arithEq_of_rows h (row := 1343) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_12
  simp only [← c82608, ← c82609, ← c82610] at e_1343_12
  have e_1343_13 := arithEq_of_rows h (row := 1343) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_13
  simp only [← c82611, ← c82612, k2, ← c82613] at e_1343_13
  have hr := e_1343_13
  simp only [e_1343_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12482)) := by
  have c82614 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82615 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82616 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82617 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82618 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82619 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_14 := arithEq_of_rows h (row := 1343) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_14
  simp only [← c82614, ← c82615, ← c82616] at e_1343_14
  have e_1343_15 := arithEq_of_rows h (row := 1343) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_15
  simp only [← c82617, ← c82618, k2, ← c82619] at e_1343_15
  have hr := e_1343_15
  simp only [e_1343_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12483)) := by
  have c82620 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82621 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82622 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82623 := (publicBatchWrapper89_copies2581 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82624 := (publicBatchWrapper89_copies2582 a h).1
  have c82625 := (publicBatchWrapper89_copies2582 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_16 := arithEq_of_rows h (row := 1343) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_16
  simp only [← c82620, ← c82621, ← c82622] at e_1343_16
  have e_1343_17 := arithEq_of_rows h (row := 1343) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_17
  simp only [← c82623, ← c82624, k2, ← c82625] at e_1343_17
  have hr := e_1343_17
  simp only [e_1343_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1343 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12484)) := by
  have c82626 := (publicBatchWrapper89_copies2582 a h).2.2.1
  have c82627 := (publicBatchWrapper89_copies2582 a h).2.2.2.1
  have c82628 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.1
  have c82629 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.1
  have c82630 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.1
  have c82631 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1343_18 := arithEq_of_rows h (row := 1343) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_18
  simp only [← c82626, ← c82627, ← c82628] at e_1343_18
  have e_1343_19 := arithEq_of_rows h (row := 1343) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1343_19
  simp only [← c82629, ← c82630, k2, ← c82631] at e_1343_19
  have hr := e_1343_19
  simp only [e_1343_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12485)) := by
  have c82632 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.1
  have c82633 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.1
  have c82634 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82635 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82636 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82637 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_0 := arithEq_of_rows h (row := 1344) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_0
  simp only [← c82632, ← c82633, ← c82634] at e_1344_0
  have e_1344_1 := arithEq_of_rows h (row := 1344) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_1
  simp only [← c82635, ← c82636, k2, ← c82637] at e_1344_1
  have hr := e_1344_1
  simp only [e_1344_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12486)) := by
  have c82638 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82639 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82640 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82641 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82642 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82643 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_2 := arithEq_of_rows h (row := 1344) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_2
  simp only [← c82638, ← c82639, ← c82640] at e_1344_2
  have e_1344_3 := arithEq_of_rows h (row := 1344) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_3
  simp only [← c82641, ← c82642, k2, ← c82643] at e_1344_3
  have hr := e_1344_3
  simp only [e_1344_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12487)) := by
  have c82644 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82645 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82646 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82647 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82648 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82649 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_4 := arithEq_of_rows h (row := 1344) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_4
  simp only [← c82644, ← c82645, ← c82646] at e_1344_4
  have e_1344_5 := arithEq_of_rows h (row := 1344) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_5
  simp only [← c82647, ← c82648, k2, ← c82649] at e_1344_5
  have hr := e_1344_5
  simp only [e_1344_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12488)) := by
  have c82650 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82651 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82652 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82653 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82654 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82655 := (publicBatchWrapper89_copies2582 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_6 := arithEq_of_rows h (row := 1344) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_6
  simp only [← c82650, ← c82651, ← c82652] at e_1344_6
  have e_1344_7 := arithEq_of_rows h (row := 1344) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_7
  simp only [← c82653, ← c82654, k2, ← c82655] at e_1344_7
  have hr := e_1344_7
  simp only [e_1344_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12489)) := by
  have c82656 := (publicBatchWrapper89_copies2583 a h).1
  have c82657 := (publicBatchWrapper89_copies2583 a h).2.1
  have c82658 := (publicBatchWrapper89_copies2583 a h).2.2.1
  have c82659 := (publicBatchWrapper89_copies2583 a h).2.2.2.1
  have c82660 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.1
  have c82661 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_8 := arithEq_of_rows h (row := 1344) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_8
  simp only [← c82656, ← c82657, ← c82658] at e_1344_8
  have e_1344_9 := arithEq_of_rows h (row := 1344) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_9
  simp only [← c82659, ← c82660, k2, ← c82661] at e_1344_9
  have hr := e_1344_9
  simp only [e_1344_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12490)) := by
  have c82662 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.1
  have c82663 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.1
  have c82664 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.1
  have c82665 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.1
  have c82666 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82667 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_10 := arithEq_of_rows h (row := 1344) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_10
  simp only [← c82662, ← c82663, ← c82664] at e_1344_10
  have e_1344_11 := arithEq_of_rows h (row := 1344) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_11
  simp only [← c82665, ← c82666, k2, ← c82667] at e_1344_11
  have hr := e_1344_11
  simp only [e_1344_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12491)) := by
  have c82668 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82669 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82670 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82671 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82672 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82673 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_12 := arithEq_of_rows h (row := 1344) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_12
  simp only [← c82668, ← c82669, ← c82670] at e_1344_12
  have e_1344_13 := arithEq_of_rows h (row := 1344) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_13
  simp only [← c82671, ← c82672, k2, ← c82673] at e_1344_13
  have hr := e_1344_13
  simp only [e_1344_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12492)) := by
  have c82674 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82675 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82676 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82677 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82678 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82679 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_14 := arithEq_of_rows h (row := 1344) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_14
  simp only [← c82674, ← c82675, ← c82676] at e_1344_14
  have e_1344_15 := arithEq_of_rows h (row := 1344) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_15
  simp only [← c82677, ← c82678, k2, ← c82679] at e_1344_15
  have hr := e_1344_15
  simp only [e_1344_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12493)) := by
  have c82680 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82681 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82682 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82683 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82684 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82685 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_16 := arithEq_of_rows h (row := 1344) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_16
  simp only [← c82680, ← c82681, ← c82682] at e_1344_16
  have e_1344_17 := arithEq_of_rows h (row := 1344) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_17
  simp only [← c82683, ← c82684, k2, ← c82685] at e_1344_17
  have hr := e_1344_17
  simp only [e_1344_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1344 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12494)) := by
  have c82686 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82687 := (publicBatchWrapper89_copies2583 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82688 := (publicBatchWrapper89_copies2584 a h).1
  have c82689 := (publicBatchWrapper89_copies2584 a h).2.1
  have c82690 := (publicBatchWrapper89_copies2584 a h).2.2.1
  have c82691 := (publicBatchWrapper89_copies2584 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1344_18 := arithEq_of_rows h (row := 1344) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_18
  simp only [← c82686, ← c82687, ← c82688] at e_1344_18
  have e_1344_19 := arithEq_of_rows h (row := 1344) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1344_19
  simp only [← c82689, ← c82690, k2, ← c82691] at e_1344_19
  have hr := e_1344_19
  simp only [e_1344_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12495)) := by
  have c82692 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.1
  have c82693 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.1
  have c82694 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.1
  have c82695 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.1
  have c82696 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.1
  have c82697 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_0 := arithEq_of_rows h (row := 1345) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_0
  simp only [← c82692, ← c82693, ← c82694] at e_1345_0
  have e_1345_1 := arithEq_of_rows h (row := 1345) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_1
  simp only [← c82695, ← c82696, k2, ← c82697] at e_1345_1
  have hr := e_1345_1
  simp only [e_1345_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12496)) := by
  have c82698 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82699 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82700 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82701 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82702 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82703 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_2 := arithEq_of_rows h (row := 1345) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_2
  simp only [← c82698, ← c82699, ← c82700] at e_1345_2
  have e_1345_3 := arithEq_of_rows h (row := 1345) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_3
  simp only [← c82701, ← c82702, k2, ← c82703] at e_1345_3
  have hr := e_1345_3
  simp only [e_1345_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12497)) := by
  have c82704 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82705 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82706 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82707 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82708 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82709 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_4 := arithEq_of_rows h (row := 1345) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_4
  simp only [← c82704, ← c82705, ← c82706] at e_1345_4
  have e_1345_5 := arithEq_of_rows h (row := 1345) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_5
  simp only [← c82707, ← c82708, k2, ← c82709] at e_1345_5
  have hr := e_1345_5
  simp only [e_1345_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12498)) := by
  have c82710 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82711 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82712 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82713 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82714 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82715 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_6 := arithEq_of_rows h (row := 1345) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_6
  simp only [← c82710, ← c82711, ← c82712] at e_1345_6
  have e_1345_7 := arithEq_of_rows h (row := 1345) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_7
  simp only [← c82713, ← c82714, k2, ← c82715] at e_1345_7
  have hr := e_1345_7
  simp only [e_1345_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12499)) := by
  have c82716 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82717 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82718 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82719 := (publicBatchWrapper89_copies2584 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82720 := (publicBatchWrapper89_copies2585 a h).1
  have c82721 := (publicBatchWrapper89_copies2585 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_8 := arithEq_of_rows h (row := 1345) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_8
  simp only [← c82716, ← c82717, ← c82718] at e_1345_8
  have e_1345_9 := arithEq_of_rows h (row := 1345) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_9
  simp only [← c82719, ← c82720, k2, ← c82721] at e_1345_9
  have hr := e_1345_9
  simp only [e_1345_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12500)) := by
  have c82722 := (publicBatchWrapper89_copies2585 a h).2.2.1
  have c82723 := (publicBatchWrapper89_copies2585 a h).2.2.2.1
  have c82724 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.1
  have c82725 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.1
  have c82726 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.1
  have c82727 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_10 := arithEq_of_rows h (row := 1345) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_10
  simp only [← c82722, ← c82723, ← c82724] at e_1345_10
  have e_1345_11 := arithEq_of_rows h (row := 1345) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_11
  simp only [← c82725, ← c82726, k2, ← c82727] at e_1345_11
  have hr := e_1345_11
  simp only [e_1345_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12501)) := by
  have c82728 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.1
  have c82729 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.1
  have c82730 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82731 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82732 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82733 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_12 := arithEq_of_rows h (row := 1345) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_12
  simp only [← c82728, ← c82729, ← c82730] at e_1345_12
  have e_1345_13 := arithEq_of_rows h (row := 1345) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_13
  simp only [← c82731, ← c82732, k2, ← c82733] at e_1345_13
  have hr := e_1345_13
  simp only [e_1345_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12502)) := by
  have c82734 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82735 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82736 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82737 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82738 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82739 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_14 := arithEq_of_rows h (row := 1345) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_14
  simp only [← c82734, ← c82735, ← c82736] at e_1345_14
  have e_1345_15 := arithEq_of_rows h (row := 1345) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_15
  simp only [← c82737, ← c82738, k2, ← c82739] at e_1345_15
  have hr := e_1345_15
  simp only [e_1345_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12503)) := by
  have c82740 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82741 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82742 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82743 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82744 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82745 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_16 := arithEq_of_rows h (row := 1345) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_16
  simp only [← c82740, ← c82741, ← c82742] at e_1345_16
  have e_1345_17 := arithEq_of_rows h (row := 1345) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_17
  simp only [← c82743, ← c82744, k2, ← c82745] at e_1345_17
  have hr := e_1345_17
  simp only [e_1345_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1345 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12504)) := by
  have c82746 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82747 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82748 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82749 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82750 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82751 := (publicBatchWrapper89_copies2585 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1345_18 := arithEq_of_rows h (row := 1345) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_18
  simp only [← c82746, ← c82747, ← c82748] at e_1345_18
  have e_1345_19 := arithEq_of_rows h (row := 1345) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1345_19
  simp only [← c82749, ← c82750, k2, ← c82751] at e_1345_19
  have hr := e_1345_19
  simp only [e_1345_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12505)) := by
  have c82752 := (publicBatchWrapper89_copies2586 a h).1
  have c82753 := (publicBatchWrapper89_copies2586 a h).2.1
  have c82754 := (publicBatchWrapper89_copies2586 a h).2.2.1
  have c82755 := (publicBatchWrapper89_copies2586 a h).2.2.2.1
  have c82756 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.1
  have c82757 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_0 := arithEq_of_rows h (row := 1346) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_0
  simp only [← c82752, ← c82753, ← c82754] at e_1346_0
  have e_1346_1 := arithEq_of_rows h (row := 1346) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_1
  simp only [← c82755, ← c82756, k2, ← c82757] at e_1346_1
  have hr := e_1346_1
  simp only [e_1346_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12506)) := by
  have c82758 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.1
  have c82759 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.1
  have c82760 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.1
  have c82761 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.1
  have c82762 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82763 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_2 := arithEq_of_rows h (row := 1346) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_2
  simp only [← c82758, ← c82759, ← c82760] at e_1346_2
  have e_1346_3 := arithEq_of_rows h (row := 1346) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_3
  simp only [← c82761, ← c82762, k2, ← c82763] at e_1346_3
  have hr := e_1346_3
  simp only [e_1346_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12507)) := by
  have c82764 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82765 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82766 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82767 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82768 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82769 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_4 := arithEq_of_rows h (row := 1346) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_4
  simp only [← c82764, ← c82765, ← c82766] at e_1346_4
  have e_1346_5 := arithEq_of_rows h (row := 1346) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_5
  simp only [← c82767, ← c82768, k2, ← c82769] at e_1346_5
  have hr := e_1346_5
  simp only [e_1346_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12508)) := by
  have c82770 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82771 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82772 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82773 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82774 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82775 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_6 := arithEq_of_rows h (row := 1346) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_6
  simp only [← c82770, ← c82771, ← c82772] at e_1346_6
  have e_1346_7 := arithEq_of_rows h (row := 1346) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_7
  simp only [← c82773, ← c82774, k2, ← c82775] at e_1346_7
  have hr := e_1346_7
  simp only [e_1346_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12509)) := by
  have c82776 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82777 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82778 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82779 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82780 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82781 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_8 := arithEq_of_rows h (row := 1346) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_8
  simp only [← c82776, ← c82777, ← c82778] at e_1346_8
  have e_1346_9 := arithEq_of_rows h (row := 1346) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_9
  simp only [← c82779, ← c82780, k2, ← c82781] at e_1346_9
  have hr := e_1346_9
  simp only [e_1346_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12510)) := by
  have c82782 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82783 := (publicBatchWrapper89_copies2586 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82784 := (publicBatchWrapper89_copies2587 a h).1
  have c82785 := (publicBatchWrapper89_copies2587 a h).2.1
  have c82786 := (publicBatchWrapper89_copies2587 a h).2.2.1
  have c82787 := (publicBatchWrapper89_copies2587 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_10 := arithEq_of_rows h (row := 1346) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_10
  simp only [← c82782, ← c82783, ← c82784] at e_1346_10
  have e_1346_11 := arithEq_of_rows h (row := 1346) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_11
  simp only [← c82785, ← c82786, k2, ← c82787] at e_1346_11
  have hr := e_1346_11
  simp only [e_1346_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12511)) := by
  have c82788 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.1
  have c82789 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.1
  have c82790 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.1
  have c82791 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.1
  have c82792 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.1
  have c82793 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_12 := arithEq_of_rows h (row := 1346) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_12
  simp only [← c82788, ← c82789, ← c82790] at e_1346_12
  have e_1346_13 := arithEq_of_rows h (row := 1346) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_13
  simp only [← c82791, ← c82792, k2, ← c82793] at e_1346_13
  have hr := e_1346_13
  simp only [e_1346_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12512)) := by
  have c82794 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82795 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82796 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82797 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82798 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82799 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_14 := arithEq_of_rows h (row := 1346) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_14
  simp only [← c82794, ← c82795, ← c82796] at e_1346_14
  have e_1346_15 := arithEq_of_rows h (row := 1346) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_15
  simp only [← c82797, ← c82798, k2, ← c82799] at e_1346_15
  have hr := e_1346_15
  simp only [e_1346_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12513)) := by
  have c82800 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82801 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82802 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82803 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82804 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82805 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_16 := arithEq_of_rows h (row := 1346) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_16
  simp only [← c82800, ← c82801, ← c82802] at e_1346_16
  have e_1346_17 := arithEq_of_rows h (row := 1346) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_17
  simp only [← c82803, ← c82804, k2, ← c82805] at e_1346_17
  have hr := e_1346_17
  simp only [e_1346_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1346 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12514)) := by
  have c82806 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82807 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82808 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82809 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82810 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82811 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1346_18 := arithEq_of_rows h (row := 1346) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_18
  simp only [← c82806, ← c82807, ← c82808] at e_1346_18
  have e_1346_19 := arithEq_of_rows h (row := 1346) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1346_19
  simp only [← c82809, ← c82810, k2, ← c82811] at e_1346_19
  have hr := e_1346_19
  simp only [e_1346_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12515)) := by
  have c82812 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82813 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82814 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82815 := (publicBatchWrapper89_copies2587 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82816 := (publicBatchWrapper89_copies2588 a h).1
  have c82817 := (publicBatchWrapper89_copies2588 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_0 := arithEq_of_rows h (row := 1347) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_0
  simp only [← c82812, ← c82813, ← c82814] at e_1347_0
  have e_1347_1 := arithEq_of_rows h (row := 1347) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_1
  simp only [← c82815, ← c82816, k2, ← c82817] at e_1347_1
  have hr := e_1347_1
  simp only [e_1347_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12516)) := by
  have c82818 := (publicBatchWrapper89_copies2588 a h).2.2.1
  have c82819 := (publicBatchWrapper89_copies2588 a h).2.2.2.1
  have c82820 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.1
  have c82821 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.1
  have c82822 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.1
  have c82823 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_2 := arithEq_of_rows h (row := 1347) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_2
  simp only [← c82818, ← c82819, ← c82820] at e_1347_2
  have e_1347_3 := arithEq_of_rows h (row := 1347) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_3
  simp only [← c82821, ← c82822, k2, ← c82823] at e_1347_3
  have hr := e_1347_3
  simp only [e_1347_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12517)) := by
  have c82824 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.1
  have c82825 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.1
  have c82826 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82827 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82828 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82829 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_4 := arithEq_of_rows h (row := 1347) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_4
  simp only [← c82824, ← c82825, ← c82826] at e_1347_4
  have e_1347_5 := arithEq_of_rows h (row := 1347) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_5
  simp only [← c82827, ← c82828, k2, ← c82829] at e_1347_5
  have hr := e_1347_5
  simp only [e_1347_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12518)) := by
  have c82830 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82831 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82832 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82833 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82834 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82835 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_6 := arithEq_of_rows h (row := 1347) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_6
  simp only [← c82830, ← c82831, ← c82832] at e_1347_6
  have e_1347_7 := arithEq_of_rows h (row := 1347) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_7
  simp only [← c82833, ← c82834, k2, ← c82835] at e_1347_7
  have hr := e_1347_7
  simp only [e_1347_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12519)) := by
  have c82836 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82837 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82838 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82839 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82840 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82841 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_8 := arithEq_of_rows h (row := 1347) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_8
  simp only [← c82836, ← c82837, ← c82838] at e_1347_8
  have e_1347_9 := arithEq_of_rows h (row := 1347) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_9
  simp only [← c82839, ← c82840, k2, ← c82841] at e_1347_9
  have hr := e_1347_9
  simp only [e_1347_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12520)) := by
  have c82842 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82843 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82844 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82845 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82846 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82847 := (publicBatchWrapper89_copies2588 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_10 := arithEq_of_rows h (row := 1347) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_10
  simp only [← c82842, ← c82843, ← c82844] at e_1347_10
  have e_1347_11 := arithEq_of_rows h (row := 1347) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_11
  simp only [← c82845, ← c82846, k2, ← c82847] at e_1347_11
  have hr := e_1347_11
  simp only [e_1347_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12521)) := by
  have c82848 := (publicBatchWrapper89_copies2589 a h).1
  have c82849 := (publicBatchWrapper89_copies2589 a h).2.1
  have c82850 := (publicBatchWrapper89_copies2589 a h).2.2.1
  have c82851 := (publicBatchWrapper89_copies2589 a h).2.2.2.1
  have c82852 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.1
  have c82853 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_12 := arithEq_of_rows h (row := 1347) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_12
  simp only [← c82848, ← c82849, ← c82850] at e_1347_12
  have e_1347_13 := arithEq_of_rows h (row := 1347) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_13
  simp only [← c82851, ← c82852, k2, ← c82853] at e_1347_13
  have hr := e_1347_13
  simp only [e_1347_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12522)) := by
  have c82854 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.1
  have c82855 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.1
  have c82856 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.1
  have c82857 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.1
  have c82858 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82859 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_14 := arithEq_of_rows h (row := 1347) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_14
  simp only [← c82854, ← c82855, ← c82856] at e_1347_14
  have e_1347_15 := arithEq_of_rows h (row := 1347) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_15
  simp only [← c82857, ← c82858, k2, ← c82859] at e_1347_15
  have hr := e_1347_15
  simp only [e_1347_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12523)) := by
  have c82860 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82861 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82862 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82863 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82864 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82865 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_16 := arithEq_of_rows h (row := 1347) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_16
  simp only [← c82860, ← c82861, ← c82862] at e_1347_16
  have e_1347_17 := arithEq_of_rows h (row := 1347) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_17
  simp only [← c82863, ← c82864, k2, ← c82865] at e_1347_17
  have hr := e_1347_17
  simp only [e_1347_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1347 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12524)) := by
  have c82866 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82867 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82868 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82869 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82870 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82871 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1347_18 := arithEq_of_rows h (row := 1347) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_18
  simp only [← c82866, ← c82867, ← c82868] at e_1347_18
  have e_1347_19 := arithEq_of_rows h (row := 1347) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1347_19
  simp only [← c82869, ← c82870, k2, ← c82871] at e_1347_19
  have hr := e_1347_19
  simp only [e_1347_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12525)) := by
  have c82872 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82873 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82874 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82875 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82876 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82877 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_0 := arithEq_of_rows h (row := 1348) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_0
  simp only [← c82872, ← c82873, ← c82874] at e_1348_0
  have e_1348_1 := arithEq_of_rows h (row := 1348) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_1
  simp only [← c82875, ← c82876, k2, ← c82877] at e_1348_1
  have hr := e_1348_1
  simp only [e_1348_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12526)) := by
  have c82878 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82879 := (publicBatchWrapper89_copies2589 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82880 := (publicBatchWrapper89_copies2590 a h).1
  have c82881 := (publicBatchWrapper89_copies2590 a h).2.1
  have c82882 := (publicBatchWrapper89_copies2590 a h).2.2.1
  have c82883 := (publicBatchWrapper89_copies2590 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_2 := arithEq_of_rows h (row := 1348) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_2
  simp only [← c82878, ← c82879, ← c82880] at e_1348_2
  have e_1348_3 := arithEq_of_rows h (row := 1348) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_3
  simp only [← c82881, ← c82882, k2, ← c82883] at e_1348_3
  have hr := e_1348_3
  simp only [e_1348_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12527)) := by
  have c82884 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.1
  have c82885 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.1
  have c82886 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.1
  have c82887 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.1
  have c82888 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.1
  have c82889 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_4 := arithEq_of_rows h (row := 1348) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_4
  simp only [← c82884, ← c82885, ← c82886] at e_1348_4
  have e_1348_5 := arithEq_of_rows h (row := 1348) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_5
  simp only [← c82887, ← c82888, k2, ← c82889] at e_1348_5
  have hr := e_1348_5
  simp only [e_1348_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12528)) := by
  have c82890 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82891 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82892 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82893 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82894 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82895 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_6 := arithEq_of_rows h (row := 1348) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_6
  simp only [← c82890, ← c82891, ← c82892] at e_1348_6
  have e_1348_7 := arithEq_of_rows h (row := 1348) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_7
  simp only [← c82893, ← c82894, k2, ← c82895] at e_1348_7
  have hr := e_1348_7
  simp only [e_1348_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12529)) := by
  have c82896 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82897 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82898 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82899 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82900 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82901 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_8 := arithEq_of_rows h (row := 1348) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_8
  simp only [← c82896, ← c82897, ← c82898] at e_1348_8
  have e_1348_9 := arithEq_of_rows h (row := 1348) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_9
  simp only [← c82899, ← c82900, k2, ← c82901] at e_1348_9
  have hr := e_1348_9
  simp only [e_1348_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12530)) := by
  have c82902 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82903 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82904 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82905 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82906 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82907 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_10 := arithEq_of_rows h (row := 1348) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_10
  simp only [← c82902, ← c82903, ← c82904] at e_1348_10
  have e_1348_11 := arithEq_of_rows h (row := 1348) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_11
  simp only [← c82905, ← c82906, k2, ← c82907] at e_1348_11
  have hr := e_1348_11
  simp only [e_1348_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12531)) := by
  have c82908 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82909 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82910 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82911 := (publicBatchWrapper89_copies2590 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82912 := (publicBatchWrapper89_copies2591 a h).1
  have c82913 := (publicBatchWrapper89_copies2591 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_12 := arithEq_of_rows h (row := 1348) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_12
  simp only [← c82908, ← c82909, ← c82910] at e_1348_12
  have e_1348_13 := arithEq_of_rows h (row := 1348) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_13
  simp only [← c82911, ← c82912, k2, ← c82913] at e_1348_13
  have hr := e_1348_13
  simp only [e_1348_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12532)) := by
  have c82914 := (publicBatchWrapper89_copies2591 a h).2.2.1
  have c82915 := (publicBatchWrapper89_copies2591 a h).2.2.2.1
  have c82916 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.1
  have c82917 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.1
  have c82918 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.1
  have c82919 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_14 := arithEq_of_rows h (row := 1348) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_14
  simp only [← c82914, ← c82915, ← c82916] at e_1348_14
  have e_1348_15 := arithEq_of_rows h (row := 1348) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_15
  simp only [← c82917, ← c82918, k2, ← c82919] at e_1348_15
  have hr := e_1348_15
  simp only [e_1348_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12533)) := by
  have c82920 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.1
  have c82921 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.1
  have c82922 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82923 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82924 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82925 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_16 := arithEq_of_rows h (row := 1348) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_16
  simp only [← c82920, ← c82921, ← c82922] at e_1348_16
  have e_1348_17 := arithEq_of_rows h (row := 1348) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_17
  simp only [← c82923, ← c82924, k2, ← c82925] at e_1348_17
  have hr := e_1348_17
  simp only [e_1348_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1348 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12534)) := by
  have c82926 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82927 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82928 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82929 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82930 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82931 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1348_18 := arithEq_of_rows h (row := 1348) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_18
  simp only [← c82926, ← c82927, ← c82928] at e_1348_18
  have e_1348_19 := arithEq_of_rows h (row := 1348) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1348_19
  simp only [← c82929, ← c82930, k2, ← c82931] at e_1348_19
  have hr := e_1348_19
  simp only [e_1348_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12535)) := by
  have c82932 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82933 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82934 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82935 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82936 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82937 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_0 := arithEq_of_rows h (row := 1349) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_0
  simp only [← c82932, ← c82933, ← c82934] at e_1349_0
  have e_1349_1 := arithEq_of_rows h (row := 1349) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_1
  simp only [← c82935, ← c82936, k2, ← c82937] at e_1349_1
  have hr := e_1349_1
  simp only [e_1349_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12536)) := by
  have c82938 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82939 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82940 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82941 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82942 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82943 := (publicBatchWrapper89_copies2591 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_2 := arithEq_of_rows h (row := 1349) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_2
  simp only [← c82938, ← c82939, ← c82940] at e_1349_2
  have e_1349_3 := arithEq_of_rows h (row := 1349) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_3
  simp only [← c82941, ← c82942, k2, ← c82943] at e_1349_3
  have hr := e_1349_3
  simp only [e_1349_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12537)) := by
  have c82944 := (publicBatchWrapper89_copies2592 a h).1
  have c82945 := (publicBatchWrapper89_copies2592 a h).2.1
  have c82946 := (publicBatchWrapper89_copies2592 a h).2.2.1
  have c82947 := (publicBatchWrapper89_copies2592 a h).2.2.2.1
  have c82948 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.1
  have c82949 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_4 := arithEq_of_rows h (row := 1349) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_4
  simp only [← c82944, ← c82945, ← c82946] at e_1349_4
  have e_1349_5 := arithEq_of_rows h (row := 1349) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_5
  simp only [← c82947, ← c82948, k2, ← c82949] at e_1349_5
  have hr := e_1349_5
  simp only [e_1349_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12538)) := by
  have c82950 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.1
  have c82951 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.1
  have c82952 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.1
  have c82953 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.1
  have c82954 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82955 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_6 := arithEq_of_rows h (row := 1349) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_6
  simp only [← c82950, ← c82951, ← c82952] at e_1349_6
  have e_1349_7 := arithEq_of_rows h (row := 1349) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_7
  simp only [← c82953, ← c82954, k2, ← c82955] at e_1349_7
  have hr := e_1349_7
  simp only [e_1349_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12539)) := by
  have c82956 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82957 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82958 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82959 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82960 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82961 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_8 := arithEq_of_rows h (row := 1349) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_8
  simp only [← c82956, ← c82957, ← c82958] at e_1349_8
  have e_1349_9 := arithEq_of_rows h (row := 1349) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_9
  simp only [← c82959, ← c82960, k2, ← c82961] at e_1349_9
  have hr := e_1349_9
  simp only [e_1349_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12540)) := by
  have c82962 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82963 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82964 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82965 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82966 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82967 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_10 := arithEq_of_rows h (row := 1349) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_10
  simp only [← c82962, ← c82963, ← c82964] at e_1349_10
  have e_1349_11 := arithEq_of_rows h (row := 1349) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_11
  simp only [← c82965, ← c82966, k2, ← c82967] at e_1349_11
  have hr := e_1349_11
  simp only [e_1349_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12541)) := by
  have c82968 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82969 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82970 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82971 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82972 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82973 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_12 := arithEq_of_rows h (row := 1349) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_12
  simp only [← c82968, ← c82969, ← c82970] at e_1349_12
  have e_1349_13 := arithEq_of_rows h (row := 1349) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_13
  simp only [← c82971, ← c82972, k2, ← c82973] at e_1349_13
  have hr := e_1349_13
  simp only [e_1349_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12542)) := by
  have c82974 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82975 := (publicBatchWrapper89_copies2592 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c82976 := (publicBatchWrapper89_copies2593 a h).1
  have c82977 := (publicBatchWrapper89_copies2593 a h).2.1
  have c82978 := (publicBatchWrapper89_copies2593 a h).2.2.1
  have c82979 := (publicBatchWrapper89_copies2593 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_14 := arithEq_of_rows h (row := 1349) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_14
  simp only [← c82974, ← c82975, ← c82976] at e_1349_14
  have e_1349_15 := arithEq_of_rows h (row := 1349) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_15
  simp only [← c82977, ← c82978, k2, ← c82979] at e_1349_15
  have hr := e_1349_15
  simp only [e_1349_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12543)) := by
  have c82980 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.1
  have c82981 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.1
  have c82982 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.1
  have c82983 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.1
  have c82984 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.1
  have c82985 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_16 := arithEq_of_rows h (row := 1349) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_16
  simp only [← c82980, ← c82981, ← c82982] at e_1349_16
  have e_1349_17 := arithEq_of_rows h (row := 1349) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_17
  simp only [← c82983, ← c82984, k2, ← c82985] at e_1349_17
  have hr := e_1349_17
  simp only [e_1349_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1349 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12544)) := by
  have c82986 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.1
  have c82987 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c82988 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82989 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82990 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82991 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1349_18 := arithEq_of_rows h (row := 1349) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_18
  simp only [← c82986, ← c82987, ← c82988] at e_1349_18
  have e_1349_19 := arithEq_of_rows h (row := 1349) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1349_19
  simp only [← c82989, ← c82990, k2, ← c82991] at e_1349_19
  have hr := e_1349_19
  simp only [e_1349_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12545)) := by
  have c82992 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82993 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82994 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82995 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82996 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82997 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_0 := arithEq_of_rows h (row := 1350) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_0
  simp only [← c82992, ← c82993, ← c82994] at e_1350_0
  have e_1350_1 := arithEq_of_rows h (row := 1350) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_1
  simp only [← c82995, ← c82996, k2, ← c82997] at e_1350_1
  have hr := e_1350_1
  simp only [e_1350_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12546)) := by
  have c82998 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c82999 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83000 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83001 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83002 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83003 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_2 := arithEq_of_rows h (row := 1350) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_2
  simp only [← c82998, ← c82999, ← c83000] at e_1350_2
  have e_1350_3 := arithEq_of_rows h (row := 1350) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_3
  simp only [← c83001, ← c83002, k2, ← c83003] at e_1350_3
  have hr := e_1350_3
  simp only [e_1350_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12547)) := by
  have c83004 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83005 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83006 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83007 := (publicBatchWrapper89_copies2593 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c83008 := (publicBatchWrapper89_copies2594 a h).1
  have c83009 := (publicBatchWrapper89_copies2594 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_4 := arithEq_of_rows h (row := 1350) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_4
  simp only [← c83004, ← c83005, ← c83006] at e_1350_4
  have e_1350_5 := arithEq_of_rows h (row := 1350) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_5
  simp only [← c83007, ← c83008, k2, ← c83009] at e_1350_5
  have hr := e_1350_5
  simp only [e_1350_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12548)) := by
  have c83010 := (publicBatchWrapper89_copies2594 a h).2.2.1
  have c83011 := (publicBatchWrapper89_copies2594 a h).2.2.2.1
  have c83012 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.1
  have c83013 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.1
  have c83014 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.1
  have c83015 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_6 := arithEq_of_rows h (row := 1350) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_6
  simp only [← c83010, ← c83011, ← c83012] at e_1350_6
  have e_1350_7 := arithEq_of_rows h (row := 1350) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_7
  simp only [← c83013, ← c83014, k2, ← c83015] at e_1350_7
  have hr := e_1350_7
  simp only [e_1350_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12549)) := by
  have c83016 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.1
  have c83017 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.1
  have c83018 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.1
  have c83019 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c83020 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83021 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_8 := arithEq_of_rows h (row := 1350) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_8
  simp only [← c83016, ← c83017, ← c83018] at e_1350_8
  have e_1350_9 := arithEq_of_rows h (row := 1350) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_9
  simp only [← c83019, ← c83020, k2, ← c83021] at e_1350_9
  have hr := e_1350_9
  simp only [e_1350_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12550)) := by
  have c83022 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83023 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83024 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83025 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83026 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83027 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_10 := arithEq_of_rows h (row := 1350) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_10
  simp only [← c83022, ← c83023, ← c83024] at e_1350_10
  have e_1350_11 := arithEq_of_rows h (row := 1350) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_11
  simp only [← c83025, ← c83026, k2, ← c83027] at e_1350_11
  have hr := e_1350_11
  simp only [e_1350_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12551)) := by
  have c83028 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83029 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83030 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83031 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83032 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83033 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_12 := arithEq_of_rows h (row := 1350) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_12
  simp only [← c83028, ← c83029, ← c83030] at e_1350_12
  have e_1350_13 := arithEq_of_rows h (row := 1350) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_13
  simp only [← c83031, ← c83032, k2, ← c83033] at e_1350_13
  have hr := e_1350_13
  simp only [e_1350_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f13088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1350 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12552)) := by
  have c83034 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83035 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83036 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83037 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83038 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c83039 := (publicBatchWrapper89_copies2594 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1350_14 := arithEq_of_rows h (row := 1350) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_14
  simp only [← c83034, ← c83035, ← c83036] at e_1350_14
  have e_1350_15 := arithEq_of_rows h (row := 1350) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1350_15
  simp only [← c83037, ← c83038, k2, ← c83039] at e_1350_15
  have hr := e_1350_15
  simp only [e_1350_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
