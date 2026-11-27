/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PublicBatchWrapper86`, split out of the generated module
  so each Lean module elaborates in bounded memory.
  Regenerate with:

      cargo run -p qp-plonky2-constraint-exporter --bin export-constraints
-/
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.LinearCombination
import Plonky2Spec.WiringGadgets
import Plonky2Spec.Generated.PublicBatchWrapper86.CopyLemma

namespace Plonky2Spec.Generated

open Plonky2Spec.Wiring

set_option maxRecDepth 200000

set_option maxHeartbeats 0

set_option linter.all false

set_option Elab.async false

variable {p : ℕ} [Fact p.Prime]

theorem publicBatchWrapper86_f11339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1174 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10505)) := by
  have c72387 := (publicBatchWrapper86_copies2262 a h).2.2.2.1
  have c72388 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.1
  have c72389 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.1
  have c72390 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.1
  have c72391 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.1
  have c72392 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1174_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_14
  simp only [← c72387, ← c72388, ← c72389] at e_1174_14
  have e_1174_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_15
  simp only [← c72390, ← c72391, k2, ← c72392] at e_1174_15
  have hr := e_1174_15
  simp only [e_1174_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1174 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10506)) := by
  have c72393 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.1
  have c72394 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72395 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72396 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72397 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72398 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1174_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_16
  simp only [← c72393, ← c72394, ← c72395] at e_1174_16
  have e_1174_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_17
  simp only [← c72396, ← c72397, k2, ← c72398] at e_1174_17
  have hr := e_1174_17
  simp only [e_1174_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1174 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10507)) := by
  have c72399 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72400 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72401 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72402 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72403 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72404 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1174_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_18
  simp only [← c72399, ← c72400, ← c72401] at e_1174_18
  have e_1174_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1174 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1174_19
  simp only [← c72402, ← c72403, k2, ← c72404] at e_1174_19
  have hr := e_1174_19
  simp only [e_1174_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10508)) := by
  have c72405 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72406 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72407 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72408 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72409 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72410 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_0
  simp only [← c72405, ← c72406, ← c72407] at e_1175_0
  have e_1175_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_1
  simp only [← c72408, ← c72409, k2, ← c72410] at e_1175_1
  have hr := e_1175_1
  simp only [e_1175_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10509)) := by
  have c72411 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72412 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72413 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72414 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72415 := (publicBatchWrapper86_copies2262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72416 := (publicBatchWrapper86_copies2263 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_2
  simp only [← c72411, ← c72412, ← c72413] at e_1175_2
  have e_1175_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_3
  simp only [← c72414, ← c72415, k2, ← c72416] at e_1175_3
  have hr := e_1175_3
  simp only [e_1175_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10510)) := by
  have c72417 := (publicBatchWrapper86_copies2263 a h).2.1
  have c72418 := (publicBatchWrapper86_copies2263 a h).2.2.1
  have c72419 := (publicBatchWrapper86_copies2263 a h).2.2.2.1
  have c72420 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.1
  have c72421 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.1
  have c72422 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_4
  simp only [← c72417, ← c72418, ← c72419] at e_1175_4
  have e_1175_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_5
  simp only [← c72420, ← c72421, k2, ← c72422] at e_1175_5
  have hr := e_1175_5
  simp only [e_1175_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10511)) := by
  have c72423 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.1
  have c72424 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.1
  have c72425 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.1
  have c72426 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72427 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72428 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_6
  simp only [← c72423, ← c72424, ← c72425] at e_1175_6
  have e_1175_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_7
  simp only [← c72426, ← c72427, k2, ← c72428] at e_1175_7
  have hr := e_1175_7
  simp only [e_1175_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10512)) := by
  have c72429 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72430 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72431 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72432 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72433 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72434 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_8
  simp only [← c72429, ← c72430, ← c72431] at e_1175_8
  have e_1175_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_9
  simp only [← c72432, ← c72433, k2, ← c72434] at e_1175_9
  have hr := e_1175_9
  simp only [e_1175_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10513)) := by
  have c72435 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72436 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72437 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72438 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72439 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72440 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_10
  simp only [← c72435, ← c72436, ← c72437] at e_1175_10
  have e_1175_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_11
  simp only [← c72438, ← c72439, k2, ← c72440] at e_1175_11
  have hr := e_1175_11
  simp only [e_1175_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10514)) := by
  have c72441 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72442 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72443 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72444 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72445 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72446 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_12
  simp only [← c72441, ← c72442, ← c72443] at e_1175_12
  have e_1175_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_13
  simp only [← c72444, ← c72445, k2, ← c72446] at e_1175_13
  have hr := e_1175_13
  simp only [e_1175_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10515)) := by
  have c72447 := (publicBatchWrapper86_copies2263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72448 := (publicBatchWrapper86_copies2264 a h).1
  have c72449 := (publicBatchWrapper86_copies2264 a h).2.1
  have c72450 := (publicBatchWrapper86_copies2264 a h).2.2.1
  have c72451 := (publicBatchWrapper86_copies2264 a h).2.2.2.1
  have c72452 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_14
  simp only [← c72447, ← c72448, ← c72449] at e_1175_14
  have e_1175_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_15
  simp only [← c72450, ← c72451, k2, ← c72452] at e_1175_15
  have hr := e_1175_15
  simp only [e_1175_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10516)) := by
  have c72453 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.1
  have c72454 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.1
  have c72455 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.1
  have c72456 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.1
  have c72457 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.1
  have c72458 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_16
  simp only [← c72453, ← c72454, ← c72455] at e_1175_16
  have e_1175_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_17
  simp only [← c72456, ← c72457, k2, ← c72458] at e_1175_17
  have hr := e_1175_17
  simp only [e_1175_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1175 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10517)) := by
  have c72459 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72460 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72461 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72462 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72463 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72464 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1175_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_18
  simp only [← c72459, ← c72460, ← c72461] at e_1175_18
  have e_1175_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1175 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1175_19
  simp only [← c72462, ← c72463, k2, ← c72464] at e_1175_19
  have hr := e_1175_19
  simp only [e_1175_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10518)) := by
  have c72465 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72466 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72467 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72468 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72469 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72470 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_0
  simp only [← c72465, ← c72466, ← c72467] at e_1176_0
  have e_1176_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_1
  simp only [← c72468, ← c72469, k2, ← c72470] at e_1176_1
  have hr := e_1176_1
  simp only [e_1176_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10519)) := by
  have c72471 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72472 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72473 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72474 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72475 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72476 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_2
  simp only [← c72471, ← c72472, ← c72473] at e_1176_2
  have e_1176_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_3
  simp only [← c72474, ← c72475, k2, ← c72476] at e_1176_3
  have hr := e_1176_3
  simp only [e_1176_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10520)) := by
  have c72477 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72478 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72479 := (publicBatchWrapper86_copies2264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72480 := (publicBatchWrapper86_copies2265 a h).1
  have c72481 := (publicBatchWrapper86_copies2265 a h).2.1
  have c72482 := (publicBatchWrapper86_copies2265 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_4
  simp only [← c72477, ← c72478, ← c72479] at e_1176_4
  have e_1176_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_5
  simp only [← c72480, ← c72481, k2, ← c72482] at e_1176_5
  have hr := e_1176_5
  simp only [e_1176_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10521)) := by
  have c72483 := (publicBatchWrapper86_copies2265 a h).2.2.2.1
  have c72484 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.1
  have c72485 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.1
  have c72486 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.1
  have c72487 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.1
  have c72488 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_6
  simp only [← c72483, ← c72484, ← c72485] at e_1176_6
  have e_1176_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_7
  simp only [← c72486, ← c72487, k2, ← c72488] at e_1176_7
  have hr := e_1176_7
  simp only [e_1176_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10522)) := by
  have c72489 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.1
  have c72490 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72491 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72492 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72493 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72494 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_8
  simp only [← c72489, ← c72490, ← c72491] at e_1176_8
  have e_1176_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_9
  simp only [← c72492, ← c72493, k2, ← c72494] at e_1176_9
  have hr := e_1176_9
  simp only [e_1176_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10523)) := by
  have c72495 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72496 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72497 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72498 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72499 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72500 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_10
  simp only [← c72495, ← c72496, ← c72497] at e_1176_10
  have e_1176_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_11
  simp only [← c72498, ← c72499, k2, ← c72500] at e_1176_11
  have hr := e_1176_11
  simp only [e_1176_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10524)) := by
  have c72501 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72502 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72503 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72504 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72505 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72506 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_12
  simp only [← c72501, ← c72502, ← c72503] at e_1176_12
  have e_1176_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_13
  simp only [← c72504, ← c72505, k2, ← c72506] at e_1176_13
  have hr := e_1176_13
  simp only [e_1176_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10525)) := by
  have c72507 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72508 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72509 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72510 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72511 := (publicBatchWrapper86_copies2265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72512 := (publicBatchWrapper86_copies2266 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_14
  simp only [← c72507, ← c72508, ← c72509] at e_1176_14
  have e_1176_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_15
  simp only [← c72510, ← c72511, k2, ← c72512] at e_1176_15
  have hr := e_1176_15
  simp only [e_1176_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10526)) := by
  have c72513 := (publicBatchWrapper86_copies2266 a h).2.1
  have c72514 := (publicBatchWrapper86_copies2266 a h).2.2.1
  have c72515 := (publicBatchWrapper86_copies2266 a h).2.2.2.1
  have c72516 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.1
  have c72517 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.1
  have c72518 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_16
  simp only [← c72513, ← c72514, ← c72515] at e_1176_16
  have e_1176_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_17
  simp only [← c72516, ← c72517, k2, ← c72518] at e_1176_17
  have hr := e_1176_17
  simp only [e_1176_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1176 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10527)) := by
  have c72519 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.1
  have c72520 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.1
  have c72521 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.1
  have c72522 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72523 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72524 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1176_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_18
  simp only [← c72519, ← c72520, ← c72521] at e_1176_18
  have e_1176_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1176 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1176_19
  simp only [← c72522, ← c72523, k2, ← c72524] at e_1176_19
  have hr := e_1176_19
  simp only [e_1176_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10528)) := by
  have c72525 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72526 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72527 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72528 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72529 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72530 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_0
  simp only [← c72525, ← c72526, ← c72527] at e_1177_0
  have e_1177_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_1
  simp only [← c72528, ← c72529, k2, ← c72530] at e_1177_1
  have hr := e_1177_1
  simp only [e_1177_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10529)) := by
  have c72531 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72532 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72533 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72534 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72535 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72536 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_2
  simp only [← c72531, ← c72532, ← c72533] at e_1177_2
  have e_1177_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_3
  simp only [← c72534, ← c72535, k2, ← c72536] at e_1177_3
  have hr := e_1177_3
  simp only [e_1177_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10530)) := by
  have c72537 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72538 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72539 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72540 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72541 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72542 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_4
  simp only [← c72537, ← c72538, ← c72539] at e_1177_4
  have e_1177_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_5
  simp only [← c72540, ← c72541, k2, ← c72542] at e_1177_5
  have hr := e_1177_5
  simp only [e_1177_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10531)) := by
  have c72543 := (publicBatchWrapper86_copies2266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72544 := (publicBatchWrapper86_copies2267 a h).1
  have c72545 := (publicBatchWrapper86_copies2267 a h).2.1
  have c72546 := (publicBatchWrapper86_copies2267 a h).2.2.1
  have c72547 := (publicBatchWrapper86_copies2267 a h).2.2.2.1
  have c72548 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_6
  simp only [← c72543, ← c72544, ← c72545] at e_1177_6
  have e_1177_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_7
  simp only [← c72546, ← c72547, k2, ← c72548] at e_1177_7
  have hr := e_1177_7
  simp only [e_1177_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10532)) := by
  have c72549 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.1
  have c72550 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.1
  have c72551 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.1
  have c72552 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.1
  have c72553 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.1
  have c72554 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_8
  simp only [← c72549, ← c72550, ← c72551] at e_1177_8
  have e_1177_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_9
  simp only [← c72552, ← c72553, k2, ← c72554] at e_1177_9
  have hr := e_1177_9
  simp only [e_1177_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10533)) := by
  have c72555 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72556 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72557 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72558 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72559 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72560 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_10
  simp only [← c72555, ← c72556, ← c72557] at e_1177_10
  have e_1177_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_11
  simp only [← c72558, ← c72559, k2, ← c72560] at e_1177_11
  have hr := e_1177_11
  simp only [e_1177_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10534)) := by
  have c72561 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72562 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72563 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72564 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72565 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72566 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_12
  simp only [← c72561, ← c72562, ← c72563] at e_1177_12
  have e_1177_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_13
  simp only [← c72564, ← c72565, k2, ← c72566] at e_1177_13
  have hr := e_1177_13
  simp only [e_1177_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10535)) := by
  have c72567 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72568 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72569 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72570 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72571 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72572 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_14
  simp only [← c72567, ← c72568, ← c72569] at e_1177_14
  have e_1177_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_15
  simp only [← c72570, ← c72571, k2, ← c72572] at e_1177_15
  have hr := e_1177_15
  simp only [e_1177_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10536)) := by
  have c72573 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72574 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72575 := (publicBatchWrapper86_copies2267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72576 := (publicBatchWrapper86_copies2268 a h).1
  have c72577 := (publicBatchWrapper86_copies2268 a h).2.1
  have c72578 := (publicBatchWrapper86_copies2268 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_16
  simp only [← c72573, ← c72574, ← c72575] at e_1177_16
  have e_1177_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_17
  simp only [← c72576, ← c72577, k2, ← c72578] at e_1177_17
  have hr := e_1177_17
  simp only [e_1177_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1177 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10537)) := by
  have c72579 := (publicBatchWrapper86_copies2268 a h).2.2.2.1
  have c72580 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.1
  have c72581 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.1
  have c72582 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.1
  have c72583 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.1
  have c72584 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1177_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_18
  simp only [← c72579, ← c72580, ← c72581] at e_1177_18
  have e_1177_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1177 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1177_19
  simp only [← c72582, ← c72583, k2, ← c72584] at e_1177_19
  have hr := e_1177_19
  simp only [e_1177_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10538)) := by
  have c72585 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.1
  have c72586 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72587 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72588 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72589 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72590 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_0
  simp only [← c72585, ← c72586, ← c72587] at e_1178_0
  have e_1178_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_1
  simp only [← c72588, ← c72589, k2, ← c72590] at e_1178_1
  have hr := e_1178_1
  simp only [e_1178_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10539)) := by
  have c72591 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72592 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72593 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72594 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72595 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72596 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_2
  simp only [← c72591, ← c72592, ← c72593] at e_1178_2
  have e_1178_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_3
  simp only [← c72594, ← c72595, k2, ← c72596] at e_1178_3
  have hr := e_1178_3
  simp only [e_1178_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10540)) := by
  have c72597 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72598 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72599 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72600 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72601 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72602 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_4
  simp only [← c72597, ← c72598, ← c72599] at e_1178_4
  have e_1178_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_5
  simp only [← c72600, ← c72601, k2, ← c72602] at e_1178_5
  have hr := e_1178_5
  simp only [e_1178_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10541)) := by
  have c72603 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72604 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72605 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72606 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72607 := (publicBatchWrapper86_copies2268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72608 := (publicBatchWrapper86_copies2269 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_6
  simp only [← c72603, ← c72604, ← c72605] at e_1178_6
  have e_1178_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_7
  simp only [← c72606, ← c72607, k2, ← c72608] at e_1178_7
  have hr := e_1178_7
  simp only [e_1178_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10542)) := by
  have c72609 := (publicBatchWrapper86_copies2269 a h).2.1
  have c72610 := (publicBatchWrapper86_copies2269 a h).2.2.1
  have c72611 := (publicBatchWrapper86_copies2269 a h).2.2.2.1
  have c72612 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.1
  have c72613 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.1
  have c72614 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_8
  simp only [← c72609, ← c72610, ← c72611] at e_1178_8
  have e_1178_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_9
  simp only [← c72612, ← c72613, k2, ← c72614] at e_1178_9
  have hr := e_1178_9
  simp only [e_1178_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10543)) := by
  have c72615 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.1
  have c72616 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.1
  have c72617 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.1
  have c72618 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72619 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72620 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_10
  simp only [← c72615, ← c72616, ← c72617] at e_1178_10
  have e_1178_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_11
  simp only [← c72618, ← c72619, k2, ← c72620] at e_1178_11
  have hr := e_1178_11
  simp only [e_1178_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10544)) := by
  have c72621 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72622 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72623 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72624 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72625 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72626 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_12
  simp only [← c72621, ← c72622, ← c72623] at e_1178_12
  have e_1178_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_13
  simp only [← c72624, ← c72625, k2, ← c72626] at e_1178_13
  have hr := e_1178_13
  simp only [e_1178_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10545)) := by
  have c72627 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72628 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72629 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72630 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72631 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72632 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_14
  simp only [← c72627, ← c72628, ← c72629] at e_1178_14
  have e_1178_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_15
  simp only [← c72630, ← c72631, k2, ← c72632] at e_1178_15
  have hr := e_1178_15
  simp only [e_1178_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10546)) := by
  have c72633 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72634 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72635 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72636 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72637 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72638 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_16
  simp only [← c72633, ← c72634, ← c72635] at e_1178_16
  have e_1178_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_17
  simp only [← c72636, ← c72637, k2, ← c72638] at e_1178_17
  have hr := e_1178_17
  simp only [e_1178_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1178 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10547)) := by
  have c72639 := (publicBatchWrapper86_copies2269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72640 := (publicBatchWrapper86_copies2270 a h).1
  have c72641 := (publicBatchWrapper86_copies2270 a h).2.1
  have c72642 := (publicBatchWrapper86_copies2270 a h).2.2.1
  have c72643 := (publicBatchWrapper86_copies2270 a h).2.2.2.1
  have c72644 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1178_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_18
  simp only [← c72639, ← c72640, ← c72641] at e_1178_18
  have e_1178_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1178 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1178_19
  simp only [← c72642, ← c72643, k2, ← c72644] at e_1178_19
  have hr := e_1178_19
  simp only [e_1178_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10548)) := by
  have c72645 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.1
  have c72646 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.1
  have c72647 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.1
  have c72648 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.1
  have c72649 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.1
  have c72650 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_0
  simp only [← c72645, ← c72646, ← c72647] at e_1179_0
  have e_1179_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_1
  simp only [← c72648, ← c72649, k2, ← c72650] at e_1179_1
  have hr := e_1179_1
  simp only [e_1179_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10549)) := by
  have c72651 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72652 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72653 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72654 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72655 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72656 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_2
  simp only [← c72651, ← c72652, ← c72653] at e_1179_2
  have e_1179_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_3
  simp only [← c72654, ← c72655, k2, ← c72656] at e_1179_3
  have hr := e_1179_3
  simp only [e_1179_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10550)) := by
  have c72657 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72658 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72659 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72660 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72661 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72662 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_4
  simp only [← c72657, ← c72658, ← c72659] at e_1179_4
  have e_1179_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_5
  simp only [← c72660, ← c72661, k2, ← c72662] at e_1179_5
  have hr := e_1179_5
  simp only [e_1179_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10551)) := by
  have c72663 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72664 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72665 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72666 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72667 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72668 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_6
  simp only [← c72663, ← c72664, ← c72665] at e_1179_6
  have e_1179_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_7
  simp only [← c72666, ← c72667, k2, ← c72668] at e_1179_7
  have hr := e_1179_7
  simp only [e_1179_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10552)) := by
  have c72669 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72670 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72671 := (publicBatchWrapper86_copies2270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72672 := (publicBatchWrapper86_copies2271 a h).1
  have c72673 := (publicBatchWrapper86_copies2271 a h).2.1
  have c72674 := (publicBatchWrapper86_copies2271 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_8
  simp only [← c72669, ← c72670, ← c72671] at e_1179_8
  have e_1179_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_9
  simp only [← c72672, ← c72673, k2, ← c72674] at e_1179_9
  have hr := e_1179_9
  simp only [e_1179_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10553)) := by
  have c72675 := (publicBatchWrapper86_copies2271 a h).2.2.2.1
  have c72676 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.1
  have c72677 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.1
  have c72678 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.1
  have c72679 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.1
  have c72680 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_10
  simp only [← c72675, ← c72676, ← c72677] at e_1179_10
  have e_1179_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_11
  simp only [← c72678, ← c72679, k2, ← c72680] at e_1179_11
  have hr := e_1179_11
  simp only [e_1179_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10554)) := by
  have c72681 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.1
  have c72682 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72683 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72684 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72685 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72686 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_12
  simp only [← c72681, ← c72682, ← c72683] at e_1179_12
  have e_1179_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_13
  simp only [← c72684, ← c72685, k2, ← c72686] at e_1179_13
  have hr := e_1179_13
  simp only [e_1179_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10555)) := by
  have c72687 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72688 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72689 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72690 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72691 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72692 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_14
  simp only [← c72687, ← c72688, ← c72689] at e_1179_14
  have e_1179_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_15
  simp only [← c72690, ← c72691, k2, ← c72692] at e_1179_15
  have hr := e_1179_15
  simp only [e_1179_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10556)) := by
  have c72693 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72694 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72695 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72696 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72697 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72698 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_16
  simp only [← c72693, ← c72694, ← c72695] at e_1179_16
  have e_1179_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_17
  simp only [← c72696, ← c72697, k2, ← c72698] at e_1179_17
  have hr := e_1179_17
  simp only [e_1179_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1179 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10557)) := by
  have c72699 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72700 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72701 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72702 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72703 := (publicBatchWrapper86_copies2271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72704 := (publicBatchWrapper86_copies2272 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1179_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_18
  simp only [← c72699, ← c72700, ← c72701] at e_1179_18
  have e_1179_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1179 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1179_19
  simp only [← c72702, ← c72703, k2, ← c72704] at e_1179_19
  have hr := e_1179_19
  simp only [e_1179_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10558)) := by
  have c72705 := (publicBatchWrapper86_copies2272 a h).2.1
  have c72706 := (publicBatchWrapper86_copies2272 a h).2.2.1
  have c72707 := (publicBatchWrapper86_copies2272 a h).2.2.2.1
  have c72708 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.1
  have c72709 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.1
  have c72710 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_0
  simp only [← c72705, ← c72706, ← c72707] at e_1180_0
  have e_1180_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_1
  simp only [← c72708, ← c72709, k2, ← c72710] at e_1180_1
  have hr := e_1180_1
  simp only [e_1180_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10559)) := by
  have c72711 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.1
  have c72712 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.1
  have c72713 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.1
  have c72714 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72715 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72716 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_2
  simp only [← c72711, ← c72712, ← c72713] at e_1180_2
  have e_1180_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_3
  simp only [← c72714, ← c72715, k2, ← c72716] at e_1180_3
  have hr := e_1180_3
  simp only [e_1180_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 23) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10560)) := by
  have c72717 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72718 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72719 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72720 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72721 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72722 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_4
  simp only [← c72717, ← c72718, ← c72719] at e_1180_4
  have e_1180_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_5
  simp only [← c72720, ← c72721, k2, ← c72722] at e_1180_5
  have hr := e_1180_5
  simp only [e_1180_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 31) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10561)) := by
  have c72723 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72724 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72725 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72726 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72727 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72728 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_6
  simp only [← c72723, ← c72724, ← c72725] at e_1180_6
  have e_1180_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_7
  simp only [← c72726, ← c72727, k2, ← c72728] at e_1180_7
  have hr := e_1180_7
  simp only [e_1180_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 39) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10562)) := by
  have c72729 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72730 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72731 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72732 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72733 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72734 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_8
  simp only [← c72729, ← c72730, ← c72731] at e_1180_8
  have e_1180_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_9
  simp only [← c72732, ← c72733, k2, ← c72734] at e_1180_9
  have hr := e_1180_9
  simp only [e_1180_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 47) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10563)) := by
  have c72735 := (publicBatchWrapper86_copies2272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72736 := (publicBatchWrapper86_copies2273 a h).1
  have c72737 := (publicBatchWrapper86_copies2273 a h).2.1
  have c72738 := (publicBatchWrapper86_copies2273 a h).2.2.1
  have c72739 := (publicBatchWrapper86_copies2273 a h).2.2.2.1
  have c72740 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_10
  simp only [← c72735, ← c72736, ← c72737] at e_1180_10
  have e_1180_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_11
  simp only [← c72738, ← c72739, k2, ← c72740] at e_1180_11
  have hr := e_1180_11
  simp only [e_1180_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 55) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10564)) := by
  have c72741 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.1
  have c72742 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.1
  have c72743 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.1
  have c72744 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.1
  have c72745 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.1
  have c72746 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_12
  simp only [← c72741, ← c72742, ← c72743] at e_1180_12
  have e_1180_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_13
  simp only [← c72744, ← c72745, k2, ← c72746] at e_1180_13
  have hr := e_1180_13
  simp only [e_1180_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 63) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10565)) := by
  have c72747 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72748 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72749 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72750 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72751 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72752 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_14
  simp only [← c72747, ← c72748, ← c72749] at e_1180_14
  have e_1180_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_15
  simp only [← c72750, ← c72751, k2, ← c72752] at e_1180_15
  have hr := e_1180_15
  simp only [e_1180_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 71) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10566)) := by
  have c72753 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72754 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72755 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72756 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72757 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72758 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_16
  simp only [← c72753, ← c72754, ← c72755] at e_1180_16
  have e_1180_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_17
  simp only [← c72756, ← c72757, k2, ← c72758] at e_1180_17
  have hr := e_1180_17
  simp only [e_1180_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1180 79) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10567)) := by
  have c72759 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72760 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72761 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72762 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72763 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72764 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1180_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_18
  simp only [← c72759, ← c72760, ← c72761] at e_1180_18
  have e_1180_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1180 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1180_19
  simp only [← c72762, ← c72763, k2, ← c72764] at e_1180_19
  have hr := e_1180_19
  simp only [e_1180_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 7) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10568)) := by
  have c72765 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72766 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72767 := (publicBatchWrapper86_copies2273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72768 := (publicBatchWrapper86_copies2274 a h).1
  have c72769 := (publicBatchWrapper86_copies2274 a h).2.1
  have c72770 := (publicBatchWrapper86_copies2274 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_0
  simp only [← c72765, ← c72766, ← c72767] at e_1181_0
  have e_1181_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_1
  simp only [← c72768, ← c72769, k2, ← c72770] at e_1181_1
  have hr := e_1181_1
  simp only [e_1181_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 15) = bselect (a (.wire 51 75)) (a (.virt 200)) (a (.virt 10569)) := by
  have c72771 := (publicBatchWrapper86_copies2274 a h).2.2.2.1
  have c72772 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.1
  have c72773 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.1
  have c72774 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.1
  have c72775 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.1
  have c72776 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_2
  simp only [← c72771, ← c72772, ← c72773] at e_1181_2
  have e_1181_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_3
  simp only [← c72774, ← c72775, k2, ← c72776] at e_1181_3
  have hr := e_1181_3
  simp only [e_1181_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10610)) := by
  have c72777 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.1
  have c72778 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72779 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72780 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72781 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72782 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_4
  simp only [← c72777, ← c72778, ← c72779] at e_1181_4
  have e_1181_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_5
  simp only [← c72780, ← c72781, k2, ← c72782] at e_1181_5
  have hr := e_1181_5
  simp only [e_1181_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10611)) := by
  have c72783 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72784 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72785 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72786 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72787 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72788 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_6
  simp only [← c72783, ← c72784, ← c72785] at e_1181_6
  have e_1181_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_7
  simp only [← c72786, ← c72787, k2, ← c72788] at e_1181_7
  have hr := e_1181_7
  simp only [e_1181_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10612)) := by
  have c72789 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72790 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72791 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72792 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72793 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72794 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_8
  simp only [← c72789, ← c72790, ← c72791] at e_1181_8
  have e_1181_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_9
  simp only [← c72792, ← c72793, k2, ← c72794] at e_1181_9
  have hr := e_1181_9
  simp only [e_1181_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10613)) := by
  have c72795 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72796 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72797 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72798 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72799 := (publicBatchWrapper86_copies2274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72800 := (publicBatchWrapper86_copies2275 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_10
  simp only [← c72795, ← c72796, ← c72797] at e_1181_10
  have e_1181_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_11
  simp only [← c72798, ← c72799, k2, ← c72800] at e_1181_11
  have hr := e_1181_11
  simp only [e_1181_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10614)) := by
  have c72801 := (publicBatchWrapper86_copies2275 a h).2.1
  have c72802 := (publicBatchWrapper86_copies2275 a h).2.2.1
  have c72803 := (publicBatchWrapper86_copies2275 a h).2.2.2.1
  have c72804 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.1
  have c72805 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.1
  have c72806 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_12
  simp only [← c72801, ← c72802, ← c72803] at e_1181_12
  have e_1181_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_13
  simp only [← c72804, ← c72805, k2, ← c72806] at e_1181_13
  have hr := e_1181_13
  simp only [e_1181_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10615)) := by
  have c72807 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.1
  have c72808 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.1
  have c72809 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.1
  have c72810 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72811 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72812 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_14
  simp only [← c72807, ← c72808, ← c72809] at e_1181_14
  have e_1181_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_15
  simp only [← c72810, ← c72811, k2, ← c72812] at e_1181_15
  have hr := e_1181_15
  simp only [e_1181_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10616)) := by
  have c72813 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72814 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72815 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72816 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72817 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72818 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_16
  simp only [← c72813, ← c72814, ← c72815] at e_1181_16
  have e_1181_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_17
  simp only [← c72816, ← c72817, k2, ← c72818] at e_1181_17
  have hr := e_1181_17
  simp only [e_1181_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1181 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10617)) := by
  have c72819 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72820 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72821 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72822 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72823 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72824 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1181_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_18
  simp only [← c72819, ← c72820, ← c72821] at e_1181_18
  have e_1181_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1181 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1181_19
  simp only [← c72822, ← c72823, k2, ← c72824] at e_1181_19
  have hr := e_1181_19
  simp only [e_1181_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10618)) := by
  have c72825 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72826 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72827 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72828 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72829 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72830 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_0
  simp only [← c72825, ← c72826, ← c72827] at e_1182_0
  have e_1182_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_1
  simp only [← c72828, ← c72829, k2, ← c72830] at e_1182_1
  have hr := e_1182_1
  simp only [e_1182_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10619)) := by
  have c72831 := (publicBatchWrapper86_copies2275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72832 := (publicBatchWrapper86_copies2276 a h).1
  have c72833 := (publicBatchWrapper86_copies2276 a h).2.1
  have c72834 := (publicBatchWrapper86_copies2276 a h).2.2.1
  have c72835 := (publicBatchWrapper86_copies2276 a h).2.2.2.1
  have c72836 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_2
  simp only [← c72831, ← c72832, ← c72833] at e_1182_2
  have e_1182_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_3
  simp only [← c72834, ← c72835, k2, ← c72836] at e_1182_3
  have hr := e_1182_3
  simp only [e_1182_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10620)) := by
  have c72837 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.1
  have c72838 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.1
  have c72839 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.1
  have c72840 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.1
  have c72841 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.1
  have c72842 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_4
  simp only [← c72837, ← c72838, ← c72839] at e_1182_4
  have e_1182_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_5
  simp only [← c72840, ← c72841, k2, ← c72842] at e_1182_5
  have hr := e_1182_5
  simp only [e_1182_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10621)) := by
  have c72843 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72844 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72845 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72846 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72847 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72848 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_6
  simp only [← c72843, ← c72844, ← c72845] at e_1182_6
  have e_1182_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_7
  simp only [← c72846, ← c72847, k2, ← c72848] at e_1182_7
  have hr := e_1182_7
  simp only [e_1182_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10622)) := by
  have c72849 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72850 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72851 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72852 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72853 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72854 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_8
  simp only [← c72849, ← c72850, ← c72851] at e_1182_8
  have e_1182_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_9
  simp only [← c72852, ← c72853, k2, ← c72854] at e_1182_9
  have hr := e_1182_9
  simp only [e_1182_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10623)) := by
  have c72855 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72856 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72857 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72858 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72859 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72860 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_10
  simp only [← c72855, ← c72856, ← c72857] at e_1182_10
  have e_1182_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_11
  simp only [← c72858, ← c72859, k2, ← c72860] at e_1182_11
  have hr := e_1182_11
  simp only [e_1182_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10624)) := by
  have c72861 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72862 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72863 := (publicBatchWrapper86_copies2276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72864 := (publicBatchWrapper86_copies2277 a h).1
  have c72865 := (publicBatchWrapper86_copies2277 a h).2.1
  have c72866 := (publicBatchWrapper86_copies2277 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_12
  simp only [← c72861, ← c72862, ← c72863] at e_1182_12
  have e_1182_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_13
  simp only [← c72864, ← c72865, k2, ← c72866] at e_1182_13
  have hr := e_1182_13
  simp only [e_1182_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10625)) := by
  have c72867 := (publicBatchWrapper86_copies2277 a h).2.2.2.1
  have c72868 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.1
  have c72869 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.1
  have c72870 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.1
  have c72871 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.1
  have c72872 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_14
  simp only [← c72867, ← c72868, ← c72869] at e_1182_14
  have e_1182_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_15
  simp only [← c72870, ← c72871, k2, ← c72872] at e_1182_15
  have hr := e_1182_15
  simp only [e_1182_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10626)) := by
  have c72873 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.1
  have c72874 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72875 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72876 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72877 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72878 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_16
  simp only [← c72873, ← c72874, ← c72875] at e_1182_16
  have e_1182_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_17
  simp only [← c72876, ← c72877, k2, ← c72878] at e_1182_17
  have hr := e_1182_17
  simp only [e_1182_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1182 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10627)) := by
  have c72879 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72880 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72881 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72882 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72883 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72884 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1182_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_18
  simp only [← c72879, ← c72880, ← c72881] at e_1182_18
  have e_1182_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1182 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1182_19
  simp only [← c72882, ← c72883, k2, ← c72884] at e_1182_19
  have hr := e_1182_19
  simp only [e_1182_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10628)) := by
  have c72885 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72886 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72887 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72888 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72889 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72890 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_0
  simp only [← c72885, ← c72886, ← c72887] at e_1183_0
  have e_1183_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_1
  simp only [← c72888, ← c72889, k2, ← c72890] at e_1183_1
  have hr := e_1183_1
  simp only [e_1183_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10629)) := by
  have c72891 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72892 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72893 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72894 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72895 := (publicBatchWrapper86_copies2277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72896 := (publicBatchWrapper86_copies2278 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_2
  simp only [← c72891, ← c72892, ← c72893] at e_1183_2
  have e_1183_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_3
  simp only [← c72894, ← c72895, k2, ← c72896] at e_1183_3
  have hr := e_1183_3
  simp only [e_1183_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10630)) := by
  have c72897 := (publicBatchWrapper86_copies2278 a h).2.1
  have c72898 := (publicBatchWrapper86_copies2278 a h).2.2.1
  have c72899 := (publicBatchWrapper86_copies2278 a h).2.2.2.1
  have c72900 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.1
  have c72901 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.1
  have c72902 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_4
  simp only [← c72897, ← c72898, ← c72899] at e_1183_4
  have e_1183_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_5
  simp only [← c72900, ← c72901, k2, ← c72902] at e_1183_5
  have hr := e_1183_5
  simp only [e_1183_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10631)) := by
  have c72903 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.1
  have c72904 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.1
  have c72905 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.1
  have c72906 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72907 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72908 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_6
  simp only [← c72903, ← c72904, ← c72905] at e_1183_6
  have e_1183_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_7
  simp only [← c72906, ← c72907, k2, ← c72908] at e_1183_7
  have hr := e_1183_7
  simp only [e_1183_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10632)) := by
  have c72909 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72910 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72911 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72912 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72913 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72914 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_8
  simp only [← c72909, ← c72910, ← c72911] at e_1183_8
  have e_1183_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_9
  simp only [← c72912, ← c72913, k2, ← c72914] at e_1183_9
  have hr := e_1183_9
  simp only [e_1183_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10633)) := by
  have c72915 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72916 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72917 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72918 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72919 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72920 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_10
  simp only [← c72915, ← c72916, ← c72917] at e_1183_10
  have e_1183_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_11
  simp only [← c72918, ← c72919, k2, ← c72920] at e_1183_11
  have hr := e_1183_11
  simp only [e_1183_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10634)) := by
  have c72921 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72922 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72923 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72924 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72925 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72926 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_12
  simp only [← c72921, ← c72922, ← c72923] at e_1183_12
  have e_1183_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_13
  simp only [← c72924, ← c72925, k2, ← c72926] at e_1183_13
  have hr := e_1183_13
  simp only [e_1183_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10635)) := by
  have c72927 := (publicBatchWrapper86_copies2278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72928 := (publicBatchWrapper86_copies2279 a h).1
  have c72929 := (publicBatchWrapper86_copies2279 a h).2.1
  have c72930 := (publicBatchWrapper86_copies2279 a h).2.2.1
  have c72931 := (publicBatchWrapper86_copies2279 a h).2.2.2.1
  have c72932 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_14
  simp only [← c72927, ← c72928, ← c72929] at e_1183_14
  have e_1183_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_15
  simp only [← c72930, ← c72931, k2, ← c72932] at e_1183_15
  have hr := e_1183_15
  simp only [e_1183_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10636)) := by
  have c72933 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.1
  have c72934 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.1
  have c72935 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.1
  have c72936 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.1
  have c72937 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.1
  have c72938 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_16
  simp only [← c72933, ← c72934, ← c72935] at e_1183_16
  have e_1183_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_17
  simp only [← c72936, ← c72937, k2, ← c72938] at e_1183_17
  have hr := e_1183_17
  simp only [e_1183_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1183 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10637)) := by
  have c72939 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72940 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72941 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72942 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72943 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72944 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1183_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_18
  simp only [← c72939, ← c72940, ← c72941] at e_1183_18
  have e_1183_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1183 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1183_19
  simp only [← c72942, ← c72943, k2, ← c72944] at e_1183_19
  have hr := e_1183_19
  simp only [e_1183_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10638)) := by
  have c72945 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72946 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72947 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72948 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72949 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72950 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_0
  simp only [← c72945, ← c72946, ← c72947] at e_1184_0
  have e_1184_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_1
  simp only [← c72948, ← c72949, k2, ← c72950] at e_1184_1
  have hr := e_1184_1
  simp only [e_1184_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10639)) := by
  have c72951 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72952 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72953 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72954 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72955 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72956 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_2
  simp only [← c72951, ← c72952, ← c72953] at e_1184_2
  have e_1184_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_3
  simp only [← c72954, ← c72955, k2, ← c72956] at e_1184_3
  have hr := e_1184_3
  simp only [e_1184_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10640)) := by
  have c72957 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72958 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72959 := (publicBatchWrapper86_copies2279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72960 := (publicBatchWrapper86_copies2280 a h).1
  have c72961 := (publicBatchWrapper86_copies2280 a h).2.1
  have c72962 := (publicBatchWrapper86_copies2280 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_4
  simp only [← c72957, ← c72958, ← c72959] at e_1184_4
  have e_1184_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_5
  simp only [← c72960, ← c72961, k2, ← c72962] at e_1184_5
  have hr := e_1184_5
  simp only [e_1184_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10641)) := by
  have c72963 := (publicBatchWrapper86_copies2280 a h).2.2.2.1
  have c72964 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.1
  have c72965 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.1
  have c72966 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.1
  have c72967 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.1
  have c72968 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_6
  simp only [← c72963, ← c72964, ← c72965] at e_1184_6
  have e_1184_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_7
  simp only [← c72966, ← c72967, k2, ← c72968] at e_1184_7
  have hr := e_1184_7
  simp only [e_1184_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10642)) := by
  have c72969 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.1
  have c72970 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c72971 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c72972 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72973 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72974 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_8
  simp only [← c72969, ← c72970, ← c72971] at e_1184_8
  have e_1184_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_9
  simp only [← c72972, ← c72973, k2, ← c72974] at e_1184_9
  have hr := e_1184_9
  simp only [e_1184_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10643)) := by
  have c72975 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72976 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72977 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72978 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72979 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72980 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_10
  simp only [← c72975, ← c72976, ← c72977] at e_1184_10
  have e_1184_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_11
  simp only [← c72978, ← c72979, k2, ← c72980] at e_1184_11
  have hr := e_1184_11
  simp only [e_1184_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10644)) := by
  have c72981 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72982 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72983 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72984 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72985 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72986 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_12
  simp only [← c72981, ← c72982, ← c72983] at e_1184_12
  have e_1184_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_13
  simp only [← c72984, ← c72985, k2, ← c72986] at e_1184_13
  have hr := e_1184_13
  simp only [e_1184_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10645)) := by
  have c72987 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72988 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72989 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72990 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c72991 := (publicBatchWrapper86_copies2280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c72992 := (publicBatchWrapper86_copies2281 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_14
  simp only [← c72987, ← c72988, ← c72989] at e_1184_14
  have e_1184_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_15
  simp only [← c72990, ← c72991, k2, ← c72992] at e_1184_15
  have hr := e_1184_15
  simp only [e_1184_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10646)) := by
  have c72993 := (publicBatchWrapper86_copies2281 a h).2.1
  have c72994 := (publicBatchWrapper86_copies2281 a h).2.2.1
  have c72995 := (publicBatchWrapper86_copies2281 a h).2.2.2.1
  have c72996 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.1
  have c72997 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.1
  have c72998 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_16
  simp only [← c72993, ← c72994, ← c72995] at e_1184_16
  have e_1184_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_17
  simp only [← c72996, ← c72997, k2, ← c72998] at e_1184_17
  have hr := e_1184_17
  simp only [e_1184_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1184 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10647)) := by
  have c72999 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.1
  have c73000 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.1
  have c73001 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.1
  have c73002 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73003 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73004 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1184_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_18
  simp only [← c72999, ← c73000, ← c73001] at e_1184_18
  have e_1184_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1184 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1184_19
  simp only [← c73002, ← c73003, k2, ← c73004] at e_1184_19
  have hr := e_1184_19
  simp only [e_1184_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10648)) := by
  have c73005 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73006 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73007 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73008 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73009 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73010 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_0
  simp only [← c73005, ← c73006, ← c73007] at e_1185_0
  have e_1185_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_1
  simp only [← c73008, ← c73009, k2, ← c73010] at e_1185_1
  have hr := e_1185_1
  simp only [e_1185_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10649)) := by
  have c73011 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73012 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73013 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73014 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73015 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73016 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_2
  simp only [← c73011, ← c73012, ← c73013] at e_1185_2
  have e_1185_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_3
  simp only [← c73014, ← c73015, k2, ← c73016] at e_1185_3
  have hr := e_1185_3
  simp only [e_1185_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10650)) := by
  have c73017 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73018 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73019 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73020 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73021 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73022 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_4
  simp only [← c73017, ← c73018, ← c73019] at e_1185_4
  have e_1185_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_5
  simp only [← c73020, ← c73021, k2, ← c73022] at e_1185_5
  have hr := e_1185_5
  simp only [e_1185_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10651)) := by
  have c73023 := (publicBatchWrapper86_copies2281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73024 := (publicBatchWrapper86_copies2282 a h).1
  have c73025 := (publicBatchWrapper86_copies2282 a h).2.1
  have c73026 := (publicBatchWrapper86_copies2282 a h).2.2.1
  have c73027 := (publicBatchWrapper86_copies2282 a h).2.2.2.1
  have c73028 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_6
  simp only [← c73023, ← c73024, ← c73025] at e_1185_6
  have e_1185_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_7
  simp only [← c73026, ← c73027, k2, ← c73028] at e_1185_7
  have hr := e_1185_7
  simp only [e_1185_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10652)) := by
  have c73029 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.1
  have c73030 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.1
  have c73031 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.1
  have c73032 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.1
  have c73033 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.1
  have c73034 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_8
  simp only [← c73029, ← c73030, ← c73031] at e_1185_8
  have e_1185_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_9
  simp only [← c73032, ← c73033, k2, ← c73034] at e_1185_9
  have hr := e_1185_9
  simp only [e_1185_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10653)) := by
  have c73035 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73036 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73037 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73038 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73039 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73040 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_10
  simp only [← c73035, ← c73036, ← c73037] at e_1185_10
  have e_1185_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_11
  simp only [← c73038, ← c73039, k2, ← c73040] at e_1185_11
  have hr := e_1185_11
  simp only [e_1185_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10654)) := by
  have c73041 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73042 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73043 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73044 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73045 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73046 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_12
  simp only [← c73041, ← c73042, ← c73043] at e_1185_12
  have e_1185_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_13
  simp only [← c73044, ← c73045, k2, ← c73046] at e_1185_13
  have hr := e_1185_13
  simp only [e_1185_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10655)) := by
  have c73047 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73048 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73049 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73050 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73051 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73052 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_14
  simp only [← c73047, ← c73048, ← c73049] at e_1185_14
  have e_1185_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_15
  simp only [← c73050, ← c73051, k2, ← c73052] at e_1185_15
  have hr := e_1185_15
  simp only [e_1185_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10656)) := by
  have c73053 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73054 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73055 := (publicBatchWrapper86_copies2282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73056 := (publicBatchWrapper86_copies2283 a h).1
  have c73057 := (publicBatchWrapper86_copies2283 a h).2.1
  have c73058 := (publicBatchWrapper86_copies2283 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_16
  simp only [← c73053, ← c73054, ← c73055] at e_1185_16
  have e_1185_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_17
  simp only [← c73056, ← c73057, k2, ← c73058] at e_1185_17
  have hr := e_1185_17
  simp only [e_1185_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1185 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10657)) := by
  have c73059 := (publicBatchWrapper86_copies2283 a h).2.2.2.1
  have c73060 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.1
  have c73061 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.1
  have c73062 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.1
  have c73063 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.1
  have c73064 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1185_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_18
  simp only [← c73059, ← c73060, ← c73061] at e_1185_18
  have e_1185_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1185 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1185_19
  simp only [← c73062, ← c73063, k2, ← c73064] at e_1185_19
  have hr := e_1185_19
  simp only [e_1185_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10658)) := by
  have c73065 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.1
  have c73066 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73067 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73068 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73069 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73070 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_0
  simp only [← c73065, ← c73066, ← c73067] at e_1186_0
  have e_1186_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_1
  simp only [← c73068, ← c73069, k2, ← c73070] at e_1186_1
  have hr := e_1186_1
  simp only [e_1186_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10659)) := by
  have c73071 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73072 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73073 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73074 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73075 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73076 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_2
  simp only [← c73071, ← c73072, ← c73073] at e_1186_2
  have e_1186_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_3
  simp only [← c73074, ← c73075, k2, ← c73076] at e_1186_3
  have hr := e_1186_3
  simp only [e_1186_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10660)) := by
  have c73077 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73078 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73079 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73080 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73081 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73082 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_4
  simp only [← c73077, ← c73078, ← c73079] at e_1186_4
  have e_1186_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_5
  simp only [← c73080, ← c73081, k2, ← c73082] at e_1186_5
  have hr := e_1186_5
  simp only [e_1186_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10661)) := by
  have c73083 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73084 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73085 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73086 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73087 := (publicBatchWrapper86_copies2283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73088 := (publicBatchWrapper86_copies2284 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_6
  simp only [← c73083, ← c73084, ← c73085] at e_1186_6
  have e_1186_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_7
  simp only [← c73086, ← c73087, k2, ← c73088] at e_1186_7
  have hr := e_1186_7
  simp only [e_1186_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10662)) := by
  have c73089 := (publicBatchWrapper86_copies2284 a h).2.1
  have c73090 := (publicBatchWrapper86_copies2284 a h).2.2.1
  have c73091 := (publicBatchWrapper86_copies2284 a h).2.2.2.1
  have c73092 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.1
  have c73093 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.1
  have c73094 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_8
  simp only [← c73089, ← c73090, ← c73091] at e_1186_8
  have e_1186_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_9
  simp only [← c73092, ← c73093, k2, ← c73094] at e_1186_9
  have hr := e_1186_9
  simp only [e_1186_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10663)) := by
  have c73095 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.1
  have c73096 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.1
  have c73097 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.1
  have c73098 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73099 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73100 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_10
  simp only [← c73095, ← c73096, ← c73097] at e_1186_10
  have e_1186_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_11
  simp only [← c73098, ← c73099, k2, ← c73100] at e_1186_11
  have hr := e_1186_11
  simp only [e_1186_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10664)) := by
  have c73101 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73102 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73103 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73104 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73105 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73106 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_12
  simp only [← c73101, ← c73102, ← c73103] at e_1186_12
  have e_1186_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_13
  simp only [← c73104, ← c73105, k2, ← c73106] at e_1186_13
  have hr := e_1186_13
  simp only [e_1186_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10665)) := by
  have c73107 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73108 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73109 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73110 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73111 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73112 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_14
  simp only [← c73107, ← c73108, ← c73109] at e_1186_14
  have e_1186_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_15
  simp only [← c73110, ← c73111, k2, ← c73112] at e_1186_15
  have hr := e_1186_15
  simp only [e_1186_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10666)) := by
  have c73113 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73114 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73115 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73116 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73117 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73118 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_16
  simp only [← c73113, ← c73114, ← c73115] at e_1186_16
  have e_1186_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_17
  simp only [← c73116, ← c73117, k2, ← c73118] at e_1186_17
  have hr := e_1186_17
  simp only [e_1186_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1186 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10667)) := by
  have c73119 := (publicBatchWrapper86_copies2284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73120 := (publicBatchWrapper86_copies2285 a h).1
  have c73121 := (publicBatchWrapper86_copies2285 a h).2.1
  have c73122 := (publicBatchWrapper86_copies2285 a h).2.2.1
  have c73123 := (publicBatchWrapper86_copies2285 a h).2.2.2.1
  have c73124 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1186_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_18
  simp only [← c73119, ← c73120, ← c73121] at e_1186_18
  have e_1186_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1186 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1186_19
  simp only [← c73122, ← c73123, k2, ← c73124] at e_1186_19
  have hr := e_1186_19
  simp only [e_1186_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10668)) := by
  have c73125 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.1
  have c73126 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.1
  have c73127 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.1
  have c73128 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.1
  have c73129 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.1
  have c73130 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_0
  simp only [← c73125, ← c73126, ← c73127] at e_1187_0
  have e_1187_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_1
  simp only [← c73128, ← c73129, k2, ← c73130] at e_1187_1
  have hr := e_1187_1
  simp only [e_1187_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10669)) := by
  have c73131 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73132 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73133 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73134 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73135 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73136 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_2
  simp only [← c73131, ← c73132, ← c73133] at e_1187_2
  have e_1187_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_3
  simp only [← c73134, ← c73135, k2, ← c73136] at e_1187_3
  have hr := e_1187_3
  simp only [e_1187_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10670)) := by
  have c73137 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73138 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73139 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73140 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73141 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73142 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_4
  simp only [← c73137, ← c73138, ← c73139] at e_1187_4
  have e_1187_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_5
  simp only [← c73140, ← c73141, k2, ← c73142] at e_1187_5
  have hr := e_1187_5
  simp only [e_1187_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10671)) := by
  have c73143 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73144 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73145 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73146 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73147 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73148 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_6
  simp only [← c73143, ← c73144, ← c73145] at e_1187_6
  have e_1187_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_7
  simp only [← c73146, ← c73147, k2, ← c73148] at e_1187_7
  have hr := e_1187_7
  simp only [e_1187_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10672)) := by
  have c73149 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73150 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73151 := (publicBatchWrapper86_copies2285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73152 := (publicBatchWrapper86_copies2286 a h).1
  have c73153 := (publicBatchWrapper86_copies2286 a h).2.1
  have c73154 := (publicBatchWrapper86_copies2286 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_8
  simp only [← c73149, ← c73150, ← c73151] at e_1187_8
  have e_1187_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_9
  simp only [← c73152, ← c73153, k2, ← c73154] at e_1187_9
  have hr := e_1187_9
  simp only [e_1187_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10673)) := by
  have c73155 := (publicBatchWrapper86_copies2286 a h).2.2.2.1
  have c73156 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.1
  have c73157 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.1
  have c73158 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.1
  have c73159 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.1
  have c73160 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_10
  simp only [← c73155, ← c73156, ← c73157] at e_1187_10
  have e_1187_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_11
  simp only [← c73158, ← c73159, k2, ← c73160] at e_1187_11
  have hr := e_1187_11
  simp only [e_1187_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10674)) := by
  have c73161 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.1
  have c73162 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73163 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73164 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73165 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73166 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_12
  simp only [← c73161, ← c73162, ← c73163] at e_1187_12
  have e_1187_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_13
  simp only [← c73164, ← c73165, k2, ← c73166] at e_1187_13
  have hr := e_1187_13
  simp only [e_1187_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10675)) := by
  have c73167 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73168 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73169 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73170 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73171 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73172 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_14
  simp only [← c73167, ← c73168, ← c73169] at e_1187_14
  have e_1187_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_15
  simp only [← c73170, ← c73171, k2, ← c73172] at e_1187_15
  have hr := e_1187_15
  simp only [e_1187_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10676)) := by
  have c73173 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73174 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73175 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73176 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73177 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73178 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_16
  simp only [← c73173, ← c73174, ← c73175] at e_1187_16
  have e_1187_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_17
  simp only [← c73176, ← c73177, k2, ← c73178] at e_1187_17
  have hr := e_1187_17
  simp only [e_1187_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1187 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10677)) := by
  have c73179 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73180 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73181 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73182 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73183 := (publicBatchWrapper86_copies2286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73184 := (publicBatchWrapper86_copies2287 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1187_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_18
  simp only [← c73179, ← c73180, ← c73181] at e_1187_18
  have e_1187_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1187 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1187_19
  simp only [← c73182, ← c73183, k2, ← c73184] at e_1187_19
  have hr := e_1187_19
  simp only [e_1187_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10678)) := by
  have c73185 := (publicBatchWrapper86_copies2287 a h).2.1
  have c73186 := (publicBatchWrapper86_copies2287 a h).2.2.1
  have c73187 := (publicBatchWrapper86_copies2287 a h).2.2.2.1
  have c73188 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.1
  have c73189 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.1
  have c73190 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_0
  simp only [← c73185, ← c73186, ← c73187] at e_1188_0
  have e_1188_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_1
  simp only [← c73188, ← c73189, k2, ← c73190] at e_1188_1
  have hr := e_1188_1
  simp only [e_1188_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10679)) := by
  have c73191 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.1
  have c73192 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.1
  have c73193 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.1
  have c73194 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73195 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73196 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_2
  simp only [← c73191, ← c73192, ← c73193] at e_1188_2
  have e_1188_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_3
  simp only [← c73194, ← c73195, k2, ← c73196] at e_1188_3
  have hr := e_1188_3
  simp only [e_1188_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10680)) := by
  have c73197 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73198 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73199 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73200 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73201 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73202 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_4
  simp only [← c73197, ← c73198, ← c73199] at e_1188_4
  have e_1188_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_5
  simp only [← c73200, ← c73201, k2, ← c73202] at e_1188_5
  have hr := e_1188_5
  simp only [e_1188_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10681)) := by
  have c73203 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73204 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73205 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73206 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73207 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73208 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_6
  simp only [← c73203, ← c73204, ← c73205] at e_1188_6
  have e_1188_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_7
  simp only [← c73206, ← c73207, k2, ← c73208] at e_1188_7
  have hr := e_1188_7
  simp only [e_1188_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10682)) := by
  have c73209 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73210 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73211 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73212 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73213 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73214 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_8
  simp only [← c73209, ← c73210, ← c73211] at e_1188_8
  have e_1188_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_9
  simp only [← c73212, ← c73213, k2, ← c73214] at e_1188_9
  have hr := e_1188_9
  simp only [e_1188_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10683)) := by
  have c73215 := (publicBatchWrapper86_copies2287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73216 := (publicBatchWrapper86_copies2288 a h).1
  have c73217 := (publicBatchWrapper86_copies2288 a h).2.1
  have c73218 := (publicBatchWrapper86_copies2288 a h).2.2.1
  have c73219 := (publicBatchWrapper86_copies2288 a h).2.2.2.1
  have c73220 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_10
  simp only [← c73215, ← c73216, ← c73217] at e_1188_10
  have e_1188_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_11
  simp only [← c73218, ← c73219, k2, ← c73220] at e_1188_11
  have hr := e_1188_11
  simp only [e_1188_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10684)) := by
  have c73221 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.1
  have c73222 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.1
  have c73223 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.1
  have c73224 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.1
  have c73225 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.1
  have c73226 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_12
  simp only [← c73221, ← c73222, ← c73223] at e_1188_12
  have e_1188_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_13
  simp only [← c73224, ← c73225, k2, ← c73226] at e_1188_13
  have hr := e_1188_13
  simp only [e_1188_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10685)) := by
  have c73227 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73228 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73229 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73230 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73231 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73232 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_14
  simp only [← c73227, ← c73228, ← c73229] at e_1188_14
  have e_1188_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_15
  simp only [← c73230, ← c73231, k2, ← c73232] at e_1188_15
  have hr := e_1188_15
  simp only [e_1188_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10686)) := by
  have c73233 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73234 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73235 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73236 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73237 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73238 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_16
  simp only [← c73233, ← c73234, ← c73235] at e_1188_16
  have e_1188_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_17
  simp only [← c73236, ← c73237, k2, ← c73238] at e_1188_17
  have hr := e_1188_17
  simp only [e_1188_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1188 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10687)) := by
  have c73239 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73240 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73241 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73242 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73243 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73244 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1188_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_18
  simp only [← c73239, ← c73240, ← c73241] at e_1188_18
  have e_1188_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1188 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1188_19
  simp only [← c73242, ← c73243, k2, ← c73244] at e_1188_19
  have hr := e_1188_19
  simp only [e_1188_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10688)) := by
  have c73245 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73246 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73247 := (publicBatchWrapper86_copies2288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73248 := (publicBatchWrapper86_copies2289 a h).1
  have c73249 := (publicBatchWrapper86_copies2289 a h).2.1
  have c73250 := (publicBatchWrapper86_copies2289 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_0
  simp only [← c73245, ← c73246, ← c73247] at e_1189_0
  have e_1189_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_1
  simp only [← c73248, ← c73249, k2, ← c73250] at e_1189_1
  have hr := e_1189_1
  simp only [e_1189_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10689)) := by
  have c73251 := (publicBatchWrapper86_copies2289 a h).2.2.2.1
  have c73252 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.1
  have c73253 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.1
  have c73254 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.1
  have c73255 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.1
  have c73256 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_2
  simp only [← c73251, ← c73252, ← c73253] at e_1189_2
  have e_1189_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_3
  simp only [← c73254, ← c73255, k2, ← c73256] at e_1189_3
  have hr := e_1189_3
  simp only [e_1189_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10690)) := by
  have c73257 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.1
  have c73258 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73259 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73260 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73261 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73262 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_4
  simp only [← c73257, ← c73258, ← c73259] at e_1189_4
  have e_1189_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_5
  simp only [← c73260, ← c73261, k2, ← c73262] at e_1189_5
  have hr := e_1189_5
  simp only [e_1189_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10691)) := by
  have c73263 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73264 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73265 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73266 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73267 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73268 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_6
  simp only [← c73263, ← c73264, ← c73265] at e_1189_6
  have e_1189_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_7
  simp only [← c73266, ← c73267, k2, ← c73268] at e_1189_7
  have hr := e_1189_7
  simp only [e_1189_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10692)) := by
  have c73269 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73270 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73271 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73272 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73273 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73274 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_8
  simp only [← c73269, ← c73270, ← c73271] at e_1189_8
  have e_1189_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_9
  simp only [← c73272, ← c73273, k2, ← c73274] at e_1189_9
  have hr := e_1189_9
  simp only [e_1189_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10693)) := by
  have c73275 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73276 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73277 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73278 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73279 := (publicBatchWrapper86_copies2289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73280 := (publicBatchWrapper86_copies2290 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_10
  simp only [← c73275, ← c73276, ← c73277] at e_1189_10
  have e_1189_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_11
  simp only [← c73278, ← c73279, k2, ← c73280] at e_1189_11
  have hr := e_1189_11
  simp only [e_1189_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10694)) := by
  have c73281 := (publicBatchWrapper86_copies2290 a h).2.1
  have c73282 := (publicBatchWrapper86_copies2290 a h).2.2.1
  have c73283 := (publicBatchWrapper86_copies2290 a h).2.2.2.1
  have c73284 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.1
  have c73285 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.1
  have c73286 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_12
  simp only [← c73281, ← c73282, ← c73283] at e_1189_12
  have e_1189_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_13
  simp only [← c73284, ← c73285, k2, ← c73286] at e_1189_13
  have hr := e_1189_13
  simp only [e_1189_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10695)) := by
  have c73287 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.1
  have c73288 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.1
  have c73289 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.1
  have c73290 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73291 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73292 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_14
  simp only [← c73287, ← c73288, ← c73289] at e_1189_14
  have e_1189_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_15
  simp only [← c73290, ← c73291, k2, ← c73292] at e_1189_15
  have hr := e_1189_15
  simp only [e_1189_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10696)) := by
  have c73293 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73294 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73295 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73296 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73297 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73298 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_16
  simp only [← c73293, ← c73294, ← c73295] at e_1189_16
  have e_1189_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_17
  simp only [← c73296, ← c73297, k2, ← c73298] at e_1189_17
  have hr := e_1189_17
  simp only [e_1189_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1189 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10697)) := by
  have c73299 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73300 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73301 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73302 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73303 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73304 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1189_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_18
  simp only [← c73299, ← c73300, ← c73301] at e_1189_18
  have e_1189_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1189 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1189_19
  simp only [← c73302, ← c73303, k2, ← c73304] at e_1189_19
  have hr := e_1189_19
  simp only [e_1189_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10698)) := by
  have c73305 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73306 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73307 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73308 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73309 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73310 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_0
  simp only [← c73305, ← c73306, ← c73307] at e_1190_0
  have e_1190_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_1
  simp only [← c73308, ← c73309, k2, ← c73310] at e_1190_1
  have hr := e_1190_1
  simp only [e_1190_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10699)) := by
  have c73311 := (publicBatchWrapper86_copies2290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73312 := (publicBatchWrapper86_copies2291 a h).1
  have c73313 := (publicBatchWrapper86_copies2291 a h).2.1
  have c73314 := (publicBatchWrapper86_copies2291 a h).2.2.1
  have c73315 := (publicBatchWrapper86_copies2291 a h).2.2.2.1
  have c73316 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_2
  simp only [← c73311, ← c73312, ← c73313] at e_1190_2
  have e_1190_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_3
  simp only [← c73314, ← c73315, k2, ← c73316] at e_1190_3
  have hr := e_1190_3
  simp only [e_1190_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10700)) := by
  have c73317 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.1
  have c73318 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.1
  have c73319 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.1
  have c73320 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.1
  have c73321 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.1
  have c73322 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_4
  simp only [← c73317, ← c73318, ← c73319] at e_1190_4
  have e_1190_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_5
  simp only [← c73320, ← c73321, k2, ← c73322] at e_1190_5
  have hr := e_1190_5
  simp only [e_1190_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10701)) := by
  have c73323 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73324 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73325 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73326 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73327 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73328 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_6
  simp only [← c73323, ← c73324, ← c73325] at e_1190_6
  have e_1190_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_7
  simp only [← c73326, ← c73327, k2, ← c73328] at e_1190_7
  have hr := e_1190_7
  simp only [e_1190_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10702)) := by
  have c73329 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73330 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73331 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73332 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73333 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73334 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_8
  simp only [← c73329, ← c73330, ← c73331] at e_1190_8
  have e_1190_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_9
  simp only [← c73332, ← c73333, k2, ← c73334] at e_1190_9
  have hr := e_1190_9
  simp only [e_1190_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10703)) := by
  have c73335 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73336 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73337 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73338 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73339 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73340 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_10
  simp only [← c73335, ← c73336, ← c73337] at e_1190_10
  have e_1190_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_11
  simp only [← c73338, ← c73339, k2, ← c73340] at e_1190_11
  have hr := e_1190_11
  simp only [e_1190_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10704)) := by
  have c73341 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73342 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73343 := (publicBatchWrapper86_copies2291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73344 := (publicBatchWrapper86_copies2292 a h).1
  have c73345 := (publicBatchWrapper86_copies2292 a h).2.1
  have c73346 := (publicBatchWrapper86_copies2292 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_12
  simp only [← c73341, ← c73342, ← c73343] at e_1190_12
  have e_1190_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_13
  simp only [← c73344, ← c73345, k2, ← c73346] at e_1190_13
  have hr := e_1190_13
  simp only [e_1190_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10705)) := by
  have c73347 := (publicBatchWrapper86_copies2292 a h).2.2.2.1
  have c73348 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.1
  have c73349 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.1
  have c73350 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.1
  have c73351 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.1
  have c73352 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_14
  simp only [← c73347, ← c73348, ← c73349] at e_1190_14
  have e_1190_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_15
  simp only [← c73350, ← c73351, k2, ← c73352] at e_1190_15
  have hr := e_1190_15
  simp only [e_1190_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10706)) := by
  have c73353 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.1
  have c73354 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73355 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73356 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73357 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73358 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_16
  simp only [← c73353, ← c73354, ← c73355] at e_1190_16
  have e_1190_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_17
  simp only [← c73356, ← c73357, k2, ← c73358] at e_1190_17
  have hr := e_1190_17
  simp only [e_1190_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1190 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10707)) := by
  have c73359 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73360 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73361 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73362 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73363 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73364 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1190_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_18
  simp only [← c73359, ← c73360, ← c73361] at e_1190_18
  have e_1190_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1190 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1190_19
  simp only [← c73362, ← c73363, k2, ← c73364] at e_1190_19
  have hr := e_1190_19
  simp only [e_1190_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10708)) := by
  have c73365 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73366 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73367 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73368 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73369 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73370 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_0
  simp only [← c73365, ← c73366, ← c73367] at e_1191_0
  have e_1191_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_1
  simp only [← c73368, ← c73369, k2, ← c73370] at e_1191_1
  have hr := e_1191_1
  simp only [e_1191_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10709)) := by
  have c73371 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73372 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73373 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73374 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73375 := (publicBatchWrapper86_copies2292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73376 := (publicBatchWrapper86_copies2293 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_2
  simp only [← c73371, ← c73372, ← c73373] at e_1191_2
  have e_1191_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_3
  simp only [← c73374, ← c73375, k2, ← c73376] at e_1191_3
  have hr := e_1191_3
  simp only [e_1191_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10710)) := by
  have c73377 := (publicBatchWrapper86_copies2293 a h).2.1
  have c73378 := (publicBatchWrapper86_copies2293 a h).2.2.1
  have c73379 := (publicBatchWrapper86_copies2293 a h).2.2.2.1
  have c73380 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.1
  have c73381 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.1
  have c73382 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_4
  simp only [← c73377, ← c73378, ← c73379] at e_1191_4
  have e_1191_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_5
  simp only [← c73380, ← c73381, k2, ← c73382] at e_1191_5
  have hr := e_1191_5
  simp only [e_1191_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10711)) := by
  have c73383 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.1
  have c73384 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.1
  have c73385 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.1
  have c73386 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73387 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73388 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_6
  simp only [← c73383, ← c73384, ← c73385] at e_1191_6
  have e_1191_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_7
  simp only [← c73386, ← c73387, k2, ← c73388] at e_1191_7
  have hr := e_1191_7
  simp only [e_1191_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10712)) := by
  have c73389 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73390 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73391 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73392 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73393 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73394 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_8
  simp only [← c73389, ← c73390, ← c73391] at e_1191_8
  have e_1191_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_9
  simp only [← c73392, ← c73393, k2, ← c73394] at e_1191_9
  have hr := e_1191_9
  simp only [e_1191_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10713)) := by
  have c73395 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73396 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73397 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73398 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73399 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73400 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_10
  simp only [← c73395, ← c73396, ← c73397] at e_1191_10
  have e_1191_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_11
  simp only [← c73398, ← c73399, k2, ← c73400] at e_1191_11
  have hr := e_1191_11
  simp only [e_1191_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10714)) := by
  have c73401 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73402 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73403 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73404 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73405 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73406 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_12
  simp only [← c73401, ← c73402, ← c73403] at e_1191_12
  have e_1191_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_13
  simp only [← c73404, ← c73405, k2, ← c73406] at e_1191_13
  have hr := e_1191_13
  simp only [e_1191_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10715)) := by
  have c73407 := (publicBatchWrapper86_copies2293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73408 := (publicBatchWrapper86_copies2294 a h).1
  have c73409 := (publicBatchWrapper86_copies2294 a h).2.1
  have c73410 := (publicBatchWrapper86_copies2294 a h).2.2.1
  have c73411 := (publicBatchWrapper86_copies2294 a h).2.2.2.1
  have c73412 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_14
  simp only [← c73407, ← c73408, ← c73409] at e_1191_14
  have e_1191_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_15
  simp only [← c73410, ← c73411, k2, ← c73412] at e_1191_15
  have hr := e_1191_15
  simp only [e_1191_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10716)) := by
  have c73413 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.1
  have c73414 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.1
  have c73415 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.1
  have c73416 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.1
  have c73417 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.1
  have c73418 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_16
  simp only [← c73413, ← c73414, ← c73415] at e_1191_16
  have e_1191_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_17
  simp only [← c73416, ← c73417, k2, ← c73418] at e_1191_17
  have hr := e_1191_17
  simp only [e_1191_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1191 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10717)) := by
  have c73419 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73420 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73421 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73422 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73423 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73424 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1191_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_18
  simp only [← c73419, ← c73420, ← c73421] at e_1191_18
  have e_1191_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1191 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1191_19
  simp only [← c73422, ← c73423, k2, ← c73424] at e_1191_19
  have hr := e_1191_19
  simp only [e_1191_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10718)) := by
  have c73425 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73426 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73427 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73428 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73429 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73430 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_0
  simp only [← c73425, ← c73426, ← c73427] at e_1192_0
  have e_1192_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_1
  simp only [← c73428, ← c73429, k2, ← c73430] at e_1192_1
  have hr := e_1192_1
  simp only [e_1192_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10719)) := by
  have c73431 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73432 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73433 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73434 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73435 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73436 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_2
  simp only [← c73431, ← c73432, ← c73433] at e_1192_2
  have e_1192_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_3
  simp only [← c73434, ← c73435, k2, ← c73436] at e_1192_3
  have hr := e_1192_3
  simp only [e_1192_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10720)) := by
  have c73437 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73438 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73439 := (publicBatchWrapper86_copies2294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73440 := (publicBatchWrapper86_copies2295 a h).1
  have c73441 := (publicBatchWrapper86_copies2295 a h).2.1
  have c73442 := (publicBatchWrapper86_copies2295 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_4
  simp only [← c73437, ← c73438, ← c73439] at e_1192_4
  have e_1192_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_5
  simp only [← c73440, ← c73441, k2, ← c73442] at e_1192_5
  have hr := e_1192_5
  simp only [e_1192_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10721)) := by
  have c73443 := (publicBatchWrapper86_copies2295 a h).2.2.2.1
  have c73444 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.1
  have c73445 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.1
  have c73446 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.1
  have c73447 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.1
  have c73448 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_6
  simp only [← c73443, ← c73444, ← c73445] at e_1192_6
  have e_1192_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_7
  simp only [← c73446, ← c73447, k2, ← c73448] at e_1192_7
  have hr := e_1192_7
  simp only [e_1192_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10722)) := by
  have c73449 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.1
  have c73450 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73451 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73452 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73453 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73454 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_8
  simp only [← c73449, ← c73450, ← c73451] at e_1192_8
  have e_1192_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_9
  simp only [← c73452, ← c73453, k2, ← c73454] at e_1192_9
  have hr := e_1192_9
  simp only [e_1192_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10723)) := by
  have c73455 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73456 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73457 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73458 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73459 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73460 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_10
  simp only [← c73455, ← c73456, ← c73457] at e_1192_10
  have e_1192_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_11
  simp only [← c73458, ← c73459, k2, ← c73460] at e_1192_11
  have hr := e_1192_11
  simp only [e_1192_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10724)) := by
  have c73461 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73462 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73463 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73464 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73465 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73466 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_12
  simp only [← c73461, ← c73462, ← c73463] at e_1192_12
  have e_1192_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_13
  simp only [← c73464, ← c73465, k2, ← c73466] at e_1192_13
  have hr := e_1192_13
  simp only [e_1192_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10725)) := by
  have c73467 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73468 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73469 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73470 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73471 := (publicBatchWrapper86_copies2295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73472 := (publicBatchWrapper86_copies2296 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_14
  simp only [← c73467, ← c73468, ← c73469] at e_1192_14
  have e_1192_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_15
  simp only [← c73470, ← c73471, k2, ← c73472] at e_1192_15
  have hr := e_1192_15
  simp only [e_1192_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10726)) := by
  have c73473 := (publicBatchWrapper86_copies2296 a h).2.1
  have c73474 := (publicBatchWrapper86_copies2296 a h).2.2.1
  have c73475 := (publicBatchWrapper86_copies2296 a h).2.2.2.1
  have c73476 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.1
  have c73477 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.1
  have c73478 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_16
  simp only [← c73473, ← c73474, ← c73475] at e_1192_16
  have e_1192_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_17
  simp only [← c73476, ← c73477, k2, ← c73478] at e_1192_17
  have hr := e_1192_17
  simp only [e_1192_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1192 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10727)) := by
  have c73479 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.1
  have c73480 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.1
  have c73481 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.1
  have c73482 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73483 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73484 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1192_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_18
  simp only [← c73479, ← c73480, ← c73481] at e_1192_18
  have e_1192_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1192 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1192_19
  simp only [← c73482, ← c73483, k2, ← c73484] at e_1192_19
  have hr := e_1192_19
  simp only [e_1192_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10728)) := by
  have c73485 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73486 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73487 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73488 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73489 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73490 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_0
  simp only [← c73485, ← c73486, ← c73487] at e_1193_0
  have e_1193_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_1
  simp only [← c73488, ← c73489, k2, ← c73490] at e_1193_1
  have hr := e_1193_1
  simp only [e_1193_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10729)) := by
  have c73491 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73492 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73493 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73494 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73495 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73496 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_2
  simp only [← c73491, ← c73492, ← c73493] at e_1193_2
  have e_1193_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_3
  simp only [← c73494, ← c73495, k2, ← c73496] at e_1193_3
  have hr := e_1193_3
  simp only [e_1193_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10730)) := by
  have c73497 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73498 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73499 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73500 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73501 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73502 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_4
  simp only [← c73497, ← c73498, ← c73499] at e_1193_4
  have e_1193_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_5
  simp only [← c73500, ← c73501, k2, ← c73502] at e_1193_5
  have hr := e_1193_5
  simp only [e_1193_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10731)) := by
  have c73503 := (publicBatchWrapper86_copies2296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73504 := (publicBatchWrapper86_copies2297 a h).1
  have c73505 := (publicBatchWrapper86_copies2297 a h).2.1
  have c73506 := (publicBatchWrapper86_copies2297 a h).2.2.1
  have c73507 := (publicBatchWrapper86_copies2297 a h).2.2.2.1
  have c73508 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_6
  simp only [← c73503, ← c73504, ← c73505] at e_1193_6
  have e_1193_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_7
  simp only [← c73506, ← c73507, k2, ← c73508] at e_1193_7
  have hr := e_1193_7
  simp only [e_1193_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10732)) := by
  have c73509 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.1
  have c73510 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.1
  have c73511 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.1
  have c73512 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.1
  have c73513 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.1
  have c73514 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_8
  simp only [← c73509, ← c73510, ← c73511] at e_1193_8
  have e_1193_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_9
  simp only [← c73512, ← c73513, k2, ← c73514] at e_1193_9
  have hr := e_1193_9
  simp only [e_1193_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10733)) := by
  have c73515 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73516 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73517 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73518 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73519 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73520 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_10
  simp only [← c73515, ← c73516, ← c73517] at e_1193_10
  have e_1193_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_11
  simp only [← c73518, ← c73519, k2, ← c73520] at e_1193_11
  have hr := e_1193_11
  simp only [e_1193_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10734)) := by
  have c73521 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73522 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73523 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73524 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73525 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73526 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_12
  simp only [← c73521, ← c73522, ← c73523] at e_1193_12
  have e_1193_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_13
  simp only [← c73524, ← c73525, k2, ← c73526] at e_1193_13
  have hr := e_1193_13
  simp only [e_1193_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10735)) := by
  have c73527 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73528 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73529 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73530 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73531 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73532 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_14
  simp only [← c73527, ← c73528, ← c73529] at e_1193_14
  have e_1193_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_15
  simp only [← c73530, ← c73531, k2, ← c73532] at e_1193_15
  have hr := e_1193_15
  simp only [e_1193_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10736)) := by
  have c73533 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73534 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73535 := (publicBatchWrapper86_copies2297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73536 := (publicBatchWrapper86_copies2298 a h).1
  have c73537 := (publicBatchWrapper86_copies2298 a h).2.1
  have c73538 := (publicBatchWrapper86_copies2298 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_16
  simp only [← c73533, ← c73534, ← c73535] at e_1193_16
  have e_1193_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_17
  simp only [← c73536, ← c73537, k2, ← c73538] at e_1193_17
  have hr := e_1193_17
  simp only [e_1193_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1193 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10737)) := by
  have c73539 := (publicBatchWrapper86_copies2298 a h).2.2.2.1
  have c73540 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.1
  have c73541 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.1
  have c73542 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.1
  have c73543 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.1
  have c73544 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1193_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_18
  simp only [← c73539, ← c73540, ← c73541] at e_1193_18
  have e_1193_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1193 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1193_19
  simp only [← c73542, ← c73543, k2, ← c73544] at e_1193_19
  have hr := e_1193_19
  simp only [e_1193_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10738)) := by
  have c73545 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.1
  have c73546 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73547 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73548 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73549 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73550 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_0
  simp only [← c73545, ← c73546, ← c73547] at e_1194_0
  have e_1194_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_1
  simp only [← c73548, ← c73549, k2, ← c73550] at e_1194_1
  have hr := e_1194_1
  simp only [e_1194_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10739)) := by
  have c73551 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73552 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73553 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73554 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73555 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73556 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_2
  simp only [← c73551, ← c73552, ← c73553] at e_1194_2
  have e_1194_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_3
  simp only [← c73554, ← c73555, k2, ← c73556] at e_1194_3
  have hr := e_1194_3
  simp only [e_1194_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10740)) := by
  have c73557 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73558 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73559 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73560 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73561 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73562 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_4
  simp only [← c73557, ← c73558, ← c73559] at e_1194_4
  have e_1194_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_5
  simp only [← c73560, ← c73561, k2, ← c73562] at e_1194_5
  have hr := e_1194_5
  simp only [e_1194_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10741)) := by
  have c73563 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73564 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73565 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73566 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73567 := (publicBatchWrapper86_copies2298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73568 := (publicBatchWrapper86_copies2299 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_6
  simp only [← c73563, ← c73564, ← c73565] at e_1194_6
  have e_1194_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_7
  simp only [← c73566, ← c73567, k2, ← c73568] at e_1194_7
  have hr := e_1194_7
  simp only [e_1194_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10742)) := by
  have c73569 := (publicBatchWrapper86_copies2299 a h).2.1
  have c73570 := (publicBatchWrapper86_copies2299 a h).2.2.1
  have c73571 := (publicBatchWrapper86_copies2299 a h).2.2.2.1
  have c73572 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.1
  have c73573 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.1
  have c73574 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_8
  simp only [← c73569, ← c73570, ← c73571] at e_1194_8
  have e_1194_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_9
  simp only [← c73572, ← c73573, k2, ← c73574] at e_1194_9
  have hr := e_1194_9
  simp only [e_1194_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10743)) := by
  have c73575 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.1
  have c73576 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.1
  have c73577 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.1
  have c73578 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73579 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73580 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_10
  simp only [← c73575, ← c73576, ← c73577] at e_1194_10
  have e_1194_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_11
  simp only [← c73578, ← c73579, k2, ← c73580] at e_1194_11
  have hr := e_1194_11
  simp only [e_1194_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10744)) := by
  have c73581 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73582 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73583 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73584 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73585 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73586 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_12
  simp only [← c73581, ← c73582, ← c73583] at e_1194_12
  have e_1194_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_13
  simp only [← c73584, ← c73585, k2, ← c73586] at e_1194_13
  have hr := e_1194_13
  simp only [e_1194_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10745)) := by
  have c73587 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73588 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73589 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73590 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73591 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73592 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_14
  simp only [← c73587, ← c73588, ← c73589] at e_1194_14
  have e_1194_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_15
  simp only [← c73590, ← c73591, k2, ← c73592] at e_1194_15
  have hr := e_1194_15
  simp only [e_1194_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10746)) := by
  have c73593 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73594 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73595 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73596 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73597 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73598 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_16
  simp only [← c73593, ← c73594, ← c73595] at e_1194_16
  have e_1194_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_17
  simp only [← c73596, ← c73597, k2, ← c73598] at e_1194_17
  have hr := e_1194_17
  simp only [e_1194_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1194 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10747)) := by
  have c73599 := (publicBatchWrapper86_copies2299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73600 := (publicBatchWrapper86_copies2300 a h).1
  have c73601 := (publicBatchWrapper86_copies2300 a h).2.1
  have c73602 := (publicBatchWrapper86_copies2300 a h).2.2.1
  have c73603 := (publicBatchWrapper86_copies2300 a h).2.2.2.1
  have c73604 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1194_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_18
  simp only [← c73599, ← c73600, ← c73601] at e_1194_18
  have e_1194_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1194 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1194_19
  simp only [← c73602, ← c73603, k2, ← c73604] at e_1194_19
  have hr := e_1194_19
  simp only [e_1194_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10748)) := by
  have c73605 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.1
  have c73606 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.1
  have c73607 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.1
  have c73608 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.1
  have c73609 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.1
  have c73610 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_0
  simp only [← c73605, ← c73606, ← c73607] at e_1195_0
  have e_1195_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_1
  simp only [← c73608, ← c73609, k2, ← c73610] at e_1195_1
  have hr := e_1195_1
  simp only [e_1195_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10749)) := by
  have c73611 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73612 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73613 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73614 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73615 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73616 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_2
  simp only [← c73611, ← c73612, ← c73613] at e_1195_2
  have e_1195_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_3
  simp only [← c73614, ← c73615, k2, ← c73616] at e_1195_3
  have hr := e_1195_3
  simp only [e_1195_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10750)) := by
  have c73617 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73618 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73619 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73620 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73621 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73622 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_4
  simp only [← c73617, ← c73618, ← c73619] at e_1195_4
  have e_1195_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_5
  simp only [← c73620, ← c73621, k2, ← c73622] at e_1195_5
  have hr := e_1195_5
  simp only [e_1195_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10751)) := by
  have c73623 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73624 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73625 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73626 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73627 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73628 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_6
  simp only [← c73623, ← c73624, ← c73625] at e_1195_6
  have e_1195_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_7
  simp only [← c73626, ← c73627, k2, ← c73628] at e_1195_7
  have hr := e_1195_7
  simp only [e_1195_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10752)) := by
  have c73629 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73630 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73631 := (publicBatchWrapper86_copies2300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73632 := (publicBatchWrapper86_copies2301 a h).1
  have c73633 := (publicBatchWrapper86_copies2301 a h).2.1
  have c73634 := (publicBatchWrapper86_copies2301 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_8
  simp only [← c73629, ← c73630, ← c73631] at e_1195_8
  have e_1195_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_9
  simp only [← c73632, ← c73633, k2, ← c73634] at e_1195_9
  have hr := e_1195_9
  simp only [e_1195_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10753)) := by
  have c73635 := (publicBatchWrapper86_copies2301 a h).2.2.2.1
  have c73636 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.1
  have c73637 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.1
  have c73638 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.1
  have c73639 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.1
  have c73640 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_10
  simp only [← c73635, ← c73636, ← c73637] at e_1195_10
  have e_1195_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_11
  simp only [← c73638, ← c73639, k2, ← c73640] at e_1195_11
  have hr := e_1195_11
  simp only [e_1195_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10754)) := by
  have c73641 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.1
  have c73642 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73643 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73644 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73645 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73646 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_12
  simp only [← c73641, ← c73642, ← c73643] at e_1195_12
  have e_1195_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_13
  simp only [← c73644, ← c73645, k2, ← c73646] at e_1195_13
  have hr := e_1195_13
  simp only [e_1195_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10755)) := by
  have c73647 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73648 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73649 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73650 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73651 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73652 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_14
  simp only [← c73647, ← c73648, ← c73649] at e_1195_14
  have e_1195_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_15
  simp only [← c73650, ← c73651, k2, ← c73652] at e_1195_15
  have hr := e_1195_15
  simp only [e_1195_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10756)) := by
  have c73653 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73654 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73655 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73656 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73657 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73658 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_16
  simp only [← c73653, ← c73654, ← c73655] at e_1195_16
  have e_1195_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_17
  simp only [← c73656, ← c73657, k2, ← c73658] at e_1195_17
  have hr := e_1195_17
  simp only [e_1195_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1195 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10757)) := by
  have c73659 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73660 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73661 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73662 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73663 := (publicBatchWrapper86_copies2301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73664 := (publicBatchWrapper86_copies2302 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1195_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_18
  simp only [← c73659, ← c73660, ← c73661] at e_1195_18
  have e_1195_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1195 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1195_19
  simp only [← c73662, ← c73663, k2, ← c73664] at e_1195_19
  have hr := e_1195_19
  simp only [e_1195_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10758)) := by
  have c73665 := (publicBatchWrapper86_copies2302 a h).2.1
  have c73666 := (publicBatchWrapper86_copies2302 a h).2.2.1
  have c73667 := (publicBatchWrapper86_copies2302 a h).2.2.2.1
  have c73668 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.1
  have c73669 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.1
  have c73670 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_0
  simp only [← c73665, ← c73666, ← c73667] at e_1196_0
  have e_1196_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_1
  simp only [← c73668, ← c73669, k2, ← c73670] at e_1196_1
  have hr := e_1196_1
  simp only [e_1196_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10759)) := by
  have c73671 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.1
  have c73672 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.1
  have c73673 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.1
  have c73674 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73675 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73676 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_2
  simp only [← c73671, ← c73672, ← c73673] at e_1196_2
  have e_1196_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_3
  simp only [← c73674, ← c73675, k2, ← c73676] at e_1196_3
  have hr := e_1196_3
  simp only [e_1196_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 23) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10760)) := by
  have c73677 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73678 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73679 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73680 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73681 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73682 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_4
  simp only [← c73677, ← c73678, ← c73679] at e_1196_4
  have e_1196_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_5
  simp only [← c73680, ← c73681, k2, ← c73682] at e_1196_5
  have hr := e_1196_5
  simp only [e_1196_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 31) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10761)) := by
  have c73683 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73684 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73685 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73686 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73687 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73688 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_6
  simp only [← c73683, ← c73684, ← c73685] at e_1196_6
  have e_1196_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_7
  simp only [← c73686, ← c73687, k2, ← c73688] at e_1196_7
  have hr := e_1196_7
  simp only [e_1196_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 39) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10762)) := by
  have c73689 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73690 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73691 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73692 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73693 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73694 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_8
  simp only [← c73689, ← c73690, ← c73691] at e_1196_8
  have e_1196_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_9
  simp only [← c73692, ← c73693, k2, ← c73694] at e_1196_9
  have hr := e_1196_9
  simp only [e_1196_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 47) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10763)) := by
  have c73695 := (publicBatchWrapper86_copies2302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73696 := (publicBatchWrapper86_copies2303 a h).1
  have c73697 := (publicBatchWrapper86_copies2303 a h).2.1
  have c73698 := (publicBatchWrapper86_copies2303 a h).2.2.1
  have c73699 := (publicBatchWrapper86_copies2303 a h).2.2.2.1
  have c73700 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_10
  simp only [← c73695, ← c73696, ← c73697] at e_1196_10
  have e_1196_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_11
  simp only [← c73698, ← c73699, k2, ← c73700] at e_1196_11
  have hr := e_1196_11
  simp only [e_1196_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 55) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10764)) := by
  have c73701 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.1
  have c73702 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.1
  have c73703 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.1
  have c73704 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.1
  have c73705 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.1
  have c73706 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_12
  simp only [← c73701, ← c73702, ← c73703] at e_1196_12
  have e_1196_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_13
  simp only [← c73704, ← c73705, k2, ← c73706] at e_1196_13
  have hr := e_1196_13
  simp only [e_1196_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 63) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10765)) := by
  have c73707 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73708 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73709 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73710 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73711 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73712 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_14
  simp only [← c73707, ← c73708, ← c73709] at e_1196_14
  have e_1196_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_15
  simp only [← c73710, ← c73711, k2, ← c73712] at e_1196_15
  have hr := e_1196_15
  simp only [e_1196_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 71) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10766)) := by
  have c73713 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73714 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73715 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73716 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73717 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73718 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_16
  simp only [← c73713, ← c73714, ← c73715] at e_1196_16
  have e_1196_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_17
  simp only [← c73716, ← c73717, k2, ← c73718] at e_1196_17
  have hr := e_1196_17
  simp only [e_1196_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1196 79) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10767)) := by
  have c73719 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73720 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73721 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73722 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73723 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73724 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1196_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_18
  simp only [← c73719, ← c73720, ← c73721] at e_1196_18
  have e_1196_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1196 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1196_19
  simp only [← c73722, ← c73723, k2, ← c73724] at e_1196_19
  have hr := e_1196_19
  simp only [e_1196_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 7) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10768)) := by
  have c73725 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73726 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73727 := (publicBatchWrapper86_copies2303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73728 := (publicBatchWrapper86_copies2304 a h).1
  have c73729 := (publicBatchWrapper86_copies2304 a h).2.1
  have c73730 := (publicBatchWrapper86_copies2304 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_0
  simp only [← c73725, ← c73726, ← c73727] at e_1197_0
  have e_1197_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_1
  simp only [← c73728, ← c73729, k2, ← c73730] at e_1197_1
  have hr := e_1197_1
  simp only [e_1197_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 15) = bselect (a (.wire 53 39)) (a (.virt 200)) (a (.virt 10769)) := by
  have c73731 := (publicBatchWrapper86_copies2304 a h).2.2.2.1
  have c73732 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.1
  have c73733 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.1
  have c73734 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.1
  have c73735 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.1
  have c73736 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_2
  simp only [← c73731, ← c73732, ← c73733] at e_1197_2
  have e_1197_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_3
  simp only [← c73734, ← c73735, k2, ← c73736] at e_1197_3
  have hr := e_1197_3
  simp only [e_1197_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10810)) := by
  have c73737 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.1
  have c73738 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73739 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73740 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73741 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73742 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_4
  simp only [← c73737, ← c73738, ← c73739] at e_1197_4
  have e_1197_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_5
  simp only [← c73740, ← c73741, k2, ← c73742] at e_1197_5
  have hr := e_1197_5
  simp only [e_1197_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10811)) := by
  have c73743 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73744 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73745 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73746 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73747 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73748 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_6
  simp only [← c73743, ← c73744, ← c73745] at e_1197_6
  have e_1197_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_7
  simp only [← c73746, ← c73747, k2, ← c73748] at e_1197_7
  have hr := e_1197_7
  simp only [e_1197_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10812)) := by
  have c73749 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73750 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73751 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73752 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73753 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73754 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_8
  simp only [← c73749, ← c73750, ← c73751] at e_1197_8
  have e_1197_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_9
  simp only [← c73752, ← c73753, k2, ← c73754] at e_1197_9
  have hr := e_1197_9
  simp only [e_1197_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10813)) := by
  have c73755 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73756 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73757 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73758 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73759 := (publicBatchWrapper86_copies2304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73760 := (publicBatchWrapper86_copies2305 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_10
  simp only [← c73755, ← c73756, ← c73757] at e_1197_10
  have e_1197_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_11
  simp only [← c73758, ← c73759, k2, ← c73760] at e_1197_11
  have hr := e_1197_11
  simp only [e_1197_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10814)) := by
  have c73761 := (publicBatchWrapper86_copies2305 a h).2.1
  have c73762 := (publicBatchWrapper86_copies2305 a h).2.2.1
  have c73763 := (publicBatchWrapper86_copies2305 a h).2.2.2.1
  have c73764 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.1
  have c73765 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.1
  have c73766 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_12
  simp only [← c73761, ← c73762, ← c73763] at e_1197_12
  have e_1197_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_13
  simp only [← c73764, ← c73765, k2, ← c73766] at e_1197_13
  have hr := e_1197_13
  simp only [e_1197_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10815)) := by
  have c73767 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.1
  have c73768 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.1
  have c73769 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.1
  have c73770 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73771 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73772 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_14
  simp only [← c73767, ← c73768, ← c73769] at e_1197_14
  have e_1197_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_15
  simp only [← c73770, ← c73771, k2, ← c73772] at e_1197_15
  have hr := e_1197_15
  simp only [e_1197_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10816)) := by
  have c73773 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73774 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73775 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73776 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73777 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73778 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_16
  simp only [← c73773, ← c73774, ← c73775] at e_1197_16
  have e_1197_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_17
  simp only [← c73776, ← c73777, k2, ← c73778] at e_1197_17
  have hr := e_1197_17
  simp only [e_1197_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1197 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10817)) := by
  have c73779 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73780 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73781 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73782 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73783 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73784 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1197_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_18
  simp only [← c73779, ← c73780, ← c73781] at e_1197_18
  have e_1197_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1197 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1197_19
  simp only [← c73782, ← c73783, k2, ← c73784] at e_1197_19
  have hr := e_1197_19
  simp only [e_1197_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10818)) := by
  have c73785 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73786 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73787 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73788 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73789 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73790 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_0
  simp only [← c73785, ← c73786, ← c73787] at e_1198_0
  have e_1198_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_1
  simp only [← c73788, ← c73789, k2, ← c73790] at e_1198_1
  have hr := e_1198_1
  simp only [e_1198_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10819)) := by
  have c73791 := (publicBatchWrapper86_copies2305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73792 := (publicBatchWrapper86_copies2306 a h).1
  have c73793 := (publicBatchWrapper86_copies2306 a h).2.1
  have c73794 := (publicBatchWrapper86_copies2306 a h).2.2.1
  have c73795 := (publicBatchWrapper86_copies2306 a h).2.2.2.1
  have c73796 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_2
  simp only [← c73791, ← c73792, ← c73793] at e_1198_2
  have e_1198_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_3
  simp only [← c73794, ← c73795, k2, ← c73796] at e_1198_3
  have hr := e_1198_3
  simp only [e_1198_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10820)) := by
  have c73797 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.1
  have c73798 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.1
  have c73799 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.1
  have c73800 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.1
  have c73801 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.1
  have c73802 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_4
  simp only [← c73797, ← c73798, ← c73799] at e_1198_4
  have e_1198_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_5
  simp only [← c73800, ← c73801, k2, ← c73802] at e_1198_5
  have hr := e_1198_5
  simp only [e_1198_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10821)) := by
  have c73803 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73804 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73805 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73806 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73807 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73808 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_6
  simp only [← c73803, ← c73804, ← c73805] at e_1198_6
  have e_1198_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_7
  simp only [← c73806, ← c73807, k2, ← c73808] at e_1198_7
  have hr := e_1198_7
  simp only [e_1198_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10822)) := by
  have c73809 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73810 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73811 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73812 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73813 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73814 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_8
  simp only [← c73809, ← c73810, ← c73811] at e_1198_8
  have e_1198_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_9
  simp only [← c73812, ← c73813, k2, ← c73814] at e_1198_9
  have hr := e_1198_9
  simp only [e_1198_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10823)) := by
  have c73815 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73816 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73817 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73818 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73819 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73820 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_10
  simp only [← c73815, ← c73816, ← c73817] at e_1198_10
  have e_1198_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_11
  simp only [← c73818, ← c73819, k2, ← c73820] at e_1198_11
  have hr := e_1198_11
  simp only [e_1198_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10824)) := by
  have c73821 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73822 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73823 := (publicBatchWrapper86_copies2306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73824 := (publicBatchWrapper86_copies2307 a h).1
  have c73825 := (publicBatchWrapper86_copies2307 a h).2.1
  have c73826 := (publicBatchWrapper86_copies2307 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_12
  simp only [← c73821, ← c73822, ← c73823] at e_1198_12
  have e_1198_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_13
  simp only [← c73824, ← c73825, k2, ← c73826] at e_1198_13
  have hr := e_1198_13
  simp only [e_1198_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10825)) := by
  have c73827 := (publicBatchWrapper86_copies2307 a h).2.2.2.1
  have c73828 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.1
  have c73829 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.1
  have c73830 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.1
  have c73831 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.1
  have c73832 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_14
  simp only [← c73827, ← c73828, ← c73829] at e_1198_14
  have e_1198_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_15
  simp only [← c73830, ← c73831, k2, ← c73832] at e_1198_15
  have hr := e_1198_15
  simp only [e_1198_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10826)) := by
  have c73833 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.1
  have c73834 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73835 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73836 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73837 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73838 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_16
  simp only [← c73833, ← c73834, ← c73835] at e_1198_16
  have e_1198_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_17
  simp only [← c73836, ← c73837, k2, ← c73838] at e_1198_17
  have hr := e_1198_17
  simp only [e_1198_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1198 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10827)) := by
  have c73839 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73840 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73841 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73842 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73843 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73844 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1198_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_18
  simp only [← c73839, ← c73840, ← c73841] at e_1198_18
  have e_1198_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1198 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1198_19
  simp only [← c73842, ← c73843, k2, ← c73844] at e_1198_19
  have hr := e_1198_19
  simp only [e_1198_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10828)) := by
  have c73845 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73846 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73847 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73848 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73849 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73850 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_0
  simp only [← c73845, ← c73846, ← c73847] at e_1199_0
  have e_1199_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_1
  simp only [← c73848, ← c73849, k2, ← c73850] at e_1199_1
  have hr := e_1199_1
  simp only [e_1199_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10829)) := by
  have c73851 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73852 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73853 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73854 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73855 := (publicBatchWrapper86_copies2307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73856 := (publicBatchWrapper86_copies2308 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_2
  simp only [← c73851, ← c73852, ← c73853] at e_1199_2
  have e_1199_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_3
  simp only [← c73854, ← c73855, k2, ← c73856] at e_1199_3
  have hr := e_1199_3
  simp only [e_1199_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10830)) := by
  have c73857 := (publicBatchWrapper86_copies2308 a h).2.1
  have c73858 := (publicBatchWrapper86_copies2308 a h).2.2.1
  have c73859 := (publicBatchWrapper86_copies2308 a h).2.2.2.1
  have c73860 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.1
  have c73861 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.1
  have c73862 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_4
  simp only [← c73857, ← c73858, ← c73859] at e_1199_4
  have e_1199_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_5
  simp only [← c73860, ← c73861, k2, ← c73862] at e_1199_5
  have hr := e_1199_5
  simp only [e_1199_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10831)) := by
  have c73863 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.1
  have c73864 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.1
  have c73865 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.1
  have c73866 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73867 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73868 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_6
  simp only [← c73863, ← c73864, ← c73865] at e_1199_6
  have e_1199_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_7
  simp only [← c73866, ← c73867, k2, ← c73868] at e_1199_7
  have hr := e_1199_7
  simp only [e_1199_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10832)) := by
  have c73869 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73870 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73871 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73872 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73873 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73874 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_8
  simp only [← c73869, ← c73870, ← c73871] at e_1199_8
  have e_1199_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_9
  simp only [← c73872, ← c73873, k2, ← c73874] at e_1199_9
  have hr := e_1199_9
  simp only [e_1199_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10833)) := by
  have c73875 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73876 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73877 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73878 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73879 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73880 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_10
  simp only [← c73875, ← c73876, ← c73877] at e_1199_10
  have e_1199_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_11
  simp only [← c73878, ← c73879, k2, ← c73880] at e_1199_11
  have hr := e_1199_11
  simp only [e_1199_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10834)) := by
  have c73881 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73882 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73883 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73884 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73885 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73886 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_12
  simp only [← c73881, ← c73882, ← c73883] at e_1199_12
  have e_1199_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_13
  simp only [← c73884, ← c73885, k2, ← c73886] at e_1199_13
  have hr := e_1199_13
  simp only [e_1199_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10835)) := by
  have c73887 := (publicBatchWrapper86_copies2308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73888 := (publicBatchWrapper86_copies2309 a h).1
  have c73889 := (publicBatchWrapper86_copies2309 a h).2.1
  have c73890 := (publicBatchWrapper86_copies2309 a h).2.2.1
  have c73891 := (publicBatchWrapper86_copies2309 a h).2.2.2.1
  have c73892 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_14
  simp only [← c73887, ← c73888, ← c73889] at e_1199_14
  have e_1199_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_15
  simp only [← c73890, ← c73891, k2, ← c73892] at e_1199_15
  have hr := e_1199_15
  simp only [e_1199_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10836)) := by
  have c73893 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.1
  have c73894 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.1
  have c73895 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.1
  have c73896 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.1
  have c73897 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.1
  have c73898 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_16
  simp only [← c73893, ← c73894, ← c73895] at e_1199_16
  have e_1199_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_17
  simp only [← c73896, ← c73897, k2, ← c73898] at e_1199_17
  have hr := e_1199_17
  simp only [e_1199_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1199 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10837)) := by
  have c73899 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73900 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73901 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73902 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73903 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73904 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1199_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_18
  simp only [← c73899, ← c73900, ← c73901] at e_1199_18
  have e_1199_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1199 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1199_19
  simp only [← c73902, ← c73903, k2, ← c73904] at e_1199_19
  have hr := e_1199_19
  simp only [e_1199_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10838)) := by
  have c73905 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73906 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73907 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73908 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73909 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73910 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_0
  simp only [← c73905, ← c73906, ← c73907] at e_1200_0
  have e_1200_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_1
  simp only [← c73908, ← c73909, k2, ← c73910] at e_1200_1
  have hr := e_1200_1
  simp only [e_1200_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10839)) := by
  have c73911 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73912 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73913 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73914 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73915 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73916 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_2
  simp only [← c73911, ← c73912, ← c73913] at e_1200_2
  have e_1200_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_3
  simp only [← c73914, ← c73915, k2, ← c73916] at e_1200_3
  have hr := e_1200_3
  simp only [e_1200_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10840)) := by
  have c73917 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73918 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73919 := (publicBatchWrapper86_copies2309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73920 := (publicBatchWrapper86_copies2310 a h).1
  have c73921 := (publicBatchWrapper86_copies2310 a h).2.1
  have c73922 := (publicBatchWrapper86_copies2310 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_4
  simp only [← c73917, ← c73918, ← c73919] at e_1200_4
  have e_1200_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_5
  simp only [← c73920, ← c73921, k2, ← c73922] at e_1200_5
  have hr := e_1200_5
  simp only [e_1200_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10841)) := by
  have c73923 := (publicBatchWrapper86_copies2310 a h).2.2.2.1
  have c73924 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.1
  have c73925 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.1
  have c73926 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.1
  have c73927 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.1
  have c73928 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_6
  simp only [← c73923, ← c73924, ← c73925] at e_1200_6
  have e_1200_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_7
  simp only [← c73926, ← c73927, k2, ← c73928] at e_1200_7
  have hr := e_1200_7
  simp only [e_1200_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10842)) := by
  have c73929 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.1
  have c73930 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73931 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73932 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73933 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73934 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_8
  simp only [← c73929, ← c73930, ← c73931] at e_1200_8
  have e_1200_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_9
  simp only [← c73932, ← c73933, k2, ← c73934] at e_1200_9
  have hr := e_1200_9
  simp only [e_1200_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10843)) := by
  have c73935 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73936 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73937 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73938 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73939 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73940 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_10
  simp only [← c73935, ← c73936, ← c73937] at e_1200_10
  have e_1200_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_11
  simp only [← c73938, ← c73939, k2, ← c73940] at e_1200_11
  have hr := e_1200_11
  simp only [e_1200_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10844)) := by
  have c73941 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73942 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73943 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73944 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73945 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73946 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_12
  simp only [← c73941, ← c73942, ← c73943] at e_1200_12
  have e_1200_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_13
  simp only [← c73944, ← c73945, k2, ← c73946] at e_1200_13
  have hr := e_1200_13
  simp only [e_1200_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10845)) := by
  have c73947 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73948 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73949 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73950 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73951 := (publicBatchWrapper86_copies2310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73952 := (publicBatchWrapper86_copies2311 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_14
  simp only [← c73947, ← c73948, ← c73949] at e_1200_14
  have e_1200_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_15
  simp only [← c73950, ← c73951, k2, ← c73952] at e_1200_15
  have hr := e_1200_15
  simp only [e_1200_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10846)) := by
  have c73953 := (publicBatchWrapper86_copies2311 a h).2.1
  have c73954 := (publicBatchWrapper86_copies2311 a h).2.2.1
  have c73955 := (publicBatchWrapper86_copies2311 a h).2.2.2.1
  have c73956 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.1
  have c73957 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.1
  have c73958 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_16
  simp only [← c73953, ← c73954, ← c73955] at e_1200_16
  have e_1200_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_17
  simp only [← c73956, ← c73957, k2, ← c73958] at e_1200_17
  have hr := e_1200_17
  simp only [e_1200_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1200 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10847)) := by
  have c73959 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.1
  have c73960 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.1
  have c73961 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.1
  have c73962 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.1
  have c73963 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73964 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1200_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_18
  simp only [← c73959, ← c73960, ← c73961] at e_1200_18
  have e_1200_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1200 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1200_19
  simp only [← c73962, ← c73963, k2, ← c73964] at e_1200_19
  have hr := e_1200_19
  simp only [e_1200_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10848)) := by
  have c73965 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73966 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73967 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73968 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73969 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73970 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_0
  simp only [← c73965, ← c73966, ← c73967] at e_1201_0
  have e_1201_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_1
  simp only [← c73968, ← c73969, k2, ← c73970] at e_1201_1
  have hr := e_1201_1
  simp only [e_1201_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10849)) := by
  have c73971 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73972 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73973 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73974 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73975 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73976 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_2
  simp only [← c73971, ← c73972, ← c73973] at e_1201_2
  have e_1201_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_3
  simp only [← c73974, ← c73975, k2, ← c73976] at e_1201_3
  have hr := e_1201_3
  simp only [e_1201_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10850)) := by
  have c73977 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73978 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73979 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73980 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73981 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73982 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_4
  simp only [← c73977, ← c73978, ← c73979] at e_1201_4
  have e_1201_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_5
  simp only [← c73980, ← c73981, k2, ← c73982] at e_1201_5
  have hr := e_1201_5
  simp only [e_1201_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10851)) := by
  have c73983 := (publicBatchWrapper86_copies2311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c73984 := (publicBatchWrapper86_copies2312 a h).1
  have c73985 := (publicBatchWrapper86_copies2312 a h).2.1
  have c73986 := (publicBatchWrapper86_copies2312 a h).2.2.1
  have c73987 := (publicBatchWrapper86_copies2312 a h).2.2.2.1
  have c73988 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_6
  simp only [← c73983, ← c73984, ← c73985] at e_1201_6
  have e_1201_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_7
  simp only [← c73986, ← c73987, k2, ← c73988] at e_1201_7
  have hr := e_1201_7
  simp only [e_1201_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10852)) := by
  have c73989 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.1
  have c73990 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.1
  have c73991 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.1
  have c73992 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.1
  have c73993 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.1
  have c73994 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_8
  simp only [← c73989, ← c73990, ← c73991] at e_1201_8
  have e_1201_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_9
  simp only [← c73992, ← c73993, k2, ← c73994] at e_1201_9
  have hr := e_1201_9
  simp only [e_1201_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10853)) := by
  have c73995 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c73996 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73997 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73998 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c73999 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74000 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_10
  simp only [← c73995, ← c73996, ← c73997] at e_1201_10
  have e_1201_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_11
  simp only [← c73998, ← c73999, k2, ← c74000] at e_1201_11
  have hr := e_1201_11
  simp only [e_1201_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10854)) := by
  have c74001 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74002 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74003 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74004 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74005 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74006 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_12
  simp only [← c74001, ← c74002, ← c74003] at e_1201_12
  have e_1201_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_13
  simp only [← c74004, ← c74005, k2, ← c74006] at e_1201_13
  have hr := e_1201_13
  simp only [e_1201_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10855)) := by
  have c74007 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74008 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74009 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74010 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74011 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74012 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_14
  simp only [← c74007, ← c74008, ← c74009] at e_1201_14
  have e_1201_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_15
  simp only [← c74010, ← c74011, k2, ← c74012] at e_1201_15
  have hr := e_1201_15
  simp only [e_1201_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10856)) := by
  have c74013 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74014 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74015 := (publicBatchWrapper86_copies2312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74016 := (publicBatchWrapper86_copies2313 a h).1
  have c74017 := (publicBatchWrapper86_copies2313 a h).2.1
  have c74018 := (publicBatchWrapper86_copies2313 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_16
  simp only [← c74013, ← c74014, ← c74015] at e_1201_16
  have e_1201_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_17
  simp only [← c74016, ← c74017, k2, ← c74018] at e_1201_17
  have hr := e_1201_17
  simp only [e_1201_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1201 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10857)) := by
  have c74019 := (publicBatchWrapper86_copies2313 a h).2.2.2.1
  have c74020 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.1
  have c74021 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.1
  have c74022 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.1
  have c74023 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.1
  have c74024 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1201_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_18
  simp only [← c74019, ← c74020, ← c74021] at e_1201_18
  have e_1201_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1201 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1201_19
  simp only [← c74022, ← c74023, k2, ← c74024] at e_1201_19
  have hr := e_1201_19
  simp only [e_1201_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10858)) := by
  have c74025 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.1
  have c74026 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74027 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74028 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74029 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74030 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_0
  simp only [← c74025, ← c74026, ← c74027] at e_1202_0
  have e_1202_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_1
  simp only [← c74028, ← c74029, k2, ← c74030] at e_1202_1
  have hr := e_1202_1
  simp only [e_1202_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10859)) := by
  have c74031 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74032 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74033 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74034 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74035 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74036 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_2
  simp only [← c74031, ← c74032, ← c74033] at e_1202_2
  have e_1202_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_3
  simp only [← c74034, ← c74035, k2, ← c74036] at e_1202_3
  have hr := e_1202_3
  simp only [e_1202_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10860)) := by
  have c74037 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74038 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74039 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74040 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74041 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74042 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_4
  simp only [← c74037, ← c74038, ← c74039] at e_1202_4
  have e_1202_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_5
  simp only [← c74040, ← c74041, k2, ← c74042] at e_1202_5
  have hr := e_1202_5
  simp only [e_1202_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10861)) := by
  have c74043 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74044 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74045 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74046 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74047 := (publicBatchWrapper86_copies2313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74048 := (publicBatchWrapper86_copies2314 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_6
  simp only [← c74043, ← c74044, ← c74045] at e_1202_6
  have e_1202_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_7
  simp only [← c74046, ← c74047, k2, ← c74048] at e_1202_7
  have hr := e_1202_7
  simp only [e_1202_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10862)) := by
  have c74049 := (publicBatchWrapper86_copies2314 a h).2.1
  have c74050 := (publicBatchWrapper86_copies2314 a h).2.2.1
  have c74051 := (publicBatchWrapper86_copies2314 a h).2.2.2.1
  have c74052 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.1
  have c74053 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.1
  have c74054 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_8
  simp only [← c74049, ← c74050, ← c74051] at e_1202_8
  have e_1202_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_9
  simp only [← c74052, ← c74053, k2, ← c74054] at e_1202_9
  have hr := e_1202_9
  simp only [e_1202_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10863)) := by
  have c74055 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.1
  have c74056 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.1
  have c74057 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.1
  have c74058 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74059 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74060 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_10
  simp only [← c74055, ← c74056, ← c74057] at e_1202_10
  have e_1202_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_11
  simp only [← c74058, ← c74059, k2, ← c74060] at e_1202_11
  have hr := e_1202_11
  simp only [e_1202_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10864)) := by
  have c74061 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74062 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74063 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74064 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74065 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74066 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_12
  simp only [← c74061, ← c74062, ← c74063] at e_1202_12
  have e_1202_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_13
  simp only [← c74064, ← c74065, k2, ← c74066] at e_1202_13
  have hr := e_1202_13
  simp only [e_1202_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10865)) := by
  have c74067 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74068 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74069 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74070 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74071 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74072 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_14
  simp only [← c74067, ← c74068, ← c74069] at e_1202_14
  have e_1202_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_15
  simp only [← c74070, ← c74071, k2, ← c74072] at e_1202_15
  have hr := e_1202_15
  simp only [e_1202_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10866)) := by
  have c74073 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74074 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74075 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74076 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74077 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74078 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_16
  simp only [← c74073, ← c74074, ← c74075] at e_1202_16
  have e_1202_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_17
  simp only [← c74076, ← c74077, k2, ← c74078] at e_1202_17
  have hr := e_1202_17
  simp only [e_1202_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1202 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10867)) := by
  have c74079 := (publicBatchWrapper86_copies2314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74080 := (publicBatchWrapper86_copies2315 a h).1
  have c74081 := (publicBatchWrapper86_copies2315 a h).2.1
  have c74082 := (publicBatchWrapper86_copies2315 a h).2.2.1
  have c74083 := (publicBatchWrapper86_copies2315 a h).2.2.2.1
  have c74084 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1202_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_18
  simp only [← c74079, ← c74080, ← c74081] at e_1202_18
  have e_1202_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1202 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1202_19
  simp only [← c74082, ← c74083, k2, ← c74084] at e_1202_19
  have hr := e_1202_19
  simp only [e_1202_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10868)) := by
  have c74085 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.1
  have c74086 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.1
  have c74087 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.1
  have c74088 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.1
  have c74089 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.1
  have c74090 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_0
  simp only [← c74085, ← c74086, ← c74087] at e_1203_0
  have e_1203_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_1
  simp only [← c74088, ← c74089, k2, ← c74090] at e_1203_1
  have hr := e_1203_1
  simp only [e_1203_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10869)) := by
  have c74091 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74092 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74093 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74094 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74095 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74096 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_2
  simp only [← c74091, ← c74092, ← c74093] at e_1203_2
  have e_1203_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_3
  simp only [← c74094, ← c74095, k2, ← c74096] at e_1203_3
  have hr := e_1203_3
  simp only [e_1203_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10870)) := by
  have c74097 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74098 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74099 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74100 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74101 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74102 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_4
  simp only [← c74097, ← c74098, ← c74099] at e_1203_4
  have e_1203_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_5
  simp only [← c74100, ← c74101, k2, ← c74102] at e_1203_5
  have hr := e_1203_5
  simp only [e_1203_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10871)) := by
  have c74103 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74104 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74105 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74106 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74107 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74108 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_6
  simp only [← c74103, ← c74104, ← c74105] at e_1203_6
  have e_1203_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_7
  simp only [← c74106, ← c74107, k2, ← c74108] at e_1203_7
  have hr := e_1203_7
  simp only [e_1203_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10872)) := by
  have c74109 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74110 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74111 := (publicBatchWrapper86_copies2315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74112 := (publicBatchWrapper86_copies2316 a h).1
  have c74113 := (publicBatchWrapper86_copies2316 a h).2.1
  have c74114 := (publicBatchWrapper86_copies2316 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_8
  simp only [← c74109, ← c74110, ← c74111] at e_1203_8
  have e_1203_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_9
  simp only [← c74112, ← c74113, k2, ← c74114] at e_1203_9
  have hr := e_1203_9
  simp only [e_1203_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10873)) := by
  have c74115 := (publicBatchWrapper86_copies2316 a h).2.2.2.1
  have c74116 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.1
  have c74117 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.1
  have c74118 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.1
  have c74119 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.1
  have c74120 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_10
  simp only [← c74115, ← c74116, ← c74117] at e_1203_10
  have e_1203_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_11
  simp only [← c74118, ← c74119, k2, ← c74120] at e_1203_11
  have hr := e_1203_11
  simp only [e_1203_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10874)) := by
  have c74121 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.1
  have c74122 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74123 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74124 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74125 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74126 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_12
  simp only [← c74121, ← c74122, ← c74123] at e_1203_12
  have e_1203_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_13
  simp only [← c74124, ← c74125, k2, ← c74126] at e_1203_13
  have hr := e_1203_13
  simp only [e_1203_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10875)) := by
  have c74127 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74128 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74129 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74130 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74131 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74132 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_14
  simp only [← c74127, ← c74128, ← c74129] at e_1203_14
  have e_1203_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_15
  simp only [← c74130, ← c74131, k2, ← c74132] at e_1203_15
  have hr := e_1203_15
  simp only [e_1203_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10876)) := by
  have c74133 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74134 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74135 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74136 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74137 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74138 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_16
  simp only [← c74133, ← c74134, ← c74135] at e_1203_16
  have e_1203_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_17
  simp only [← c74136, ← c74137, k2, ← c74138] at e_1203_17
  have hr := e_1203_17
  simp only [e_1203_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1203 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10877)) := by
  have c74139 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74140 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74141 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74142 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74143 := (publicBatchWrapper86_copies2316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74144 := (publicBatchWrapper86_copies2317 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1203_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_18
  simp only [← c74139, ← c74140, ← c74141] at e_1203_18
  have e_1203_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1203 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1203_19
  simp only [← c74142, ← c74143, k2, ← c74144] at e_1203_19
  have hr := e_1203_19
  simp only [e_1203_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10878)) := by
  have c74145 := (publicBatchWrapper86_copies2317 a h).2.1
  have c74146 := (publicBatchWrapper86_copies2317 a h).2.2.1
  have c74147 := (publicBatchWrapper86_copies2317 a h).2.2.2.1
  have c74148 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.1
  have c74149 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.1
  have c74150 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_0
  simp only [← c74145, ← c74146, ← c74147] at e_1204_0
  have e_1204_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_1
  simp only [← c74148, ← c74149, k2, ← c74150] at e_1204_1
  have hr := e_1204_1
  simp only [e_1204_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10879)) := by
  have c74151 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.1
  have c74152 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.1
  have c74153 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.1
  have c74154 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74155 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74156 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_2
  simp only [← c74151, ← c74152, ← c74153] at e_1204_2
  have e_1204_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_3
  simp only [← c74154, ← c74155, k2, ← c74156] at e_1204_3
  have hr := e_1204_3
  simp only [e_1204_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10880)) := by
  have c74157 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74158 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74159 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74160 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74161 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74162 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_4
  simp only [← c74157, ← c74158, ← c74159] at e_1204_4
  have e_1204_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_5
  simp only [← c74160, ← c74161, k2, ← c74162] at e_1204_5
  have hr := e_1204_5
  simp only [e_1204_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10881)) := by
  have c74163 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74164 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74165 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74166 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74167 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74168 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_6
  simp only [← c74163, ← c74164, ← c74165] at e_1204_6
  have e_1204_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_7
  simp only [← c74166, ← c74167, k2, ← c74168] at e_1204_7
  have hr := e_1204_7
  simp only [e_1204_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10882)) := by
  have c74169 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74170 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74171 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74172 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74173 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74174 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_8
  simp only [← c74169, ← c74170, ← c74171] at e_1204_8
  have e_1204_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_9
  simp only [← c74172, ← c74173, k2, ← c74174] at e_1204_9
  have hr := e_1204_9
  simp only [e_1204_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10883)) := by
  have c74175 := (publicBatchWrapper86_copies2317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74176 := (publicBatchWrapper86_copies2318 a h).1
  have c74177 := (publicBatchWrapper86_copies2318 a h).2.1
  have c74178 := (publicBatchWrapper86_copies2318 a h).2.2.1
  have c74179 := (publicBatchWrapper86_copies2318 a h).2.2.2.1
  have c74180 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_10
  simp only [← c74175, ← c74176, ← c74177] at e_1204_10
  have e_1204_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_11
  simp only [← c74178, ← c74179, k2, ← c74180] at e_1204_11
  have hr := e_1204_11
  simp only [e_1204_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10884)) := by
  have c74181 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.1
  have c74182 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.1
  have c74183 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.1
  have c74184 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.1
  have c74185 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.1
  have c74186 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_12
  simp only [← c74181, ← c74182, ← c74183] at e_1204_12
  have e_1204_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_13
  simp only [← c74184, ← c74185, k2, ← c74186] at e_1204_13
  have hr := e_1204_13
  simp only [e_1204_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10885)) := by
  have c74187 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74188 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74189 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74190 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74191 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74192 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_14
  simp only [← c74187, ← c74188, ← c74189] at e_1204_14
  have e_1204_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_15
  simp only [← c74190, ← c74191, k2, ← c74192] at e_1204_15
  have hr := e_1204_15
  simp only [e_1204_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10886)) := by
  have c74193 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74194 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74195 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74196 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74197 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74198 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_16
  simp only [← c74193, ← c74194, ← c74195] at e_1204_16
  have e_1204_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_17
  simp only [← c74196, ← c74197, k2, ← c74198] at e_1204_17
  have hr := e_1204_17
  simp only [e_1204_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1204 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10887)) := by
  have c74199 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74200 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74201 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74202 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74203 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74204 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1204_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_18
  simp only [← c74199, ← c74200, ← c74201] at e_1204_18
  have e_1204_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1204 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1204_19
  simp only [← c74202, ← c74203, k2, ← c74204] at e_1204_19
  have hr := e_1204_19
  simp only [e_1204_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10888)) := by
  have c74205 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74206 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74207 := (publicBatchWrapper86_copies2318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74208 := (publicBatchWrapper86_copies2319 a h).1
  have c74209 := (publicBatchWrapper86_copies2319 a h).2.1
  have c74210 := (publicBatchWrapper86_copies2319 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_0
  simp only [← c74205, ← c74206, ← c74207] at e_1205_0
  have e_1205_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_1
  simp only [← c74208, ← c74209, k2, ← c74210] at e_1205_1
  have hr := e_1205_1
  simp only [e_1205_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10889)) := by
  have c74211 := (publicBatchWrapper86_copies2319 a h).2.2.2.1
  have c74212 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.1
  have c74213 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.1
  have c74214 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.1
  have c74215 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.1
  have c74216 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_2
  simp only [← c74211, ← c74212, ← c74213] at e_1205_2
  have e_1205_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_3
  simp only [← c74214, ← c74215, k2, ← c74216] at e_1205_3
  have hr := e_1205_3
  simp only [e_1205_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10890)) := by
  have c74217 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.1
  have c74218 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74219 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74220 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74221 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74222 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_4
  simp only [← c74217, ← c74218, ← c74219] at e_1205_4
  have e_1205_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_5
  simp only [← c74220, ← c74221, k2, ← c74222] at e_1205_5
  have hr := e_1205_5
  simp only [e_1205_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10891)) := by
  have c74223 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74224 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74225 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74226 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74227 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74228 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_6
  simp only [← c74223, ← c74224, ← c74225] at e_1205_6
  have e_1205_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_7
  simp only [← c74226, ← c74227, k2, ← c74228] at e_1205_7
  have hr := e_1205_7
  simp only [e_1205_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10892)) := by
  have c74229 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74230 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74231 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74232 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74233 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74234 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_8
  simp only [← c74229, ← c74230, ← c74231] at e_1205_8
  have e_1205_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_9
  simp only [← c74232, ← c74233, k2, ← c74234] at e_1205_9
  have hr := e_1205_9
  simp only [e_1205_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10893)) := by
  have c74235 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74236 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74237 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74238 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74239 := (publicBatchWrapper86_copies2319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74240 := (publicBatchWrapper86_copies2320 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_10
  simp only [← c74235, ← c74236, ← c74237] at e_1205_10
  have e_1205_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_11
  simp only [← c74238, ← c74239, k2, ← c74240] at e_1205_11
  have hr := e_1205_11
  simp only [e_1205_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10894)) := by
  have c74241 := (publicBatchWrapper86_copies2320 a h).2.1
  have c74242 := (publicBatchWrapper86_copies2320 a h).2.2.1
  have c74243 := (publicBatchWrapper86_copies2320 a h).2.2.2.1
  have c74244 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.1
  have c74245 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.1
  have c74246 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_12
  simp only [← c74241, ← c74242, ← c74243] at e_1205_12
  have e_1205_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_13
  simp only [← c74244, ← c74245, k2, ← c74246] at e_1205_13
  have hr := e_1205_13
  simp only [e_1205_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10895)) := by
  have c74247 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.1
  have c74248 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.1
  have c74249 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.1
  have c74250 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74251 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74252 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_14
  simp only [← c74247, ← c74248, ← c74249] at e_1205_14
  have e_1205_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_15
  simp only [← c74250, ← c74251, k2, ← c74252] at e_1205_15
  have hr := e_1205_15
  simp only [e_1205_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10896)) := by
  have c74253 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74254 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74255 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74256 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74257 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74258 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_16
  simp only [← c74253, ← c74254, ← c74255] at e_1205_16
  have e_1205_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_17
  simp only [← c74256, ← c74257, k2, ← c74258] at e_1205_17
  have hr := e_1205_17
  simp only [e_1205_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1205 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10897)) := by
  have c74259 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74260 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74261 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74262 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74263 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74264 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1205_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_18
  simp only [← c74259, ← c74260, ← c74261] at e_1205_18
  have e_1205_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1205 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1205_19
  simp only [← c74262, ← c74263, k2, ← c74264] at e_1205_19
  have hr := e_1205_19
  simp only [e_1205_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10898)) := by
  have c74265 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74266 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74267 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74268 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74269 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74270 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_0
  simp only [← c74265, ← c74266, ← c74267] at e_1206_0
  have e_1206_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_1
  simp only [← c74268, ← c74269, k2, ← c74270] at e_1206_1
  have hr := e_1206_1
  simp only [e_1206_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10899)) := by
  have c74271 := (publicBatchWrapper86_copies2320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74272 := (publicBatchWrapper86_copies2321 a h).1
  have c74273 := (publicBatchWrapper86_copies2321 a h).2.1
  have c74274 := (publicBatchWrapper86_copies2321 a h).2.2.1
  have c74275 := (publicBatchWrapper86_copies2321 a h).2.2.2.1
  have c74276 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_2
  simp only [← c74271, ← c74272, ← c74273] at e_1206_2
  have e_1206_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_3
  simp only [← c74274, ← c74275, k2, ← c74276] at e_1206_3
  have hr := e_1206_3
  simp only [e_1206_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10900)) := by
  have c74277 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.1
  have c74278 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.1
  have c74279 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.1
  have c74280 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.1
  have c74281 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.1
  have c74282 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_4
  simp only [← c74277, ← c74278, ← c74279] at e_1206_4
  have e_1206_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_5
  simp only [← c74280, ← c74281, k2, ← c74282] at e_1206_5
  have hr := e_1206_5
  simp only [e_1206_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10901)) := by
  have c74283 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74284 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74285 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74286 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74287 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74288 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_6
  simp only [← c74283, ← c74284, ← c74285] at e_1206_6
  have e_1206_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_7
  simp only [← c74286, ← c74287, k2, ← c74288] at e_1206_7
  have hr := e_1206_7
  simp only [e_1206_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10902)) := by
  have c74289 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74290 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74291 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74292 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74293 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74294 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_8
  simp only [← c74289, ← c74290, ← c74291] at e_1206_8
  have e_1206_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_9
  simp only [← c74292, ← c74293, k2, ← c74294] at e_1206_9
  have hr := e_1206_9
  simp only [e_1206_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10903)) := by
  have c74295 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74296 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74297 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74298 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74299 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74300 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_10
  simp only [← c74295, ← c74296, ← c74297] at e_1206_10
  have e_1206_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_11
  simp only [← c74298, ← c74299, k2, ← c74300] at e_1206_11
  have hr := e_1206_11
  simp only [e_1206_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10904)) := by
  have c74301 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74302 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74303 := (publicBatchWrapper86_copies2321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74304 := (publicBatchWrapper86_copies2322 a h).1
  have c74305 := (publicBatchWrapper86_copies2322 a h).2.1
  have c74306 := (publicBatchWrapper86_copies2322 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_12
  simp only [← c74301, ← c74302, ← c74303] at e_1206_12
  have e_1206_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_13
  simp only [← c74304, ← c74305, k2, ← c74306] at e_1206_13
  have hr := e_1206_13
  simp only [e_1206_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10905)) := by
  have c74307 := (publicBatchWrapper86_copies2322 a h).2.2.2.1
  have c74308 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.1
  have c74309 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.1
  have c74310 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.1
  have c74311 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.1
  have c74312 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_14
  simp only [← c74307, ← c74308, ← c74309] at e_1206_14
  have e_1206_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_15
  simp only [← c74310, ← c74311, k2, ← c74312] at e_1206_15
  have hr := e_1206_15
  simp only [e_1206_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10906)) := by
  have c74313 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.1
  have c74314 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74315 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74316 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74317 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74318 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_16
  simp only [← c74313, ← c74314, ← c74315] at e_1206_16
  have e_1206_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_17
  simp only [← c74316, ← c74317, k2, ← c74318] at e_1206_17
  have hr := e_1206_17
  simp only [e_1206_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1206 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10907)) := by
  have c74319 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74320 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74321 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74322 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74323 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74324 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1206_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_18
  simp only [← c74319, ← c74320, ← c74321] at e_1206_18
  have e_1206_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1206 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1206_19
  simp only [← c74322, ← c74323, k2, ← c74324] at e_1206_19
  have hr := e_1206_19
  simp only [e_1206_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10908)) := by
  have c74325 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74326 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74327 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74328 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74329 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74330 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_0
  simp only [← c74325, ← c74326, ← c74327] at e_1207_0
  have e_1207_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_1
  simp only [← c74328, ← c74329, k2, ← c74330] at e_1207_1
  have hr := e_1207_1
  simp only [e_1207_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10909)) := by
  have c74331 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74332 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74333 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74334 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74335 := (publicBatchWrapper86_copies2322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74336 := (publicBatchWrapper86_copies2323 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_2
  simp only [← c74331, ← c74332, ← c74333] at e_1207_2
  have e_1207_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_3
  simp only [← c74334, ← c74335, k2, ← c74336] at e_1207_3
  have hr := e_1207_3
  simp only [e_1207_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10910)) := by
  have c74337 := (publicBatchWrapper86_copies2323 a h).2.1
  have c74338 := (publicBatchWrapper86_copies2323 a h).2.2.1
  have c74339 := (publicBatchWrapper86_copies2323 a h).2.2.2.1
  have c74340 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.1
  have c74341 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.1
  have c74342 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_4
  simp only [← c74337, ← c74338, ← c74339] at e_1207_4
  have e_1207_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_5
  simp only [← c74340, ← c74341, k2, ← c74342] at e_1207_5
  have hr := e_1207_5
  simp only [e_1207_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10911)) := by
  have c74343 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.1
  have c74344 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.1
  have c74345 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.1
  have c74346 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74347 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74348 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_6
  simp only [← c74343, ← c74344, ← c74345] at e_1207_6
  have e_1207_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_7
  simp only [← c74346, ← c74347, k2, ← c74348] at e_1207_7
  have hr := e_1207_7
  simp only [e_1207_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10912)) := by
  have c74349 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74350 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74351 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74352 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74353 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74354 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_8
  simp only [← c74349, ← c74350, ← c74351] at e_1207_8
  have e_1207_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_9
  simp only [← c74352, ← c74353, k2, ← c74354] at e_1207_9
  have hr := e_1207_9
  simp only [e_1207_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10913)) := by
  have c74355 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74356 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74357 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74358 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74359 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74360 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_10
  simp only [← c74355, ← c74356, ← c74357] at e_1207_10
  have e_1207_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_11
  simp only [← c74358, ← c74359, k2, ← c74360] at e_1207_11
  have hr := e_1207_11
  simp only [e_1207_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10914)) := by
  have c74361 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74362 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74363 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74364 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74365 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74366 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_12
  simp only [← c74361, ← c74362, ← c74363] at e_1207_12
  have e_1207_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_13
  simp only [← c74364, ← c74365, k2, ← c74366] at e_1207_13
  have hr := e_1207_13
  simp only [e_1207_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10915)) := by
  have c74367 := (publicBatchWrapper86_copies2323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74368 := (publicBatchWrapper86_copies2324 a h).1
  have c74369 := (publicBatchWrapper86_copies2324 a h).2.1
  have c74370 := (publicBatchWrapper86_copies2324 a h).2.2.1
  have c74371 := (publicBatchWrapper86_copies2324 a h).2.2.2.1
  have c74372 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_14
  simp only [← c74367, ← c74368, ← c74369] at e_1207_14
  have e_1207_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_15
  simp only [← c74370, ← c74371, k2, ← c74372] at e_1207_15
  have hr := e_1207_15
  simp only [e_1207_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10916)) := by
  have c74373 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.1
  have c74374 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.1
  have c74375 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.1
  have c74376 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.1
  have c74377 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.1
  have c74378 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_16
  simp only [← c74373, ← c74374, ← c74375] at e_1207_16
  have e_1207_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_17
  simp only [← c74376, ← c74377, k2, ← c74378] at e_1207_17
  have hr := e_1207_17
  simp only [e_1207_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1207 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10917)) := by
  have c74379 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74380 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74381 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74382 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74383 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74384 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1207_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_18
  simp only [← c74379, ← c74380, ← c74381] at e_1207_18
  have e_1207_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1207 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1207_19
  simp only [← c74382, ← c74383, k2, ← c74384] at e_1207_19
  have hr := e_1207_19
  simp only [e_1207_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10918)) := by
  have c74385 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74386 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74387 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74388 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74389 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74390 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_0
  simp only [← c74385, ← c74386, ← c74387] at e_1208_0
  have e_1208_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_1
  simp only [← c74388, ← c74389, k2, ← c74390] at e_1208_1
  have hr := e_1208_1
  simp only [e_1208_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10919)) := by
  have c74391 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74392 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74393 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74394 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74395 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74396 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_2
  simp only [← c74391, ← c74392, ← c74393] at e_1208_2
  have e_1208_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_3
  simp only [← c74394, ← c74395, k2, ← c74396] at e_1208_3
  have hr := e_1208_3
  simp only [e_1208_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10920)) := by
  have c74397 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74398 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74399 := (publicBatchWrapper86_copies2324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74400 := (publicBatchWrapper86_copies2325 a h).1
  have c74401 := (publicBatchWrapper86_copies2325 a h).2.1
  have c74402 := (publicBatchWrapper86_copies2325 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_4
  simp only [← c74397, ← c74398, ← c74399] at e_1208_4
  have e_1208_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_5
  simp only [← c74400, ← c74401, k2, ← c74402] at e_1208_5
  have hr := e_1208_5
  simp only [e_1208_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10921)) := by
  have c74403 := (publicBatchWrapper86_copies2325 a h).2.2.2.1
  have c74404 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.1
  have c74405 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.1
  have c74406 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.1
  have c74407 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.1
  have c74408 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_6
  simp only [← c74403, ← c74404, ← c74405] at e_1208_6
  have e_1208_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_7
  simp only [← c74406, ← c74407, k2, ← c74408] at e_1208_7
  have hr := e_1208_7
  simp only [e_1208_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10922)) := by
  have c74409 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.1
  have c74410 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74411 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74412 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74413 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74414 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_8
  simp only [← c74409, ← c74410, ← c74411] at e_1208_8
  have e_1208_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_9
  simp only [← c74412, ← c74413, k2, ← c74414] at e_1208_9
  have hr := e_1208_9
  simp only [e_1208_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10923)) := by
  have c74415 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74416 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74417 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74418 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74419 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74420 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_10
  simp only [← c74415, ← c74416, ← c74417] at e_1208_10
  have e_1208_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_11
  simp only [← c74418, ← c74419, k2, ← c74420] at e_1208_11
  have hr := e_1208_11
  simp only [e_1208_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10924)) := by
  have c74421 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74422 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74423 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74424 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74425 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74426 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_12
  simp only [← c74421, ← c74422, ← c74423] at e_1208_12
  have e_1208_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_13
  simp only [← c74424, ← c74425, k2, ← c74426] at e_1208_13
  have hr := e_1208_13
  simp only [e_1208_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10925)) := by
  have c74427 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74428 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74429 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74430 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74431 := (publicBatchWrapper86_copies2325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74432 := (publicBatchWrapper86_copies2326 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_14
  simp only [← c74427, ← c74428, ← c74429] at e_1208_14
  have e_1208_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_15
  simp only [← c74430, ← c74431, k2, ← c74432] at e_1208_15
  have hr := e_1208_15
  simp only [e_1208_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10926)) := by
  have c74433 := (publicBatchWrapper86_copies2326 a h).2.1
  have c74434 := (publicBatchWrapper86_copies2326 a h).2.2.1
  have c74435 := (publicBatchWrapper86_copies2326 a h).2.2.2.1
  have c74436 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.1
  have c74437 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.1
  have c74438 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_16
  simp only [← c74433, ← c74434, ← c74435] at e_1208_16
  have e_1208_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_17
  simp only [← c74436, ← c74437, k2, ← c74438] at e_1208_17
  have hr := e_1208_17
  simp only [e_1208_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1208 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10927)) := by
  have c74439 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.1
  have c74440 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.1
  have c74441 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.1
  have c74442 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74443 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74444 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1208_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_18
  simp only [← c74439, ← c74440, ← c74441] at e_1208_18
  have e_1208_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1208 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1208_19
  simp only [← c74442, ← c74443, k2, ← c74444] at e_1208_19
  have hr := e_1208_19
  simp only [e_1208_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10928)) := by
  have c74445 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74446 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74447 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74448 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74449 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74450 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_0
  simp only [← c74445, ← c74446, ← c74447] at e_1209_0
  have e_1209_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_1
  simp only [← c74448, ← c74449, k2, ← c74450] at e_1209_1
  have hr := e_1209_1
  simp only [e_1209_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10929)) := by
  have c74451 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74452 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74453 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74454 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74455 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74456 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_2
  simp only [← c74451, ← c74452, ← c74453] at e_1209_2
  have e_1209_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_3
  simp only [← c74454, ← c74455, k2, ← c74456] at e_1209_3
  have hr := e_1209_3
  simp only [e_1209_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10930)) := by
  have c74457 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74458 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74459 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74460 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74461 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74462 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_4
  simp only [← c74457, ← c74458, ← c74459] at e_1209_4
  have e_1209_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_5
  simp only [← c74460, ← c74461, k2, ← c74462] at e_1209_5
  have hr := e_1209_5
  simp only [e_1209_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10931)) := by
  have c74463 := (publicBatchWrapper86_copies2326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74464 := (publicBatchWrapper86_copies2327 a h).1
  have c74465 := (publicBatchWrapper86_copies2327 a h).2.1
  have c74466 := (publicBatchWrapper86_copies2327 a h).2.2.1
  have c74467 := (publicBatchWrapper86_copies2327 a h).2.2.2.1
  have c74468 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_6
  simp only [← c74463, ← c74464, ← c74465] at e_1209_6
  have e_1209_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_7
  simp only [← c74466, ← c74467, k2, ← c74468] at e_1209_7
  have hr := e_1209_7
  simp only [e_1209_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10932)) := by
  have c74469 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.1
  have c74470 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.1
  have c74471 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.1
  have c74472 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.1
  have c74473 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.1
  have c74474 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_8
  simp only [← c74469, ← c74470, ← c74471] at e_1209_8
  have e_1209_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_9
  simp only [← c74472, ← c74473, k2, ← c74474] at e_1209_9
  have hr := e_1209_9
  simp only [e_1209_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10933)) := by
  have c74475 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74476 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74477 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74478 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74479 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74480 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_10
  simp only [← c74475, ← c74476, ← c74477] at e_1209_10
  have e_1209_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_11
  simp only [← c74478, ← c74479, k2, ← c74480] at e_1209_11
  have hr := e_1209_11
  simp only [e_1209_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10934)) := by
  have c74481 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74482 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74483 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74484 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74485 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74486 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_12
  simp only [← c74481, ← c74482, ← c74483] at e_1209_12
  have e_1209_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_13
  simp only [← c74484, ← c74485, k2, ← c74486] at e_1209_13
  have hr := e_1209_13
  simp only [e_1209_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10935)) := by
  have c74487 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74488 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74489 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74490 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74491 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74492 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_14
  simp only [← c74487, ← c74488, ← c74489] at e_1209_14
  have e_1209_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_15
  simp only [← c74490, ← c74491, k2, ← c74492] at e_1209_15
  have hr := e_1209_15
  simp only [e_1209_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10936)) := by
  have c74493 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74494 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74495 := (publicBatchWrapper86_copies2327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74496 := (publicBatchWrapper86_copies2328 a h).1
  have c74497 := (publicBatchWrapper86_copies2328 a h).2.1
  have c74498 := (publicBatchWrapper86_copies2328 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_16
  simp only [← c74493, ← c74494, ← c74495] at e_1209_16
  have e_1209_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_17
  simp only [← c74496, ← c74497, k2, ← c74498] at e_1209_17
  have hr := e_1209_17
  simp only [e_1209_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1209 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10937)) := by
  have c74499 := (publicBatchWrapper86_copies2328 a h).2.2.2.1
  have c74500 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.1
  have c74501 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.1
  have c74502 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.1
  have c74503 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.1
  have c74504 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1209_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_18
  simp only [← c74499, ← c74500, ← c74501] at e_1209_18
  have e_1209_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1209 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1209_19
  simp only [← c74502, ← c74503, k2, ← c74504] at e_1209_19
  have hr := e_1209_19
  simp only [e_1209_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10938)) := by
  have c74505 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.1
  have c74506 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74507 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74508 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74509 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74510 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_0
  simp only [← c74505, ← c74506, ← c74507] at e_1210_0
  have e_1210_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_1
  simp only [← c74508, ← c74509, k2, ← c74510] at e_1210_1
  have hr := e_1210_1
  simp only [e_1210_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10939)) := by
  have c74511 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74512 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74513 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74514 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74515 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74516 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_2
  simp only [← c74511, ← c74512, ← c74513] at e_1210_2
  have e_1210_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_3
  simp only [← c74514, ← c74515, k2, ← c74516] at e_1210_3
  have hr := e_1210_3
  simp only [e_1210_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10940)) := by
  have c74517 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74518 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74519 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74520 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74521 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74522 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_4
  simp only [← c74517, ← c74518, ← c74519] at e_1210_4
  have e_1210_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_5
  simp only [← c74520, ← c74521, k2, ← c74522] at e_1210_5
  have hr := e_1210_5
  simp only [e_1210_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10941)) := by
  have c74523 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74524 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74525 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74526 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74527 := (publicBatchWrapper86_copies2328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74528 := (publicBatchWrapper86_copies2329 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_6
  simp only [← c74523, ← c74524, ← c74525] at e_1210_6
  have e_1210_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_7
  simp only [← c74526, ← c74527, k2, ← c74528] at e_1210_7
  have hr := e_1210_7
  simp only [e_1210_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10942)) := by
  have c74529 := (publicBatchWrapper86_copies2329 a h).2.1
  have c74530 := (publicBatchWrapper86_copies2329 a h).2.2.1
  have c74531 := (publicBatchWrapper86_copies2329 a h).2.2.2.1
  have c74532 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.1
  have c74533 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.1
  have c74534 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_8
  simp only [← c74529, ← c74530, ← c74531] at e_1210_8
  have e_1210_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_9
  simp only [← c74532, ← c74533, k2, ← c74534] at e_1210_9
  have hr := e_1210_9
  simp only [e_1210_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10943)) := by
  have c74535 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.1
  have c74536 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.1
  have c74537 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.1
  have c74538 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74539 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74540 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_10
  simp only [← c74535, ← c74536, ← c74537] at e_1210_10
  have e_1210_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_11
  simp only [← c74538, ← c74539, k2, ← c74540] at e_1210_11
  have hr := e_1210_11
  simp only [e_1210_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10944)) := by
  have c74541 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74542 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74543 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74544 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74545 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74546 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_12
  simp only [← c74541, ← c74542, ← c74543] at e_1210_12
  have e_1210_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_13
  simp only [← c74544, ← c74545, k2, ← c74546] at e_1210_13
  have hr := e_1210_13
  simp only [e_1210_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10945)) := by
  have c74547 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74548 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74549 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74550 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74551 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74552 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_14
  simp only [← c74547, ← c74548, ← c74549] at e_1210_14
  have e_1210_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_15
  simp only [← c74550, ← c74551, k2, ← c74552] at e_1210_15
  have hr := e_1210_15
  simp only [e_1210_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10946)) := by
  have c74553 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74554 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74555 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74556 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74557 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74558 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_16
  simp only [← c74553, ← c74554, ← c74555] at e_1210_16
  have e_1210_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_17
  simp only [← c74556, ← c74557, k2, ← c74558] at e_1210_17
  have hr := e_1210_17
  simp only [e_1210_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1210 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10947)) := by
  have c74559 := (publicBatchWrapper86_copies2329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74560 := (publicBatchWrapper86_copies2330 a h).1
  have c74561 := (publicBatchWrapper86_copies2330 a h).2.1
  have c74562 := (publicBatchWrapper86_copies2330 a h).2.2.1
  have c74563 := (publicBatchWrapper86_copies2330 a h).2.2.2.1
  have c74564 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1210_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_18
  simp only [← c74559, ← c74560, ← c74561] at e_1210_18
  have e_1210_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1210 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1210_19
  simp only [← c74562, ← c74563, k2, ← c74564] at e_1210_19
  have hr := e_1210_19
  simp only [e_1210_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10948)) := by
  have c74565 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.1
  have c74566 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.1
  have c74567 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.1
  have c74568 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.1
  have c74569 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.1
  have c74570 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_0
  simp only [← c74565, ← c74566, ← c74567] at e_1211_0
  have e_1211_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_1
  simp only [← c74568, ← c74569, k2, ← c74570] at e_1211_1
  have hr := e_1211_1
  simp only [e_1211_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10949)) := by
  have c74571 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74572 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74573 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74574 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74575 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74576 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_2
  simp only [← c74571, ← c74572, ← c74573] at e_1211_2
  have e_1211_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_3
  simp only [← c74574, ← c74575, k2, ← c74576] at e_1211_3
  have hr := e_1211_3
  simp only [e_1211_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10950)) := by
  have c74577 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74578 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74579 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74580 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74581 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74582 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_4
  simp only [← c74577, ← c74578, ← c74579] at e_1211_4
  have e_1211_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_5
  simp only [← c74580, ← c74581, k2, ← c74582] at e_1211_5
  have hr := e_1211_5
  simp only [e_1211_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10951)) := by
  have c74583 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74584 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74585 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74586 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74587 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74588 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_6
  simp only [← c74583, ← c74584, ← c74585] at e_1211_6
  have e_1211_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_7
  simp only [← c74586, ← c74587, k2, ← c74588] at e_1211_7
  have hr := e_1211_7
  simp only [e_1211_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10952)) := by
  have c74589 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74590 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74591 := (publicBatchWrapper86_copies2330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74592 := (publicBatchWrapper86_copies2331 a h).1
  have c74593 := (publicBatchWrapper86_copies2331 a h).2.1
  have c74594 := (publicBatchWrapper86_copies2331 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_8
  simp only [← c74589, ← c74590, ← c74591] at e_1211_8
  have e_1211_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_9
  simp only [← c74592, ← c74593, k2, ← c74594] at e_1211_9
  have hr := e_1211_9
  simp only [e_1211_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10953)) := by
  have c74595 := (publicBatchWrapper86_copies2331 a h).2.2.2.1
  have c74596 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.1
  have c74597 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.1
  have c74598 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.1
  have c74599 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.1
  have c74600 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_10
  simp only [← c74595, ← c74596, ← c74597] at e_1211_10
  have e_1211_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_11
  simp only [← c74598, ← c74599, k2, ← c74600] at e_1211_11
  have hr := e_1211_11
  simp only [e_1211_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10954)) := by
  have c74601 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.1
  have c74602 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74603 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74604 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74605 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74606 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_12
  simp only [← c74601, ← c74602, ← c74603] at e_1211_12
  have e_1211_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_13
  simp only [← c74604, ← c74605, k2, ← c74606] at e_1211_13
  have hr := e_1211_13
  simp only [e_1211_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10955)) := by
  have c74607 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74608 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74609 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74610 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74611 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74612 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_14
  simp only [← c74607, ← c74608, ← c74609] at e_1211_14
  have e_1211_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_15
  simp only [← c74610, ← c74611, k2, ← c74612] at e_1211_15
  have hr := e_1211_15
  simp only [e_1211_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10956)) := by
  have c74613 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74614 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74615 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74616 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74617 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74618 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_16
  simp only [← c74613, ← c74614, ← c74615] at e_1211_16
  have e_1211_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_17
  simp only [← c74616, ← c74617, k2, ← c74618] at e_1211_17
  have hr := e_1211_17
  simp only [e_1211_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1211 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10957)) := by
  have c74619 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74620 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74621 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74622 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74623 := (publicBatchWrapper86_copies2331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74624 := (publicBatchWrapper86_copies2332 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1211_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_18
  simp only [← c74619, ← c74620, ← c74621] at e_1211_18
  have e_1211_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1211 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1211_19
  simp only [← c74622, ← c74623, k2, ← c74624] at e_1211_19
  have hr := e_1211_19
  simp only [e_1211_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10958)) := by
  have c74625 := (publicBatchWrapper86_copies2332 a h).2.1
  have c74626 := (publicBatchWrapper86_copies2332 a h).2.2.1
  have c74627 := (publicBatchWrapper86_copies2332 a h).2.2.2.1
  have c74628 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.1
  have c74629 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.1
  have c74630 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_0
  simp only [← c74625, ← c74626, ← c74627] at e_1212_0
  have e_1212_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_1
  simp only [← c74628, ← c74629, k2, ← c74630] at e_1212_1
  have hr := e_1212_1
  simp only [e_1212_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10959)) := by
  have c74631 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.1
  have c74632 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.1
  have c74633 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.1
  have c74634 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74635 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74636 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_2
  simp only [← c74631, ← c74632, ← c74633] at e_1212_2
  have e_1212_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_3
  simp only [← c74634, ← c74635, k2, ← c74636] at e_1212_3
  have hr := e_1212_3
  simp only [e_1212_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 23) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10960)) := by
  have c74637 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74638 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74639 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74640 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74641 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74642 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_4
  simp only [← c74637, ← c74638, ← c74639] at e_1212_4
  have e_1212_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_5
  simp only [← c74640, ← c74641, k2, ← c74642] at e_1212_5
  have hr := e_1212_5
  simp only [e_1212_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 31) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10961)) := by
  have c74643 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74644 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74645 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74646 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74647 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74648 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_6
  simp only [← c74643, ← c74644, ← c74645] at e_1212_6
  have e_1212_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_7
  simp only [← c74646, ← c74647, k2, ← c74648] at e_1212_7
  have hr := e_1212_7
  simp only [e_1212_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 39) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10962)) := by
  have c74649 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74650 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74651 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74652 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74653 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74654 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_8
  simp only [← c74649, ← c74650, ← c74651] at e_1212_8
  have e_1212_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_9
  simp only [← c74652, ← c74653, k2, ← c74654] at e_1212_9
  have hr := e_1212_9
  simp only [e_1212_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 47) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10963)) := by
  have c74655 := (publicBatchWrapper86_copies2332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74656 := (publicBatchWrapper86_copies2333 a h).1
  have c74657 := (publicBatchWrapper86_copies2333 a h).2.1
  have c74658 := (publicBatchWrapper86_copies2333 a h).2.2.1
  have c74659 := (publicBatchWrapper86_copies2333 a h).2.2.2.1
  have c74660 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_10
  simp only [← c74655, ← c74656, ← c74657] at e_1212_10
  have e_1212_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_11
  simp only [← c74658, ← c74659, k2, ← c74660] at e_1212_11
  have hr := e_1212_11
  simp only [e_1212_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 55) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10964)) := by
  have c74661 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.1
  have c74662 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.1
  have c74663 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.1
  have c74664 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.1
  have c74665 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.1
  have c74666 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_12
  simp only [← c74661, ← c74662, ← c74663] at e_1212_12
  have e_1212_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_13
  simp only [← c74664, ← c74665, k2, ← c74666] at e_1212_13
  have hr := e_1212_13
  simp only [e_1212_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 63) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10965)) := by
  have c74667 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74668 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74669 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74670 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74671 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74672 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_14
  simp only [← c74667, ← c74668, ← c74669] at e_1212_14
  have e_1212_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_15
  simp only [← c74670, ← c74671, k2, ← c74672] at e_1212_15
  have hr := e_1212_15
  simp only [e_1212_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 71) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10966)) := by
  have c74673 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74674 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74675 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74676 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74677 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74678 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_16
  simp only [← c74673, ← c74674, ← c74675] at e_1212_16
  have e_1212_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_17
  simp only [← c74676, ← c74677, k2, ← c74678] at e_1212_17
  have hr := e_1212_17
  simp only [e_1212_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1212 79) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10967)) := by
  have c74679 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74680 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74681 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74682 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74683 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74684 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1212_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_18
  simp only [← c74679, ← c74680, ← c74681] at e_1212_18
  have e_1212_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1212 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1212_19
  simp only [← c74682, ← c74683, k2, ← c74684] at e_1212_19
  have hr := e_1212_19
  simp only [e_1212_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 7) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10968)) := by
  have c74685 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74686 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74687 := (publicBatchWrapper86_copies2333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74688 := (publicBatchWrapper86_copies2334 a h).1
  have c74689 := (publicBatchWrapper86_copies2334 a h).2.1
  have c74690 := (publicBatchWrapper86_copies2334 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_0
  simp only [← c74685, ← c74686, ← c74687] at e_1213_0
  have e_1213_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_1
  simp only [← c74688, ← c74689, k2, ← c74690] at e_1213_1
  have hr := e_1213_1
  simp only [e_1213_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 15) = bselect (a (.wire 54 3)) (a (.virt 200)) (a (.virt 10969)) := by
  have c74691 := (publicBatchWrapper86_copies2334 a h).2.2.2.1
  have c74692 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.1
  have c74693 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.1
  have c74694 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.1
  have c74695 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.1
  have c74696 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_2
  simp only [← c74691, ← c74692, ← c74693] at e_1213_2
  have e_1213_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_3
  simp only [← c74694, ← c74695, k2, ← c74696] at e_1213_3
  have hr := e_1213_3
  simp only [e_1213_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11010)) := by
  have c74697 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.1
  have c74698 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74699 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74700 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74701 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74702 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_4
  simp only [← c74697, ← c74698, ← c74699] at e_1213_4
  have e_1213_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_5
  simp only [← c74700, ← c74701, k2, ← c74702] at e_1213_5
  have hr := e_1213_5
  simp only [e_1213_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11011)) := by
  have c74703 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74704 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74705 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74706 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74707 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74708 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_6
  simp only [← c74703, ← c74704, ← c74705] at e_1213_6
  have e_1213_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_7
  simp only [← c74706, ← c74707, k2, ← c74708] at e_1213_7
  have hr := e_1213_7
  simp only [e_1213_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11012)) := by
  have c74709 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74710 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74711 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74712 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74713 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74714 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_8
  simp only [← c74709, ← c74710, ← c74711] at e_1213_8
  have e_1213_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_9
  simp only [← c74712, ← c74713, k2, ← c74714] at e_1213_9
  have hr := e_1213_9
  simp only [e_1213_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11013)) := by
  have c74715 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74716 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74717 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74718 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74719 := (publicBatchWrapper86_copies2334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74720 := (publicBatchWrapper86_copies2335 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_10
  simp only [← c74715, ← c74716, ← c74717] at e_1213_10
  have e_1213_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_11
  simp only [← c74718, ← c74719, k2, ← c74720] at e_1213_11
  have hr := e_1213_11
  simp only [e_1213_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11014)) := by
  have c74721 := (publicBatchWrapper86_copies2335 a h).2.1
  have c74722 := (publicBatchWrapper86_copies2335 a h).2.2.1
  have c74723 := (publicBatchWrapper86_copies2335 a h).2.2.2.1
  have c74724 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.1
  have c74725 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.1
  have c74726 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_12
  simp only [← c74721, ← c74722, ← c74723] at e_1213_12
  have e_1213_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_13
  simp only [← c74724, ← c74725, k2, ← c74726] at e_1213_13
  have hr := e_1213_13
  simp only [e_1213_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11015)) := by
  have c74727 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.1
  have c74728 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.1
  have c74729 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.1
  have c74730 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74731 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74732 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_14
  simp only [← c74727, ← c74728, ← c74729] at e_1213_14
  have e_1213_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_15
  simp only [← c74730, ← c74731, k2, ← c74732] at e_1213_15
  have hr := e_1213_15
  simp only [e_1213_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11016)) := by
  have c74733 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74734 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74735 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74736 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74737 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74738 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_16
  simp only [← c74733, ← c74734, ← c74735] at e_1213_16
  have e_1213_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_17
  simp only [← c74736, ← c74737, k2, ← c74738] at e_1213_17
  have hr := e_1213_17
  simp only [e_1213_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1213 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11017)) := by
  have c74739 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74740 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74741 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74742 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74743 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74744 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1213_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_18
  simp only [← c74739, ← c74740, ← c74741] at e_1213_18
  have e_1213_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1213 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1213_19
  simp only [← c74742, ← c74743, k2, ← c74744] at e_1213_19
  have hr := e_1213_19
  simp only [e_1213_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11018)) := by
  have c74745 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74746 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74747 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74748 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74749 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74750 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_0
  simp only [← c74745, ← c74746, ← c74747] at e_1214_0
  have e_1214_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_1
  simp only [← c74748, ← c74749, k2, ← c74750] at e_1214_1
  have hr := e_1214_1
  simp only [e_1214_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11019)) := by
  have c74751 := (publicBatchWrapper86_copies2335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74752 := (publicBatchWrapper86_copies2336 a h).1
  have c74753 := (publicBatchWrapper86_copies2336 a h).2.1
  have c74754 := (publicBatchWrapper86_copies2336 a h).2.2.1
  have c74755 := (publicBatchWrapper86_copies2336 a h).2.2.2.1
  have c74756 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_2
  simp only [← c74751, ← c74752, ← c74753] at e_1214_2
  have e_1214_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_3
  simp only [← c74754, ← c74755, k2, ← c74756] at e_1214_3
  have hr := e_1214_3
  simp only [e_1214_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11020)) := by
  have c74757 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.1
  have c74758 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.1
  have c74759 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.1
  have c74760 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.1
  have c74761 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.1
  have c74762 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_4
  simp only [← c74757, ← c74758, ← c74759] at e_1214_4
  have e_1214_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_5
  simp only [← c74760, ← c74761, k2, ← c74762] at e_1214_5
  have hr := e_1214_5
  simp only [e_1214_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11021)) := by
  have c74763 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74764 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74765 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74766 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74767 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74768 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_6
  simp only [← c74763, ← c74764, ← c74765] at e_1214_6
  have e_1214_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_7
  simp only [← c74766, ← c74767, k2, ← c74768] at e_1214_7
  have hr := e_1214_7
  simp only [e_1214_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11022)) := by
  have c74769 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74770 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74771 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74772 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74773 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74774 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_8
  simp only [← c74769, ← c74770, ← c74771] at e_1214_8
  have e_1214_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_9
  simp only [← c74772, ← c74773, k2, ← c74774] at e_1214_9
  have hr := e_1214_9
  simp only [e_1214_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11023)) := by
  have c74775 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74776 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74777 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74778 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74779 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74780 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_10
  simp only [← c74775, ← c74776, ← c74777] at e_1214_10
  have e_1214_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_11
  simp only [← c74778, ← c74779, k2, ← c74780] at e_1214_11
  have hr := e_1214_11
  simp only [e_1214_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11024)) := by
  have c74781 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74782 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74783 := (publicBatchWrapper86_copies2336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74784 := (publicBatchWrapper86_copies2337 a h).1
  have c74785 := (publicBatchWrapper86_copies2337 a h).2.1
  have c74786 := (publicBatchWrapper86_copies2337 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_12
  simp only [← c74781, ← c74782, ← c74783] at e_1214_12
  have e_1214_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_13
  simp only [← c74784, ← c74785, k2, ← c74786] at e_1214_13
  have hr := e_1214_13
  simp only [e_1214_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11025)) := by
  have c74787 := (publicBatchWrapper86_copies2337 a h).2.2.2.1
  have c74788 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.1
  have c74789 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.1
  have c74790 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.1
  have c74791 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.1
  have c74792 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_14
  simp only [← c74787, ← c74788, ← c74789] at e_1214_14
  have e_1214_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_15
  simp only [← c74790, ← c74791, k2, ← c74792] at e_1214_15
  have hr := e_1214_15
  simp only [e_1214_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11026)) := by
  have c74793 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.1
  have c74794 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74795 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74796 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74797 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74798 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_16
  simp only [← c74793, ← c74794, ← c74795] at e_1214_16
  have e_1214_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_17
  simp only [← c74796, ← c74797, k2, ← c74798] at e_1214_17
  have hr := e_1214_17
  simp only [e_1214_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1214 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11027)) := by
  have c74799 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74800 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74801 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74802 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74803 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74804 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1214_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_18
  simp only [← c74799, ← c74800, ← c74801] at e_1214_18
  have e_1214_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1214 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1214_19
  simp only [← c74802, ← c74803, k2, ← c74804] at e_1214_19
  have hr := e_1214_19
  simp only [e_1214_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11028)) := by
  have c74805 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74806 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74807 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74808 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74809 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74810 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_0
  simp only [← c74805, ← c74806, ← c74807] at e_1215_0
  have e_1215_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_1
  simp only [← c74808, ← c74809, k2, ← c74810] at e_1215_1
  have hr := e_1215_1
  simp only [e_1215_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11029)) := by
  have c74811 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74812 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74813 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74814 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74815 := (publicBatchWrapper86_copies2337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74816 := (publicBatchWrapper86_copies2338 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_2
  simp only [← c74811, ← c74812, ← c74813] at e_1215_2
  have e_1215_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_3
  simp only [← c74814, ← c74815, k2, ← c74816] at e_1215_3
  have hr := e_1215_3
  simp only [e_1215_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11030)) := by
  have c74817 := (publicBatchWrapper86_copies2338 a h).2.1
  have c74818 := (publicBatchWrapper86_copies2338 a h).2.2.1
  have c74819 := (publicBatchWrapper86_copies2338 a h).2.2.2.1
  have c74820 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.1
  have c74821 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.1
  have c74822 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_4
  simp only [← c74817, ← c74818, ← c74819] at e_1215_4
  have e_1215_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_5
  simp only [← c74820, ← c74821, k2, ← c74822] at e_1215_5
  have hr := e_1215_5
  simp only [e_1215_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11031)) := by
  have c74823 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.1
  have c74824 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.1
  have c74825 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.1
  have c74826 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74827 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74828 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_6
  simp only [← c74823, ← c74824, ← c74825] at e_1215_6
  have e_1215_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_7
  simp only [← c74826, ← c74827, k2, ← c74828] at e_1215_7
  have hr := e_1215_7
  simp only [e_1215_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11032)) := by
  have c74829 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74830 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74831 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74832 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74833 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74834 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_8
  simp only [← c74829, ← c74830, ← c74831] at e_1215_8
  have e_1215_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_9
  simp only [← c74832, ← c74833, k2, ← c74834] at e_1215_9
  have hr := e_1215_9
  simp only [e_1215_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11033)) := by
  have c74835 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74836 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74837 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74838 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74839 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74840 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_10
  simp only [← c74835, ← c74836, ← c74837] at e_1215_10
  have e_1215_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_11
  simp only [← c74838, ← c74839, k2, ← c74840] at e_1215_11
  have hr := e_1215_11
  simp only [e_1215_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11034)) := by
  have c74841 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74842 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74843 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74844 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74845 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74846 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_12
  simp only [← c74841, ← c74842, ← c74843] at e_1215_12
  have e_1215_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_13
  simp only [← c74844, ← c74845, k2, ← c74846] at e_1215_13
  have hr := e_1215_13
  simp only [e_1215_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11035)) := by
  have c74847 := (publicBatchWrapper86_copies2338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74848 := (publicBatchWrapper86_copies2339 a h).1
  have c74849 := (publicBatchWrapper86_copies2339 a h).2.1
  have c74850 := (publicBatchWrapper86_copies2339 a h).2.2.1
  have c74851 := (publicBatchWrapper86_copies2339 a h).2.2.2.1
  have c74852 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_14
  simp only [← c74847, ← c74848, ← c74849] at e_1215_14
  have e_1215_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_15
  simp only [← c74850, ← c74851, k2, ← c74852] at e_1215_15
  have hr := e_1215_15
  simp only [e_1215_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11036)) := by
  have c74853 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.1
  have c74854 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.1
  have c74855 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.1
  have c74856 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.1
  have c74857 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.1
  have c74858 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_16
  simp only [← c74853, ← c74854, ← c74855] at e_1215_16
  have e_1215_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_17
  simp only [← c74856, ← c74857, k2, ← c74858] at e_1215_17
  have hr := e_1215_17
  simp only [e_1215_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1215 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11037)) := by
  have c74859 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74860 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74861 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74862 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74863 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74864 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1215_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_18
  simp only [← c74859, ← c74860, ← c74861] at e_1215_18
  have e_1215_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1215 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1215_19
  simp only [← c74862, ← c74863, k2, ← c74864] at e_1215_19
  have hr := e_1215_19
  simp only [e_1215_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11038)) := by
  have c74865 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74866 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74867 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74868 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74869 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74870 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_0
  simp only [← c74865, ← c74866, ← c74867] at e_1216_0
  have e_1216_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_1
  simp only [← c74868, ← c74869, k2, ← c74870] at e_1216_1
  have hr := e_1216_1
  simp only [e_1216_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11039)) := by
  have c74871 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74872 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74873 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74874 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74875 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74876 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_2
  simp only [← c74871, ← c74872, ← c74873] at e_1216_2
  have e_1216_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_3
  simp only [← c74874, ← c74875, k2, ← c74876] at e_1216_3
  have hr := e_1216_3
  simp only [e_1216_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11040)) := by
  have c74877 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74878 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74879 := (publicBatchWrapper86_copies2339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74880 := (publicBatchWrapper86_copies2340 a h).1
  have c74881 := (publicBatchWrapper86_copies2340 a h).2.1
  have c74882 := (publicBatchWrapper86_copies2340 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_4
  simp only [← c74877, ← c74878, ← c74879] at e_1216_4
  have e_1216_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_5
  simp only [← c74880, ← c74881, k2, ← c74882] at e_1216_5
  have hr := e_1216_5
  simp only [e_1216_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11041)) := by
  have c74883 := (publicBatchWrapper86_copies2340 a h).2.2.2.1
  have c74884 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.1
  have c74885 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.1
  have c74886 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.1
  have c74887 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.1
  have c74888 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_6
  simp only [← c74883, ← c74884, ← c74885] at e_1216_6
  have e_1216_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_7
  simp only [← c74886, ← c74887, k2, ← c74888] at e_1216_7
  have hr := e_1216_7
  simp only [e_1216_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11042)) := by
  have c74889 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.1
  have c74890 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74891 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74892 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74893 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74894 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_8
  simp only [← c74889, ← c74890, ← c74891] at e_1216_8
  have e_1216_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_9
  simp only [← c74892, ← c74893, k2, ← c74894] at e_1216_9
  have hr := e_1216_9
  simp only [e_1216_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11043)) := by
  have c74895 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74896 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74897 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74898 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74899 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74900 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_10
  simp only [← c74895, ← c74896, ← c74897] at e_1216_10
  have e_1216_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_11
  simp only [← c74898, ← c74899, k2, ← c74900] at e_1216_11
  have hr := e_1216_11
  simp only [e_1216_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11044)) := by
  have c74901 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74902 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74903 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74904 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74905 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74906 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_12
  simp only [← c74901, ← c74902, ← c74903] at e_1216_12
  have e_1216_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_13
  simp only [← c74904, ← c74905, k2, ← c74906] at e_1216_13
  have hr := e_1216_13
  simp only [e_1216_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11045)) := by
  have c74907 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74908 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74909 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74910 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74911 := (publicBatchWrapper86_copies2340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74912 := (publicBatchWrapper86_copies2341 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_14
  simp only [← c74907, ← c74908, ← c74909] at e_1216_14
  have e_1216_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_15
  simp only [← c74910, ← c74911, k2, ← c74912] at e_1216_15
  have hr := e_1216_15
  simp only [e_1216_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11046)) := by
  have c74913 := (publicBatchWrapper86_copies2341 a h).2.1
  have c74914 := (publicBatchWrapper86_copies2341 a h).2.2.1
  have c74915 := (publicBatchWrapper86_copies2341 a h).2.2.2.1
  have c74916 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.1
  have c74917 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.1
  have c74918 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_16
  simp only [← c74913, ← c74914, ← c74915] at e_1216_16
  have e_1216_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_17
  simp only [← c74916, ← c74917, k2, ← c74918] at e_1216_17
  have hr := e_1216_17
  simp only [e_1216_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1216 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11047)) := by
  have c74919 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.1
  have c74920 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.1
  have c74921 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.1
  have c74922 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74923 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74924 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1216_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_18
  simp only [← c74919, ← c74920, ← c74921] at e_1216_18
  have e_1216_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1216 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1216_19
  simp only [← c74922, ← c74923, k2, ← c74924] at e_1216_19
  have hr := e_1216_19
  simp only [e_1216_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11048)) := by
  have c74925 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74926 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74927 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74928 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74929 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74930 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_0
  simp only [← c74925, ← c74926, ← c74927] at e_1217_0
  have e_1217_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_1
  simp only [← c74928, ← c74929, k2, ← c74930] at e_1217_1
  have hr := e_1217_1
  simp only [e_1217_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11049)) := by
  have c74931 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74932 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74933 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74934 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74935 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74936 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_2
  simp only [← c74931, ← c74932, ← c74933] at e_1217_2
  have e_1217_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_3
  simp only [← c74934, ← c74935, k2, ← c74936] at e_1217_3
  have hr := e_1217_3
  simp only [e_1217_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11050)) := by
  have c74937 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74938 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74939 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74940 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74941 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74942 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_4
  simp only [← c74937, ← c74938, ← c74939] at e_1217_4
  have e_1217_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_5
  simp only [← c74940, ← c74941, k2, ← c74942] at e_1217_5
  have hr := e_1217_5
  simp only [e_1217_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11051)) := by
  have c74943 := (publicBatchWrapper86_copies2341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74944 := (publicBatchWrapper86_copies2342 a h).1
  have c74945 := (publicBatchWrapper86_copies2342 a h).2.1
  have c74946 := (publicBatchWrapper86_copies2342 a h).2.2.1
  have c74947 := (publicBatchWrapper86_copies2342 a h).2.2.2.1
  have c74948 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_6
  simp only [← c74943, ← c74944, ← c74945] at e_1217_6
  have e_1217_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_7
  simp only [← c74946, ← c74947, k2, ← c74948] at e_1217_7
  have hr := e_1217_7
  simp only [e_1217_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11052)) := by
  have c74949 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.1
  have c74950 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.1
  have c74951 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.1
  have c74952 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.1
  have c74953 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.1
  have c74954 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_8
  simp only [← c74949, ← c74950, ← c74951] at e_1217_8
  have e_1217_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_9
  simp only [← c74952, ← c74953, k2, ← c74954] at e_1217_9
  have hr := e_1217_9
  simp only [e_1217_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11053)) := by
  have c74955 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74956 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74957 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74958 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74959 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74960 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_10
  simp only [← c74955, ← c74956, ← c74957] at e_1217_10
  have e_1217_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_11
  simp only [← c74958, ← c74959, k2, ← c74960] at e_1217_11
  have hr := e_1217_11
  simp only [e_1217_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11054)) := by
  have c74961 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74962 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74963 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74964 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74965 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74966 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_12
  simp only [← c74961, ← c74962, ← c74963] at e_1217_12
  have e_1217_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_13
  simp only [← c74964, ← c74965, k2, ← c74966] at e_1217_13
  have hr := e_1217_13
  simp only [e_1217_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11055)) := by
  have c74967 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74968 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74969 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74970 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74971 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74972 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_14
  simp only [← c74967, ← c74968, ← c74969] at e_1217_14
  have e_1217_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_15
  simp only [← c74970, ← c74971, k2, ← c74972] at e_1217_15
  have hr := e_1217_15
  simp only [e_1217_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11056)) := by
  have c74973 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74974 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74975 := (publicBatchWrapper86_copies2342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c74976 := (publicBatchWrapper86_copies2343 a h).1
  have c74977 := (publicBatchWrapper86_copies2343 a h).2.1
  have c74978 := (publicBatchWrapper86_copies2343 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_16
  simp only [← c74973, ← c74974, ← c74975] at e_1217_16
  have e_1217_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_17
  simp only [← c74976, ← c74977, k2, ← c74978] at e_1217_17
  have hr := e_1217_17
  simp only [e_1217_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1217 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11057)) := by
  have c74979 := (publicBatchWrapper86_copies2343 a h).2.2.2.1
  have c74980 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.1
  have c74981 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.1
  have c74982 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.1
  have c74983 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.1
  have c74984 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1217_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_18
  simp only [← c74979, ← c74980, ← c74981] at e_1217_18
  have e_1217_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1217 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1217_19
  simp only [← c74982, ← c74983, k2, ← c74984] at e_1217_19
  have hr := e_1217_19
  simp only [e_1217_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11058)) := by
  have c74985 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.1
  have c74986 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c74987 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c74988 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74989 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74990 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_0
  simp only [← c74985, ← c74986, ← c74987] at e_1218_0
  have e_1218_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_1
  simp only [← c74988, ← c74989, k2, ← c74990] at e_1218_1
  have hr := e_1218_1
  simp only [e_1218_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11059)) := by
  have c74991 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74992 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74993 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74994 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74995 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74996 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_2
  simp only [← c74991, ← c74992, ← c74993] at e_1218_2
  have e_1218_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_3
  simp only [← c74994, ← c74995, k2, ← c74996] at e_1218_3
  have hr := e_1218_3
  simp only [e_1218_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11060)) := by
  have c74997 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74998 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c74999 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75000 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75001 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75002 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_4
  simp only [← c74997, ← c74998, ← c74999] at e_1218_4
  have e_1218_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_5
  simp only [← c75000, ← c75001, k2, ← c75002] at e_1218_5
  have hr := e_1218_5
  simp only [e_1218_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11061)) := by
  have c75003 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75004 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75005 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75006 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75007 := (publicBatchWrapper86_copies2343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75008 := (publicBatchWrapper86_copies2344 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_6
  simp only [← c75003, ← c75004, ← c75005] at e_1218_6
  have e_1218_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_7
  simp only [← c75006, ← c75007, k2, ← c75008] at e_1218_7
  have hr := e_1218_7
  simp only [e_1218_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11062)) := by
  have c75009 := (publicBatchWrapper86_copies2344 a h).2.1
  have c75010 := (publicBatchWrapper86_copies2344 a h).2.2.1
  have c75011 := (publicBatchWrapper86_copies2344 a h).2.2.2.1
  have c75012 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.1
  have c75013 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.1
  have c75014 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_8
  simp only [← c75009, ← c75010, ← c75011] at e_1218_8
  have e_1218_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_9
  simp only [← c75012, ← c75013, k2, ← c75014] at e_1218_9
  have hr := e_1218_9
  simp only [e_1218_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11063)) := by
  have c75015 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.1
  have c75016 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.1
  have c75017 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.1
  have c75018 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75019 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75020 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_10
  simp only [← c75015, ← c75016, ← c75017] at e_1218_10
  have e_1218_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_11
  simp only [← c75018, ← c75019, k2, ← c75020] at e_1218_11
  have hr := e_1218_11
  simp only [e_1218_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11064)) := by
  have c75021 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75022 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75023 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75024 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75025 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75026 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_12
  simp only [← c75021, ← c75022, ← c75023] at e_1218_12
  have e_1218_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_13
  simp only [← c75024, ← c75025, k2, ← c75026] at e_1218_13
  have hr := e_1218_13
  simp only [e_1218_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11065)) := by
  have c75027 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75028 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75029 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75030 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75031 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75032 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_14
  simp only [← c75027, ← c75028, ← c75029] at e_1218_14
  have e_1218_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_15
  simp only [← c75030, ← c75031, k2, ← c75032] at e_1218_15
  have hr := e_1218_15
  simp only [e_1218_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11066)) := by
  have c75033 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75034 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75035 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75036 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75037 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75038 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_16
  simp only [← c75033, ← c75034, ← c75035] at e_1218_16
  have e_1218_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_17
  simp only [← c75036, ← c75037, k2, ← c75038] at e_1218_17
  have hr := e_1218_17
  simp only [e_1218_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1218 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11067)) := by
  have c75039 := (publicBatchWrapper86_copies2344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75040 := (publicBatchWrapper86_copies2345 a h).1
  have c75041 := (publicBatchWrapper86_copies2345 a h).2.1
  have c75042 := (publicBatchWrapper86_copies2345 a h).2.2.1
  have c75043 := (publicBatchWrapper86_copies2345 a h).2.2.2.1
  have c75044 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1218_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_18
  simp only [← c75039, ← c75040, ← c75041] at e_1218_18
  have e_1218_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1218 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1218_19
  simp only [← c75042, ← c75043, k2, ← c75044] at e_1218_19
  have hr := e_1218_19
  simp only [e_1218_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11068)) := by
  have c75045 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.1
  have c75046 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.1
  have c75047 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.1
  have c75048 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.1
  have c75049 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.1
  have c75050 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_0
  simp only [← c75045, ← c75046, ← c75047] at e_1219_0
  have e_1219_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_1
  simp only [← c75048, ← c75049, k2, ← c75050] at e_1219_1
  have hr := e_1219_1
  simp only [e_1219_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11069)) := by
  have c75051 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75052 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75053 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75054 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75055 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75056 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_2
  simp only [← c75051, ← c75052, ← c75053] at e_1219_2
  have e_1219_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_3
  simp only [← c75054, ← c75055, k2, ← c75056] at e_1219_3
  have hr := e_1219_3
  simp only [e_1219_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11070)) := by
  have c75057 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75058 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75059 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75060 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75061 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75062 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_4
  simp only [← c75057, ← c75058, ← c75059] at e_1219_4
  have e_1219_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_5
  simp only [← c75060, ← c75061, k2, ← c75062] at e_1219_5
  have hr := e_1219_5
  simp only [e_1219_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11071)) := by
  have c75063 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75064 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75065 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75066 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75067 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75068 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_6
  simp only [← c75063, ← c75064, ← c75065] at e_1219_6
  have e_1219_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_7
  simp only [← c75066, ← c75067, k2, ← c75068] at e_1219_7
  have hr := e_1219_7
  simp only [e_1219_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11072)) := by
  have c75069 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75070 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75071 := (publicBatchWrapper86_copies2345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75072 := (publicBatchWrapper86_copies2346 a h).1
  have c75073 := (publicBatchWrapper86_copies2346 a h).2.1
  have c75074 := (publicBatchWrapper86_copies2346 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_8
  simp only [← c75069, ← c75070, ← c75071] at e_1219_8
  have e_1219_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_9
  simp only [← c75072, ← c75073, k2, ← c75074] at e_1219_9
  have hr := e_1219_9
  simp only [e_1219_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11073)) := by
  have c75075 := (publicBatchWrapper86_copies2346 a h).2.2.2.1
  have c75076 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.1
  have c75077 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.1
  have c75078 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.1
  have c75079 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.1
  have c75080 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_10
  simp only [← c75075, ← c75076, ← c75077] at e_1219_10
  have e_1219_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_11
  simp only [← c75078, ← c75079, k2, ← c75080] at e_1219_11
  have hr := e_1219_11
  simp only [e_1219_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11074)) := by
  have c75081 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.1
  have c75082 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75083 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75084 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75085 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75086 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_12
  simp only [← c75081, ← c75082, ← c75083] at e_1219_12
  have e_1219_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_13
  simp only [← c75084, ← c75085, k2, ← c75086] at e_1219_13
  have hr := e_1219_13
  simp only [e_1219_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11075)) := by
  have c75087 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75088 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75089 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75090 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75091 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75092 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_14
  simp only [← c75087, ← c75088, ← c75089] at e_1219_14
  have e_1219_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_15
  simp only [← c75090, ← c75091, k2, ← c75092] at e_1219_15
  have hr := e_1219_15
  simp only [e_1219_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11076)) := by
  have c75093 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75094 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75095 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75096 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75097 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75098 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_16
  simp only [← c75093, ← c75094, ← c75095] at e_1219_16
  have e_1219_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_17
  simp only [← c75096, ← c75097, k2, ← c75098] at e_1219_17
  have hr := e_1219_17
  simp only [e_1219_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1219 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11077)) := by
  have c75099 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75100 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75101 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75102 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75103 := (publicBatchWrapper86_copies2346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75104 := (publicBatchWrapper86_copies2347 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1219_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_18
  simp only [← c75099, ← c75100, ← c75101] at e_1219_18
  have e_1219_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1219 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1219_19
  simp only [← c75102, ← c75103, k2, ← c75104] at e_1219_19
  have hr := e_1219_19
  simp only [e_1219_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11078)) := by
  have c75105 := (publicBatchWrapper86_copies2347 a h).2.1
  have c75106 := (publicBatchWrapper86_copies2347 a h).2.2.1
  have c75107 := (publicBatchWrapper86_copies2347 a h).2.2.2.1
  have c75108 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.1
  have c75109 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.1
  have c75110 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_0
  simp only [← c75105, ← c75106, ← c75107] at e_1220_0
  have e_1220_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_1
  simp only [← c75108, ← c75109, k2, ← c75110] at e_1220_1
  have hr := e_1220_1
  simp only [e_1220_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11079)) := by
  have c75111 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.1
  have c75112 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.1
  have c75113 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.1
  have c75114 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75115 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75116 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_2
  simp only [← c75111, ← c75112, ← c75113] at e_1220_2
  have e_1220_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_3
  simp only [← c75114, ← c75115, k2, ← c75116] at e_1220_3
  have hr := e_1220_3
  simp only [e_1220_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11080)) := by
  have c75117 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75118 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75119 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75120 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75121 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75122 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_4
  simp only [← c75117, ← c75118, ← c75119] at e_1220_4
  have e_1220_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_5
  simp only [← c75120, ← c75121, k2, ← c75122] at e_1220_5
  have hr := e_1220_5
  simp only [e_1220_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11081)) := by
  have c75123 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75124 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75125 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75126 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75127 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75128 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_6
  simp only [← c75123, ← c75124, ← c75125] at e_1220_6
  have e_1220_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_7
  simp only [← c75126, ← c75127, k2, ← c75128] at e_1220_7
  have hr := e_1220_7
  simp only [e_1220_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11082)) := by
  have c75129 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75130 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75131 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75132 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75133 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75134 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_8
  simp only [← c75129, ← c75130, ← c75131] at e_1220_8
  have e_1220_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_9
  simp only [← c75132, ← c75133, k2, ← c75134] at e_1220_9
  have hr := e_1220_9
  simp only [e_1220_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11083)) := by
  have c75135 := (publicBatchWrapper86_copies2347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75136 := (publicBatchWrapper86_copies2348 a h).1
  have c75137 := (publicBatchWrapper86_copies2348 a h).2.1
  have c75138 := (publicBatchWrapper86_copies2348 a h).2.2.1
  have c75139 := (publicBatchWrapper86_copies2348 a h).2.2.2.1
  have c75140 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_10
  simp only [← c75135, ← c75136, ← c75137] at e_1220_10
  have e_1220_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_11
  simp only [← c75138, ← c75139, k2, ← c75140] at e_1220_11
  have hr := e_1220_11
  simp only [e_1220_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11084)) := by
  have c75141 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.1
  have c75142 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.1
  have c75143 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.1
  have c75144 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.1
  have c75145 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.1
  have c75146 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_12
  simp only [← c75141, ← c75142, ← c75143] at e_1220_12
  have e_1220_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_13
  simp only [← c75144, ← c75145, k2, ← c75146] at e_1220_13
  have hr := e_1220_13
  simp only [e_1220_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11085)) := by
  have c75147 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75148 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75149 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75150 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75151 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75152 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_14
  simp only [← c75147, ← c75148, ← c75149] at e_1220_14
  have e_1220_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_15
  simp only [← c75150, ← c75151, k2, ← c75152] at e_1220_15
  have hr := e_1220_15
  simp only [e_1220_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11086)) := by
  have c75153 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75154 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75155 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75156 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75157 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75158 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_16
  simp only [← c75153, ← c75154, ← c75155] at e_1220_16
  have e_1220_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_17
  simp only [← c75156, ← c75157, k2, ← c75158] at e_1220_17
  have hr := e_1220_17
  simp only [e_1220_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1220 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11087)) := by
  have c75159 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75160 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75161 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75162 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75163 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75164 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1220_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_18
  simp only [← c75159, ← c75160, ← c75161] at e_1220_18
  have e_1220_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1220 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1220_19
  simp only [← c75162, ← c75163, k2, ← c75164] at e_1220_19
  have hr := e_1220_19
  simp only [e_1220_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11088)) := by
  have c75165 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75166 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75167 := (publicBatchWrapper86_copies2348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75168 := (publicBatchWrapper86_copies2349 a h).1
  have c75169 := (publicBatchWrapper86_copies2349 a h).2.1
  have c75170 := (publicBatchWrapper86_copies2349 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_0
  simp only [← c75165, ← c75166, ← c75167] at e_1221_0
  have e_1221_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_1
  simp only [← c75168, ← c75169, k2, ← c75170] at e_1221_1
  have hr := e_1221_1
  simp only [e_1221_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11089)) := by
  have c75171 := (publicBatchWrapper86_copies2349 a h).2.2.2.1
  have c75172 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.1
  have c75173 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.1
  have c75174 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.1
  have c75175 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.1
  have c75176 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_2
  simp only [← c75171, ← c75172, ← c75173] at e_1221_2
  have e_1221_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_3
  simp only [← c75174, ← c75175, k2, ← c75176] at e_1221_3
  have hr := e_1221_3
  simp only [e_1221_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11090)) := by
  have c75177 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.1
  have c75178 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75179 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75180 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75181 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75182 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_4
  simp only [← c75177, ← c75178, ← c75179] at e_1221_4
  have e_1221_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_5
  simp only [← c75180, ← c75181, k2, ← c75182] at e_1221_5
  have hr := e_1221_5
  simp only [e_1221_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11091)) := by
  have c75183 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75184 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75185 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75186 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75187 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75188 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_6
  simp only [← c75183, ← c75184, ← c75185] at e_1221_6
  have e_1221_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_7
  simp only [← c75186, ← c75187, k2, ← c75188] at e_1221_7
  have hr := e_1221_7
  simp only [e_1221_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11092)) := by
  have c75189 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75190 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75191 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75192 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75193 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75194 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_8
  simp only [← c75189, ← c75190, ← c75191] at e_1221_8
  have e_1221_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_9
  simp only [← c75192, ← c75193, k2, ← c75194] at e_1221_9
  have hr := e_1221_9
  simp only [e_1221_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11093)) := by
  have c75195 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75196 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75197 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75198 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75199 := (publicBatchWrapper86_copies2349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75200 := (publicBatchWrapper86_copies2350 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_10
  simp only [← c75195, ← c75196, ← c75197] at e_1221_10
  have e_1221_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_11
  simp only [← c75198, ← c75199, k2, ← c75200] at e_1221_11
  have hr := e_1221_11
  simp only [e_1221_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11094)) := by
  have c75201 := (publicBatchWrapper86_copies2350 a h).2.1
  have c75202 := (publicBatchWrapper86_copies2350 a h).2.2.1
  have c75203 := (publicBatchWrapper86_copies2350 a h).2.2.2.1
  have c75204 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.1
  have c75205 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.1
  have c75206 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_12
  simp only [← c75201, ← c75202, ← c75203] at e_1221_12
  have e_1221_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_13
  simp only [← c75204, ← c75205, k2, ← c75206] at e_1221_13
  have hr := e_1221_13
  simp only [e_1221_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11095)) := by
  have c75207 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.1
  have c75208 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.1
  have c75209 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.1
  have c75210 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75211 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75212 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_14
  simp only [← c75207, ← c75208, ← c75209] at e_1221_14
  have e_1221_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_15
  simp only [← c75210, ← c75211, k2, ← c75212] at e_1221_15
  have hr := e_1221_15
  simp only [e_1221_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11096)) := by
  have c75213 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75214 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75215 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75216 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75217 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75218 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_16
  simp only [← c75213, ← c75214, ← c75215] at e_1221_16
  have e_1221_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_17
  simp only [← c75216, ← c75217, k2, ← c75218] at e_1221_17
  have hr := e_1221_17
  simp only [e_1221_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1221 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11097)) := by
  have c75219 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75220 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75221 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75222 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75223 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75224 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1221_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_18
  simp only [← c75219, ← c75220, ← c75221] at e_1221_18
  have e_1221_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1221 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1221_19
  simp only [← c75222, ← c75223, k2, ← c75224] at e_1221_19
  have hr := e_1221_19
  simp only [e_1221_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11098)) := by
  have c75225 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75226 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75227 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75228 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75229 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75230 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_0
  simp only [← c75225, ← c75226, ← c75227] at e_1222_0
  have e_1222_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_1
  simp only [← c75228, ← c75229, k2, ← c75230] at e_1222_1
  have hr := e_1222_1
  simp only [e_1222_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11099)) := by
  have c75231 := (publicBatchWrapper86_copies2350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75232 := (publicBatchWrapper86_copies2351 a h).1
  have c75233 := (publicBatchWrapper86_copies2351 a h).2.1
  have c75234 := (publicBatchWrapper86_copies2351 a h).2.2.1
  have c75235 := (publicBatchWrapper86_copies2351 a h).2.2.2.1
  have c75236 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_2
  simp only [← c75231, ← c75232, ← c75233] at e_1222_2
  have e_1222_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_3
  simp only [← c75234, ← c75235, k2, ← c75236] at e_1222_3
  have hr := e_1222_3
  simp only [e_1222_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11100)) := by
  have c75237 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.1
  have c75238 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.1
  have c75239 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.1
  have c75240 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.1
  have c75241 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.1
  have c75242 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_4
  simp only [← c75237, ← c75238, ← c75239] at e_1222_4
  have e_1222_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_5
  simp only [← c75240, ← c75241, k2, ← c75242] at e_1222_5
  have hr := e_1222_5
  simp only [e_1222_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11101)) := by
  have c75243 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75244 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75245 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75246 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75247 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75248 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_6
  simp only [← c75243, ← c75244, ← c75245] at e_1222_6
  have e_1222_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_7
  simp only [← c75246, ← c75247, k2, ← c75248] at e_1222_7
  have hr := e_1222_7
  simp only [e_1222_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11102)) := by
  have c75249 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75250 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75251 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75252 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75253 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75254 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_8
  simp only [← c75249, ← c75250, ← c75251] at e_1222_8
  have e_1222_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_9
  simp only [← c75252, ← c75253, k2, ← c75254] at e_1222_9
  have hr := e_1222_9
  simp only [e_1222_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11103)) := by
  have c75255 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75256 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75257 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75258 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75259 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75260 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_10
  simp only [← c75255, ← c75256, ← c75257] at e_1222_10
  have e_1222_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_11
  simp only [← c75258, ← c75259, k2, ← c75260] at e_1222_11
  have hr := e_1222_11
  simp only [e_1222_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11104)) := by
  have c75261 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75262 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75263 := (publicBatchWrapper86_copies2351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75264 := (publicBatchWrapper86_copies2352 a h).1
  have c75265 := (publicBatchWrapper86_copies2352 a h).2.1
  have c75266 := (publicBatchWrapper86_copies2352 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_12
  simp only [← c75261, ← c75262, ← c75263] at e_1222_12
  have e_1222_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_13
  simp only [← c75264, ← c75265, k2, ← c75266] at e_1222_13
  have hr := e_1222_13
  simp only [e_1222_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11105)) := by
  have c75267 := (publicBatchWrapper86_copies2352 a h).2.2.2.1
  have c75268 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.1
  have c75269 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.1
  have c75270 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.1
  have c75271 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.1
  have c75272 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_14
  simp only [← c75267, ← c75268, ← c75269] at e_1222_14
  have e_1222_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_15
  simp only [← c75270, ← c75271, k2, ← c75272] at e_1222_15
  have hr := e_1222_15
  simp only [e_1222_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11106)) := by
  have c75273 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.1
  have c75274 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75275 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75276 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75277 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75278 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_16
  simp only [← c75273, ← c75274, ← c75275] at e_1222_16
  have e_1222_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_17
  simp only [← c75276, ← c75277, k2, ← c75278] at e_1222_17
  have hr := e_1222_17
  simp only [e_1222_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1222 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11107)) := by
  have c75279 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75280 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75281 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75282 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75283 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75284 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1222_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_18
  simp only [← c75279, ← c75280, ← c75281] at e_1222_18
  have e_1222_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1222 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1222_19
  simp only [← c75282, ← c75283, k2, ← c75284] at e_1222_19
  have hr := e_1222_19
  simp only [e_1222_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11108)) := by
  have c75285 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75286 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75287 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75288 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75289 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75290 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_0
  simp only [← c75285, ← c75286, ← c75287] at e_1223_0
  have e_1223_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_1
  simp only [← c75288, ← c75289, k2, ← c75290] at e_1223_1
  have hr := e_1223_1
  simp only [e_1223_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11109)) := by
  have c75291 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75292 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75293 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75294 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75295 := (publicBatchWrapper86_copies2352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75296 := (publicBatchWrapper86_copies2353 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_2
  simp only [← c75291, ← c75292, ← c75293] at e_1223_2
  have e_1223_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_3
  simp only [← c75294, ← c75295, k2, ← c75296] at e_1223_3
  have hr := e_1223_3
  simp only [e_1223_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11110)) := by
  have c75297 := (publicBatchWrapper86_copies2353 a h).2.1
  have c75298 := (publicBatchWrapper86_copies2353 a h).2.2.1
  have c75299 := (publicBatchWrapper86_copies2353 a h).2.2.2.1
  have c75300 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.1
  have c75301 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.1
  have c75302 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_4
  simp only [← c75297, ← c75298, ← c75299] at e_1223_4
  have e_1223_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_5
  simp only [← c75300, ← c75301, k2, ← c75302] at e_1223_5
  have hr := e_1223_5
  simp only [e_1223_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11111)) := by
  have c75303 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.1
  have c75304 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.1
  have c75305 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.1
  have c75306 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75307 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75308 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_6
  simp only [← c75303, ← c75304, ← c75305] at e_1223_6
  have e_1223_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_7
  simp only [← c75306, ← c75307, k2, ← c75308] at e_1223_7
  have hr := e_1223_7
  simp only [e_1223_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11112)) := by
  have c75309 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75310 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75311 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75312 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75313 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75314 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_8
  simp only [← c75309, ← c75310, ← c75311] at e_1223_8
  have e_1223_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_9
  simp only [← c75312, ← c75313, k2, ← c75314] at e_1223_9
  have hr := e_1223_9
  simp only [e_1223_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11113)) := by
  have c75315 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75316 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75317 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75318 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75319 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75320 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_10
  simp only [← c75315, ← c75316, ← c75317] at e_1223_10
  have e_1223_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_11
  simp only [← c75318, ← c75319, k2, ← c75320] at e_1223_11
  have hr := e_1223_11
  simp only [e_1223_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11114)) := by
  have c75321 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75322 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75323 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75324 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75325 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75326 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_12
  simp only [← c75321, ← c75322, ← c75323] at e_1223_12
  have e_1223_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_13
  simp only [← c75324, ← c75325, k2, ← c75326] at e_1223_13
  have hr := e_1223_13
  simp only [e_1223_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11115)) := by
  have c75327 := (publicBatchWrapper86_copies2353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75328 := (publicBatchWrapper86_copies2354 a h).1
  have c75329 := (publicBatchWrapper86_copies2354 a h).2.1
  have c75330 := (publicBatchWrapper86_copies2354 a h).2.2.1
  have c75331 := (publicBatchWrapper86_copies2354 a h).2.2.2.1
  have c75332 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_14
  simp only [← c75327, ← c75328, ← c75329] at e_1223_14
  have e_1223_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_15
  simp only [← c75330, ← c75331, k2, ← c75332] at e_1223_15
  have hr := e_1223_15
  simp only [e_1223_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11116)) := by
  have c75333 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.1
  have c75334 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.1
  have c75335 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.1
  have c75336 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.1
  have c75337 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.1
  have c75338 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_16
  simp only [← c75333, ← c75334, ← c75335] at e_1223_16
  have e_1223_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_17
  simp only [← c75336, ← c75337, k2, ← c75338] at e_1223_17
  have hr := e_1223_17
  simp only [e_1223_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1223 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11117)) := by
  have c75339 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75340 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75341 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75342 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75343 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75344 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1223_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_18
  simp only [← c75339, ← c75340, ← c75341] at e_1223_18
  have e_1223_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1223 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1223_19
  simp only [← c75342, ← c75343, k2, ← c75344] at e_1223_19
  have hr := e_1223_19
  simp only [e_1223_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11118)) := by
  have c75345 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75346 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75347 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75348 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75349 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75350 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_0
  simp only [← c75345, ← c75346, ← c75347] at e_1224_0
  have e_1224_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_1
  simp only [← c75348, ← c75349, k2, ← c75350] at e_1224_1
  have hr := e_1224_1
  simp only [e_1224_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11119)) := by
  have c75351 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75352 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75353 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75354 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75355 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75356 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_2
  simp only [← c75351, ← c75352, ← c75353] at e_1224_2
  have e_1224_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_3
  simp only [← c75354, ← c75355, k2, ← c75356] at e_1224_3
  have hr := e_1224_3
  simp only [e_1224_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11120)) := by
  have c75357 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75358 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75359 := (publicBatchWrapper86_copies2354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75360 := (publicBatchWrapper86_copies2355 a h).1
  have c75361 := (publicBatchWrapper86_copies2355 a h).2.1
  have c75362 := (publicBatchWrapper86_copies2355 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_4
  simp only [← c75357, ← c75358, ← c75359] at e_1224_4
  have e_1224_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_5
  simp only [← c75360, ← c75361, k2, ← c75362] at e_1224_5
  have hr := e_1224_5
  simp only [e_1224_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11121)) := by
  have c75363 := (publicBatchWrapper86_copies2355 a h).2.2.2.1
  have c75364 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.1
  have c75365 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.1
  have c75366 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.1
  have c75367 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.1
  have c75368 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_6
  simp only [← c75363, ← c75364, ← c75365] at e_1224_6
  have e_1224_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_7
  simp only [← c75366, ← c75367, k2, ← c75368] at e_1224_7
  have hr := e_1224_7
  simp only [e_1224_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11122)) := by
  have c75369 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.1
  have c75370 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75371 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75372 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75373 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75374 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_8
  simp only [← c75369, ← c75370, ← c75371] at e_1224_8
  have e_1224_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_9
  simp only [← c75372, ← c75373, k2, ← c75374] at e_1224_9
  have hr := e_1224_9
  simp only [e_1224_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11123)) := by
  have c75375 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75376 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75377 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75378 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75379 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75380 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_10
  simp only [← c75375, ← c75376, ← c75377] at e_1224_10
  have e_1224_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_11
  simp only [← c75378, ← c75379, k2, ← c75380] at e_1224_11
  have hr := e_1224_11
  simp only [e_1224_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11124)) := by
  have c75381 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75382 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75383 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75384 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75385 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75386 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_12
  simp only [← c75381, ← c75382, ← c75383] at e_1224_12
  have e_1224_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_13
  simp only [← c75384, ← c75385, k2, ← c75386] at e_1224_13
  have hr := e_1224_13
  simp only [e_1224_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11125)) := by
  have c75387 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75388 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75389 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75390 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75391 := (publicBatchWrapper86_copies2355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75392 := (publicBatchWrapper86_copies2356 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_14
  simp only [← c75387, ← c75388, ← c75389] at e_1224_14
  have e_1224_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_15
  simp only [← c75390, ← c75391, k2, ← c75392] at e_1224_15
  have hr := e_1224_15
  simp only [e_1224_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11126)) := by
  have c75393 := (publicBatchWrapper86_copies2356 a h).2.1
  have c75394 := (publicBatchWrapper86_copies2356 a h).2.2.1
  have c75395 := (publicBatchWrapper86_copies2356 a h).2.2.2.1
  have c75396 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.1
  have c75397 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.1
  have c75398 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_16
  simp only [← c75393, ← c75394, ← c75395] at e_1224_16
  have e_1224_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_17
  simp only [← c75396, ← c75397, k2, ← c75398] at e_1224_17
  have hr := e_1224_17
  simp only [e_1224_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1224 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11127)) := by
  have c75399 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.1
  have c75400 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.1
  have c75401 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.1
  have c75402 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75403 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75404 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1224_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_18
  simp only [← c75399, ← c75400, ← c75401] at e_1224_18
  have e_1224_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1224 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1224_19
  simp only [← c75402, ← c75403, k2, ← c75404] at e_1224_19
  have hr := e_1224_19
  simp only [e_1224_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11128)) := by
  have c75405 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75406 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75407 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75408 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75409 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75410 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_0
  simp only [← c75405, ← c75406, ← c75407] at e_1225_0
  have e_1225_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_1
  simp only [← c75408, ← c75409, k2, ← c75410] at e_1225_1
  have hr := e_1225_1
  simp only [e_1225_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11129)) := by
  have c75411 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75412 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75413 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75414 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75415 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75416 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_2
  simp only [← c75411, ← c75412, ← c75413] at e_1225_2
  have e_1225_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_3
  simp only [← c75414, ← c75415, k2, ← c75416] at e_1225_3
  have hr := e_1225_3
  simp only [e_1225_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11130)) := by
  have c75417 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75418 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75419 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75420 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75421 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75422 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_4
  simp only [← c75417, ← c75418, ← c75419] at e_1225_4
  have e_1225_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_5
  simp only [← c75420, ← c75421, k2, ← c75422] at e_1225_5
  have hr := e_1225_5
  simp only [e_1225_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11131)) := by
  have c75423 := (publicBatchWrapper86_copies2356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75424 := (publicBatchWrapper86_copies2357 a h).1
  have c75425 := (publicBatchWrapper86_copies2357 a h).2.1
  have c75426 := (publicBatchWrapper86_copies2357 a h).2.2.1
  have c75427 := (publicBatchWrapper86_copies2357 a h).2.2.2.1
  have c75428 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_6
  simp only [← c75423, ← c75424, ← c75425] at e_1225_6
  have e_1225_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_7
  simp only [← c75426, ← c75427, k2, ← c75428] at e_1225_7
  have hr := e_1225_7
  simp only [e_1225_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11132)) := by
  have c75429 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.1
  have c75430 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.1
  have c75431 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.1
  have c75432 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.1
  have c75433 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.1
  have c75434 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_8
  simp only [← c75429, ← c75430, ← c75431] at e_1225_8
  have e_1225_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_9
  simp only [← c75432, ← c75433, k2, ← c75434] at e_1225_9
  have hr := e_1225_9
  simp only [e_1225_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11133)) := by
  have c75435 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75436 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75437 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75438 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75439 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75440 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_10
  simp only [← c75435, ← c75436, ← c75437] at e_1225_10
  have e_1225_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_11
  simp only [← c75438, ← c75439, k2, ← c75440] at e_1225_11
  have hr := e_1225_11
  simp only [e_1225_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11134)) := by
  have c75441 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75442 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75443 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75444 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75445 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75446 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_12
  simp only [← c75441, ← c75442, ← c75443] at e_1225_12
  have e_1225_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_13
  simp only [← c75444, ← c75445, k2, ← c75446] at e_1225_13
  have hr := e_1225_13
  simp only [e_1225_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11135)) := by
  have c75447 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75448 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75449 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75450 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75451 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75452 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_14
  simp only [← c75447, ← c75448, ← c75449] at e_1225_14
  have e_1225_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_15
  simp only [← c75450, ← c75451, k2, ← c75452] at e_1225_15
  have hr := e_1225_15
  simp only [e_1225_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11136)) := by
  have c75453 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75454 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75455 := (publicBatchWrapper86_copies2357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75456 := (publicBatchWrapper86_copies2358 a h).1
  have c75457 := (publicBatchWrapper86_copies2358 a h).2.1
  have c75458 := (publicBatchWrapper86_copies2358 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_16
  simp only [← c75453, ← c75454, ← c75455] at e_1225_16
  have e_1225_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_17
  simp only [← c75456, ← c75457, k2, ← c75458] at e_1225_17
  have hr := e_1225_17
  simp only [e_1225_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1225 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11137)) := by
  have c75459 := (publicBatchWrapper86_copies2358 a h).2.2.2.1
  have c75460 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.1
  have c75461 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.1
  have c75462 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.1
  have c75463 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.1
  have c75464 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1225_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_18
  simp only [← c75459, ← c75460, ← c75461] at e_1225_18
  have e_1225_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1225 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1225_19
  simp only [← c75462, ← c75463, k2, ← c75464] at e_1225_19
  have hr := e_1225_19
  simp only [e_1225_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11138)) := by
  have c75465 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.1
  have c75466 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75467 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75468 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75469 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75470 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_0
  simp only [← c75465, ← c75466, ← c75467] at e_1226_0
  have e_1226_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_1
  simp only [← c75468, ← c75469, k2, ← c75470] at e_1226_1
  have hr := e_1226_1
  simp only [e_1226_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11139)) := by
  have c75471 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75472 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75473 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75474 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75475 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75476 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_2
  simp only [← c75471, ← c75472, ← c75473] at e_1226_2
  have e_1226_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_3
  simp only [← c75474, ← c75475, k2, ← c75476] at e_1226_3
  have hr := e_1226_3
  simp only [e_1226_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11140)) := by
  have c75477 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75478 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75479 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75480 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75481 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75482 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_4
  simp only [← c75477, ← c75478, ← c75479] at e_1226_4
  have e_1226_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_5
  simp only [← c75480, ← c75481, k2, ← c75482] at e_1226_5
  have hr := e_1226_5
  simp only [e_1226_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11141)) := by
  have c75483 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75484 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75485 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75486 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75487 := (publicBatchWrapper86_copies2358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75488 := (publicBatchWrapper86_copies2359 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_6
  simp only [← c75483, ← c75484, ← c75485] at e_1226_6
  have e_1226_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_7
  simp only [← c75486, ← c75487, k2, ← c75488] at e_1226_7
  have hr := e_1226_7
  simp only [e_1226_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11142)) := by
  have c75489 := (publicBatchWrapper86_copies2359 a h).2.1
  have c75490 := (publicBatchWrapper86_copies2359 a h).2.2.1
  have c75491 := (publicBatchWrapper86_copies2359 a h).2.2.2.1
  have c75492 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.1
  have c75493 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.1
  have c75494 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_8
  simp only [← c75489, ← c75490, ← c75491] at e_1226_8
  have e_1226_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_9
  simp only [← c75492, ← c75493, k2, ← c75494] at e_1226_9
  have hr := e_1226_9
  simp only [e_1226_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11143)) := by
  have c75495 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.1
  have c75496 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.1
  have c75497 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.1
  have c75498 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75499 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75500 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_10
  simp only [← c75495, ← c75496, ← c75497] at e_1226_10
  have e_1226_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_11
  simp only [← c75498, ← c75499, k2, ← c75500] at e_1226_11
  have hr := e_1226_11
  simp only [e_1226_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11144)) := by
  have c75501 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75502 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75503 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75504 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75505 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75506 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_12
  simp only [← c75501, ← c75502, ← c75503] at e_1226_12
  have e_1226_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_13
  simp only [← c75504, ← c75505, k2, ← c75506] at e_1226_13
  have hr := e_1226_13
  simp only [e_1226_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11145)) := by
  have c75507 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75508 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75509 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75510 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75511 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75512 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_14
  simp only [← c75507, ← c75508, ← c75509] at e_1226_14
  have e_1226_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_15
  simp only [← c75510, ← c75511, k2, ← c75512] at e_1226_15
  have hr := e_1226_15
  simp only [e_1226_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11146)) := by
  have c75513 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75514 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75515 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75516 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75517 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75518 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_16
  simp only [← c75513, ← c75514, ← c75515] at e_1226_16
  have e_1226_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_17
  simp only [← c75516, ← c75517, k2, ← c75518] at e_1226_17
  have hr := e_1226_17
  simp only [e_1226_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1226 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11147)) := by
  have c75519 := (publicBatchWrapper86_copies2359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75520 := (publicBatchWrapper86_copies2360 a h).1
  have c75521 := (publicBatchWrapper86_copies2360 a h).2.1
  have c75522 := (publicBatchWrapper86_copies2360 a h).2.2.1
  have c75523 := (publicBatchWrapper86_copies2360 a h).2.2.2.1
  have c75524 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1226_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_18
  simp only [← c75519, ← c75520, ← c75521] at e_1226_18
  have e_1226_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1226 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1226_19
  simp only [← c75522, ← c75523, k2, ← c75524] at e_1226_19
  have hr := e_1226_19
  simp only [e_1226_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11148)) := by
  have c75525 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.1
  have c75526 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.1
  have c75527 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.1
  have c75528 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.1
  have c75529 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.1
  have c75530 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_0
  simp only [← c75525, ← c75526, ← c75527] at e_1227_0
  have e_1227_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_1
  simp only [← c75528, ← c75529, k2, ← c75530] at e_1227_1
  have hr := e_1227_1
  simp only [e_1227_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11149)) := by
  have c75531 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75532 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75533 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75534 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75535 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75536 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_2
  simp only [← c75531, ← c75532, ← c75533] at e_1227_2
  have e_1227_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_3
  simp only [← c75534, ← c75535, k2, ← c75536] at e_1227_3
  have hr := e_1227_3
  simp only [e_1227_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11150)) := by
  have c75537 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75538 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75539 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75540 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75541 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75542 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_4
  simp only [← c75537, ← c75538, ← c75539] at e_1227_4
  have e_1227_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_5
  simp only [← c75540, ← c75541, k2, ← c75542] at e_1227_5
  have hr := e_1227_5
  simp only [e_1227_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11151)) := by
  have c75543 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75544 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75545 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75546 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75547 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75548 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_6
  simp only [← c75543, ← c75544, ← c75545] at e_1227_6
  have e_1227_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_7
  simp only [← c75546, ← c75547, k2, ← c75548] at e_1227_7
  have hr := e_1227_7
  simp only [e_1227_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11152)) := by
  have c75549 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75550 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75551 := (publicBatchWrapper86_copies2360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75552 := (publicBatchWrapper86_copies2361 a h).1
  have c75553 := (publicBatchWrapper86_copies2361 a h).2.1
  have c75554 := (publicBatchWrapper86_copies2361 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_8
  simp only [← c75549, ← c75550, ← c75551] at e_1227_8
  have e_1227_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_9
  simp only [← c75552, ← c75553, k2, ← c75554] at e_1227_9
  have hr := e_1227_9
  simp only [e_1227_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11153)) := by
  have c75555 := (publicBatchWrapper86_copies2361 a h).2.2.2.1
  have c75556 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.1
  have c75557 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.1
  have c75558 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.1
  have c75559 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.1
  have c75560 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_10
  simp only [← c75555, ← c75556, ← c75557] at e_1227_10
  have e_1227_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_11
  simp only [← c75558, ← c75559, k2, ← c75560] at e_1227_11
  have hr := e_1227_11
  simp only [e_1227_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11154)) := by
  have c75561 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.1
  have c75562 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75563 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75564 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75565 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75566 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_12
  simp only [← c75561, ← c75562, ← c75563] at e_1227_12
  have e_1227_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_13
  simp only [← c75564, ← c75565, k2, ← c75566] at e_1227_13
  have hr := e_1227_13
  simp only [e_1227_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11155)) := by
  have c75567 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75568 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75569 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75570 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75571 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75572 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_14
  simp only [← c75567, ← c75568, ← c75569] at e_1227_14
  have e_1227_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_15
  simp only [← c75570, ← c75571, k2, ← c75572] at e_1227_15
  have hr := e_1227_15
  simp only [e_1227_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11156)) := by
  have c75573 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75574 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75575 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75576 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75577 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75578 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_16
  simp only [← c75573, ← c75574, ← c75575] at e_1227_16
  have e_1227_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_17
  simp only [← c75576, ← c75577, k2, ← c75578] at e_1227_17
  have hr := e_1227_17
  simp only [e_1227_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1227 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11157)) := by
  have c75579 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75580 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75581 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75582 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75583 := (publicBatchWrapper86_copies2361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75584 := (publicBatchWrapper86_copies2362 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1227_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_18
  simp only [← c75579, ← c75580, ← c75581] at e_1227_18
  have e_1227_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1227 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1227_19
  simp only [← c75582, ← c75583, k2, ← c75584] at e_1227_19
  have hr := e_1227_19
  simp only [e_1227_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11158)) := by
  have c75585 := (publicBatchWrapper86_copies2362 a h).2.1
  have c75586 := (publicBatchWrapper86_copies2362 a h).2.2.1
  have c75587 := (publicBatchWrapper86_copies2362 a h).2.2.2.1
  have c75588 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.1
  have c75589 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.1
  have c75590 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_0
  simp only [← c75585, ← c75586, ← c75587] at e_1228_0
  have e_1228_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_1
  simp only [← c75588, ← c75589, k2, ← c75590] at e_1228_1
  have hr := e_1228_1
  simp only [e_1228_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11159)) := by
  have c75591 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.1
  have c75592 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.1
  have c75593 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.1
  have c75594 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75595 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75596 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_2
  simp only [← c75591, ← c75592, ← c75593] at e_1228_2
  have e_1228_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_3
  simp only [← c75594, ← c75595, k2, ← c75596] at e_1228_3
  have hr := e_1228_3
  simp only [e_1228_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 23) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11160)) := by
  have c75597 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75598 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75599 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75600 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75601 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75602 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_4
  simp only [← c75597, ← c75598, ← c75599] at e_1228_4
  have e_1228_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_5
  simp only [← c75600, ← c75601, k2, ← c75602] at e_1228_5
  have hr := e_1228_5
  simp only [e_1228_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 31) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11161)) := by
  have c75603 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75604 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75605 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75606 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75607 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75608 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_6
  simp only [← c75603, ← c75604, ← c75605] at e_1228_6
  have e_1228_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_7
  simp only [← c75606, ← c75607, k2, ← c75608] at e_1228_7
  have hr := e_1228_7
  simp only [e_1228_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 39) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11162)) := by
  have c75609 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75610 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75611 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75612 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75613 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75614 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_8
  simp only [← c75609, ← c75610, ← c75611] at e_1228_8
  have e_1228_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_9
  simp only [← c75612, ← c75613, k2, ← c75614] at e_1228_9
  have hr := e_1228_9
  simp only [e_1228_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 47) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11163)) := by
  have c75615 := (publicBatchWrapper86_copies2362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75616 := (publicBatchWrapper86_copies2363 a h).1
  have c75617 := (publicBatchWrapper86_copies2363 a h).2.1
  have c75618 := (publicBatchWrapper86_copies2363 a h).2.2.1
  have c75619 := (publicBatchWrapper86_copies2363 a h).2.2.2.1
  have c75620 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_10
  simp only [← c75615, ← c75616, ← c75617] at e_1228_10
  have e_1228_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_11
  simp only [← c75618, ← c75619, k2, ← c75620] at e_1228_11
  have hr := e_1228_11
  simp only [e_1228_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 55) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11164)) := by
  have c75621 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.1
  have c75622 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.1
  have c75623 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.1
  have c75624 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.1
  have c75625 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.1
  have c75626 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_12
  simp only [← c75621, ← c75622, ← c75623] at e_1228_12
  have e_1228_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_13
  simp only [← c75624, ← c75625, k2, ← c75626] at e_1228_13
  have hr := e_1228_13
  simp only [e_1228_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 63) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11165)) := by
  have c75627 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75628 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75629 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75630 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75631 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75632 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_14
  simp only [← c75627, ← c75628, ← c75629] at e_1228_14
  have e_1228_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_15
  simp only [← c75630, ← c75631, k2, ← c75632] at e_1228_15
  have hr := e_1228_15
  simp only [e_1228_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 71) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11166)) := by
  have c75633 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75634 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75635 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75636 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75637 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75638 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_16
  simp only [← c75633, ← c75634, ← c75635] at e_1228_16
  have e_1228_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_17
  simp only [← c75636, ← c75637, k2, ← c75638] at e_1228_17
  have hr := e_1228_17
  simp only [e_1228_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1228 79) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11167)) := by
  have c75639 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75640 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75641 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75642 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75643 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75644 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1228_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_18
  simp only [← c75639, ← c75640, ← c75641] at e_1228_18
  have e_1228_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1228 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1228_19
  simp only [← c75642, ← c75643, k2, ← c75644] at e_1228_19
  have hr := e_1228_19
  simp only [e_1228_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 7) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11168)) := by
  have c75645 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75646 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75647 := (publicBatchWrapper86_copies2363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75648 := (publicBatchWrapper86_copies2364 a h).1
  have c75649 := (publicBatchWrapper86_copies2364 a h).2.1
  have c75650 := (publicBatchWrapper86_copies2364 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_0
  simp only [← c75645, ← c75646, ← c75647] at e_1229_0
  have e_1229_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_1
  simp only [← c75648, ← c75649, k2, ← c75650] at e_1229_1
  have hr := e_1229_1
  simp only [e_1229_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 15) = bselect (a (.wire 54 47)) (a (.virt 200)) (a (.virt 11169)) := by
  have c75651 := (publicBatchWrapper86_copies2364 a h).2.2.2.1
  have c75652 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.1
  have c75653 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.1
  have c75654 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.1
  have c75655 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.1
  have c75656 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_2
  simp only [← c75651, ← c75652, ← c75653] at e_1229_2
  have e_1229_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_3
  simp only [← c75654, ← c75655, k2, ← c75656] at e_1229_3
  have hr := e_1229_3
  simp only [e_1229_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11210)) := by
  have c75657 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.1
  have c75658 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75659 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75660 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75661 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75662 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_4
  simp only [← c75657, ← c75658, ← c75659] at e_1229_4
  have e_1229_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_5
  simp only [← c75660, ← c75661, k2, ← c75662] at e_1229_5
  have hr := e_1229_5
  simp only [e_1229_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11211)) := by
  have c75663 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75664 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75665 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75666 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75667 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75668 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_6
  simp only [← c75663, ← c75664, ← c75665] at e_1229_6
  have e_1229_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_7
  simp only [← c75666, ← c75667, k2, ← c75668] at e_1229_7
  have hr := e_1229_7
  simp only [e_1229_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11212)) := by
  have c75669 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75670 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75671 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75672 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75673 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75674 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_8
  simp only [← c75669, ← c75670, ← c75671] at e_1229_8
  have e_1229_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_9
  simp only [← c75672, ← c75673, k2, ← c75674] at e_1229_9
  have hr := e_1229_9
  simp only [e_1229_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11213)) := by
  have c75675 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75676 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75677 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75678 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75679 := (publicBatchWrapper86_copies2364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75680 := (publicBatchWrapper86_copies2365 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_10
  simp only [← c75675, ← c75676, ← c75677] at e_1229_10
  have e_1229_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_11
  simp only [← c75678, ← c75679, k2, ← c75680] at e_1229_11
  have hr := e_1229_11
  simp only [e_1229_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11214)) := by
  have c75681 := (publicBatchWrapper86_copies2365 a h).2.1
  have c75682 := (publicBatchWrapper86_copies2365 a h).2.2.1
  have c75683 := (publicBatchWrapper86_copies2365 a h).2.2.2.1
  have c75684 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.1
  have c75685 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.1
  have c75686 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_12
  simp only [← c75681, ← c75682, ← c75683] at e_1229_12
  have e_1229_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_13
  simp only [← c75684, ← c75685, k2, ← c75686] at e_1229_13
  have hr := e_1229_13
  simp only [e_1229_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11215)) := by
  have c75687 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.1
  have c75688 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.1
  have c75689 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.1
  have c75690 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75691 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75692 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_14
  simp only [← c75687, ← c75688, ← c75689] at e_1229_14
  have e_1229_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_15
  simp only [← c75690, ← c75691, k2, ← c75692] at e_1229_15
  have hr := e_1229_15
  simp only [e_1229_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11216)) := by
  have c75693 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75694 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75695 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75696 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75697 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75698 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_16
  simp only [← c75693, ← c75694, ← c75695] at e_1229_16
  have e_1229_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_17
  simp only [← c75696, ← c75697, k2, ← c75698] at e_1229_17
  have hr := e_1229_17
  simp only [e_1229_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1229 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11217)) := by
  have c75699 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75700 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75701 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75702 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75703 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75704 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1229_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_18
  simp only [← c75699, ← c75700, ← c75701] at e_1229_18
  have e_1229_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1229 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1229_19
  simp only [← c75702, ← c75703, k2, ← c75704] at e_1229_19
  have hr := e_1229_19
  simp only [e_1229_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11218)) := by
  have c75705 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75706 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75707 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75708 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75709 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75710 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_0
  simp only [← c75705, ← c75706, ← c75707] at e_1230_0
  have e_1230_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_1
  simp only [← c75708, ← c75709, k2, ← c75710] at e_1230_1
  have hr := e_1230_1
  simp only [e_1230_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11219)) := by
  have c75711 := (publicBatchWrapper86_copies2365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75712 := (publicBatchWrapper86_copies2366 a h).1
  have c75713 := (publicBatchWrapper86_copies2366 a h).2.1
  have c75714 := (publicBatchWrapper86_copies2366 a h).2.2.1
  have c75715 := (publicBatchWrapper86_copies2366 a h).2.2.2.1
  have c75716 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_2
  simp only [← c75711, ← c75712, ← c75713] at e_1230_2
  have e_1230_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_3
  simp only [← c75714, ← c75715, k2, ← c75716] at e_1230_3
  have hr := e_1230_3
  simp only [e_1230_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11220)) := by
  have c75717 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.1
  have c75718 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.1
  have c75719 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.1
  have c75720 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.1
  have c75721 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.1
  have c75722 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_4
  simp only [← c75717, ← c75718, ← c75719] at e_1230_4
  have e_1230_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_5
  simp only [← c75720, ← c75721, k2, ← c75722] at e_1230_5
  have hr := e_1230_5
  simp only [e_1230_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11221)) := by
  have c75723 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75724 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75725 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75726 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75727 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75728 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_6
  simp only [← c75723, ← c75724, ← c75725] at e_1230_6
  have e_1230_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_7
  simp only [← c75726, ← c75727, k2, ← c75728] at e_1230_7
  have hr := e_1230_7
  simp only [e_1230_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11222)) := by
  have c75729 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75730 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75731 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75732 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75733 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75734 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_8
  simp only [← c75729, ← c75730, ← c75731] at e_1230_8
  have e_1230_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_9
  simp only [← c75732, ← c75733, k2, ← c75734] at e_1230_9
  have hr := e_1230_9
  simp only [e_1230_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11223)) := by
  have c75735 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75736 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75737 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75738 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75739 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75740 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_10
  simp only [← c75735, ← c75736, ← c75737] at e_1230_10
  have e_1230_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_11
  simp only [← c75738, ← c75739, k2, ← c75740] at e_1230_11
  have hr := e_1230_11
  simp only [e_1230_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11224)) := by
  have c75741 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75742 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75743 := (publicBatchWrapper86_copies2366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75744 := (publicBatchWrapper86_copies2367 a h).1
  have c75745 := (publicBatchWrapper86_copies2367 a h).2.1
  have c75746 := (publicBatchWrapper86_copies2367 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_12
  simp only [← c75741, ← c75742, ← c75743] at e_1230_12
  have e_1230_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_13
  simp only [← c75744, ← c75745, k2, ← c75746] at e_1230_13
  have hr := e_1230_13
  simp only [e_1230_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11225)) := by
  have c75747 := (publicBatchWrapper86_copies2367 a h).2.2.2.1
  have c75748 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.1
  have c75749 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.1
  have c75750 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.1
  have c75751 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.1
  have c75752 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_14
  simp only [← c75747, ← c75748, ← c75749] at e_1230_14
  have e_1230_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_15
  simp only [← c75750, ← c75751, k2, ← c75752] at e_1230_15
  have hr := e_1230_15
  simp only [e_1230_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11226)) := by
  have c75753 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.1
  have c75754 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75755 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75756 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75757 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75758 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_16
  simp only [← c75753, ← c75754, ← c75755] at e_1230_16
  have e_1230_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_17
  simp only [← c75756, ← c75757, k2, ← c75758] at e_1230_17
  have hr := e_1230_17
  simp only [e_1230_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1230 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11227)) := by
  have c75759 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75760 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75761 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75762 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75763 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75764 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1230_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_18
  simp only [← c75759, ← c75760, ← c75761] at e_1230_18
  have e_1230_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1230 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1230_19
  simp only [← c75762, ← c75763, k2, ← c75764] at e_1230_19
  have hr := e_1230_19
  simp only [e_1230_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11228)) := by
  have c75765 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75766 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75767 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75768 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75769 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75770 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_0
  simp only [← c75765, ← c75766, ← c75767] at e_1231_0
  have e_1231_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_1
  simp only [← c75768, ← c75769, k2, ← c75770] at e_1231_1
  have hr := e_1231_1
  simp only [e_1231_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11229)) := by
  have c75771 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75772 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75773 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75774 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75775 := (publicBatchWrapper86_copies2367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75776 := (publicBatchWrapper86_copies2368 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_2
  simp only [← c75771, ← c75772, ← c75773] at e_1231_2
  have e_1231_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_3
  simp only [← c75774, ← c75775, k2, ← c75776] at e_1231_3
  have hr := e_1231_3
  simp only [e_1231_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11230)) := by
  have c75777 := (publicBatchWrapper86_copies2368 a h).2.1
  have c75778 := (publicBatchWrapper86_copies2368 a h).2.2.1
  have c75779 := (publicBatchWrapper86_copies2368 a h).2.2.2.1
  have c75780 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.1
  have c75781 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.1
  have c75782 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_4
  simp only [← c75777, ← c75778, ← c75779] at e_1231_4
  have e_1231_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_5
  simp only [← c75780, ← c75781, k2, ← c75782] at e_1231_5
  have hr := e_1231_5
  simp only [e_1231_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11231)) := by
  have c75783 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.1
  have c75784 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.1
  have c75785 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.1
  have c75786 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75787 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75788 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_6
  simp only [← c75783, ← c75784, ← c75785] at e_1231_6
  have e_1231_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_7
  simp only [← c75786, ← c75787, k2, ← c75788] at e_1231_7
  have hr := e_1231_7
  simp only [e_1231_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11232)) := by
  have c75789 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75790 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75791 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75792 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75793 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75794 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_8
  simp only [← c75789, ← c75790, ← c75791] at e_1231_8
  have e_1231_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_9
  simp only [← c75792, ← c75793, k2, ← c75794] at e_1231_9
  have hr := e_1231_9
  simp only [e_1231_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11233)) := by
  have c75795 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75796 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75797 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75798 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75799 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75800 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_10
  simp only [← c75795, ← c75796, ← c75797] at e_1231_10
  have e_1231_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_11
  simp only [← c75798, ← c75799, k2, ← c75800] at e_1231_11
  have hr := e_1231_11
  simp only [e_1231_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11234)) := by
  have c75801 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75802 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75803 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75804 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75805 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75806 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_12
  simp only [← c75801, ← c75802, ← c75803] at e_1231_12
  have e_1231_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_13
  simp only [← c75804, ← c75805, k2, ← c75806] at e_1231_13
  have hr := e_1231_13
  simp only [e_1231_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11235)) := by
  have c75807 := (publicBatchWrapper86_copies2368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75808 := (publicBatchWrapper86_copies2369 a h).1
  have c75809 := (publicBatchWrapper86_copies2369 a h).2.1
  have c75810 := (publicBatchWrapper86_copies2369 a h).2.2.1
  have c75811 := (publicBatchWrapper86_copies2369 a h).2.2.2.1
  have c75812 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_14
  simp only [← c75807, ← c75808, ← c75809] at e_1231_14
  have e_1231_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_15
  simp only [← c75810, ← c75811, k2, ← c75812] at e_1231_15
  have hr := e_1231_15
  simp only [e_1231_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11236)) := by
  have c75813 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.1
  have c75814 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.1
  have c75815 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.1
  have c75816 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.1
  have c75817 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.1
  have c75818 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_16
  simp only [← c75813, ← c75814, ← c75815] at e_1231_16
  have e_1231_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_17
  simp only [← c75816, ← c75817, k2, ← c75818] at e_1231_17
  have hr := e_1231_17
  simp only [e_1231_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1231 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11237)) := by
  have c75819 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75820 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75821 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75822 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75823 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75824 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1231_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_18
  simp only [← c75819, ← c75820, ← c75821] at e_1231_18
  have e_1231_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1231 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1231_19
  simp only [← c75822, ← c75823, k2, ← c75824] at e_1231_19
  have hr := e_1231_19
  simp only [e_1231_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11238)) := by
  have c75825 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75826 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75827 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75828 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75829 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75830 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_0
  simp only [← c75825, ← c75826, ← c75827] at e_1232_0
  have e_1232_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_1
  simp only [← c75828, ← c75829, k2, ← c75830] at e_1232_1
  have hr := e_1232_1
  simp only [e_1232_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11239)) := by
  have c75831 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75832 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75833 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75834 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75835 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75836 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_2
  simp only [← c75831, ← c75832, ← c75833] at e_1232_2
  have e_1232_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_3
  simp only [← c75834, ← c75835, k2, ← c75836] at e_1232_3
  have hr := e_1232_3
  simp only [e_1232_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11240)) := by
  have c75837 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75838 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75839 := (publicBatchWrapper86_copies2369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75840 := (publicBatchWrapper86_copies2370 a h).1
  have c75841 := (publicBatchWrapper86_copies2370 a h).2.1
  have c75842 := (publicBatchWrapper86_copies2370 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_4
  simp only [← c75837, ← c75838, ← c75839] at e_1232_4
  have e_1232_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_5
  simp only [← c75840, ← c75841, k2, ← c75842] at e_1232_5
  have hr := e_1232_5
  simp only [e_1232_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11241)) := by
  have c75843 := (publicBatchWrapper86_copies2370 a h).2.2.2.1
  have c75844 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.1
  have c75845 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.1
  have c75846 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.1
  have c75847 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.1
  have c75848 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_6
  simp only [← c75843, ← c75844, ← c75845] at e_1232_6
  have e_1232_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_7
  simp only [← c75846, ← c75847, k2, ← c75848] at e_1232_7
  have hr := e_1232_7
  simp only [e_1232_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11242)) := by
  have c75849 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.1
  have c75850 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75851 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75852 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75853 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75854 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_8
  simp only [← c75849, ← c75850, ← c75851] at e_1232_8
  have e_1232_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_9
  simp only [← c75852, ← c75853, k2, ← c75854] at e_1232_9
  have hr := e_1232_9
  simp only [e_1232_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11243)) := by
  have c75855 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75856 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75857 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75858 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75859 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75860 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_10
  simp only [← c75855, ← c75856, ← c75857] at e_1232_10
  have e_1232_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_11
  simp only [← c75858, ← c75859, k2, ← c75860] at e_1232_11
  have hr := e_1232_11
  simp only [e_1232_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11244)) := by
  have c75861 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75862 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75863 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75864 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75865 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75866 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_12
  simp only [← c75861, ← c75862, ← c75863] at e_1232_12
  have e_1232_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_13
  simp only [← c75864, ← c75865, k2, ← c75866] at e_1232_13
  have hr := e_1232_13
  simp only [e_1232_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11245)) := by
  have c75867 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75868 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75869 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75870 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75871 := (publicBatchWrapper86_copies2370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75872 := (publicBatchWrapper86_copies2371 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_14
  simp only [← c75867, ← c75868, ← c75869] at e_1232_14
  have e_1232_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_15
  simp only [← c75870, ← c75871, k2, ← c75872] at e_1232_15
  have hr := e_1232_15
  simp only [e_1232_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11246)) := by
  have c75873 := (publicBatchWrapper86_copies2371 a h).2.1
  have c75874 := (publicBatchWrapper86_copies2371 a h).2.2.1
  have c75875 := (publicBatchWrapper86_copies2371 a h).2.2.2.1
  have c75876 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.1
  have c75877 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.1
  have c75878 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_16
  simp only [← c75873, ← c75874, ← c75875] at e_1232_16
  have e_1232_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_17
  simp only [← c75876, ← c75877, k2, ← c75878] at e_1232_17
  have hr := e_1232_17
  simp only [e_1232_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1232 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11247)) := by
  have c75879 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.1
  have c75880 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.1
  have c75881 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.1
  have c75882 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75883 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75884 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1232_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_18
  simp only [← c75879, ← c75880, ← c75881] at e_1232_18
  have e_1232_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1232 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1232_19
  simp only [← c75882, ← c75883, k2, ← c75884] at e_1232_19
  have hr := e_1232_19
  simp only [e_1232_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11248)) := by
  have c75885 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75886 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75887 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75888 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75889 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75890 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_0
  simp only [← c75885, ← c75886, ← c75887] at e_1233_0
  have e_1233_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_1
  simp only [← c75888, ← c75889, k2, ← c75890] at e_1233_1
  have hr := e_1233_1
  simp only [e_1233_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11249)) := by
  have c75891 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75892 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75893 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75894 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75895 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75896 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_2
  simp only [← c75891, ← c75892, ← c75893] at e_1233_2
  have e_1233_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_3
  simp only [← c75894, ← c75895, k2, ← c75896] at e_1233_3
  have hr := e_1233_3
  simp only [e_1233_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11250)) := by
  have c75897 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75898 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75899 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75900 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75901 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75902 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_4
  simp only [← c75897, ← c75898, ← c75899] at e_1233_4
  have e_1233_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_5
  simp only [← c75900, ← c75901, k2, ← c75902] at e_1233_5
  have hr := e_1233_5
  simp only [e_1233_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11251)) := by
  have c75903 := (publicBatchWrapper86_copies2371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75904 := (publicBatchWrapper86_copies2372 a h).1
  have c75905 := (publicBatchWrapper86_copies2372 a h).2.1
  have c75906 := (publicBatchWrapper86_copies2372 a h).2.2.1
  have c75907 := (publicBatchWrapper86_copies2372 a h).2.2.2.1
  have c75908 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_6
  simp only [← c75903, ← c75904, ← c75905] at e_1233_6
  have e_1233_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_7
  simp only [← c75906, ← c75907, k2, ← c75908] at e_1233_7
  have hr := e_1233_7
  simp only [e_1233_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11252)) := by
  have c75909 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.1
  have c75910 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.1
  have c75911 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.1
  have c75912 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.1
  have c75913 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.1
  have c75914 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_8
  simp only [← c75909, ← c75910, ← c75911] at e_1233_8
  have e_1233_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_9
  simp only [← c75912, ← c75913, k2, ← c75914] at e_1233_9
  have hr := e_1233_9
  simp only [e_1233_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11253)) := by
  have c75915 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75916 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75917 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75918 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75919 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75920 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_10
  simp only [← c75915, ← c75916, ← c75917] at e_1233_10
  have e_1233_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_11
  simp only [← c75918, ← c75919, k2, ← c75920] at e_1233_11
  have hr := e_1233_11
  simp only [e_1233_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11254)) := by
  have c75921 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75922 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75923 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75924 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75925 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75926 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_12
  simp only [← c75921, ← c75922, ← c75923] at e_1233_12
  have e_1233_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_13
  simp only [← c75924, ← c75925, k2, ← c75926] at e_1233_13
  have hr := e_1233_13
  simp only [e_1233_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11255)) := by
  have c75927 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75928 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75929 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75930 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75931 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75932 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_14
  simp only [← c75927, ← c75928, ← c75929] at e_1233_14
  have e_1233_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_15
  simp only [← c75930, ← c75931, k2, ← c75932] at e_1233_15
  have hr := e_1233_15
  simp only [e_1233_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11256)) := by
  have c75933 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75934 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75935 := (publicBatchWrapper86_copies2372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75936 := (publicBatchWrapper86_copies2373 a h).1
  have c75937 := (publicBatchWrapper86_copies2373 a h).2.1
  have c75938 := (publicBatchWrapper86_copies2373 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_16
  simp only [← c75933, ← c75934, ← c75935] at e_1233_16
  have e_1233_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_17
  simp only [← c75936, ← c75937, k2, ← c75938] at e_1233_17
  have hr := e_1233_17
  simp only [e_1233_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1233 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11257)) := by
  have c75939 := (publicBatchWrapper86_copies2373 a h).2.2.2.1
  have c75940 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.1
  have c75941 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.1
  have c75942 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.1
  have c75943 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.1
  have c75944 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1233_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_18
  simp only [← c75939, ← c75940, ← c75941] at e_1233_18
  have e_1233_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1233 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1233_19
  simp only [← c75942, ← c75943, k2, ← c75944] at e_1233_19
  have hr := e_1233_19
  simp only [e_1233_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11258)) := by
  have c75945 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.1
  have c75946 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75947 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75948 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75949 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75950 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_0
  simp only [← c75945, ← c75946, ← c75947] at e_1234_0
  have e_1234_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_1
  simp only [← c75948, ← c75949, k2, ← c75950] at e_1234_1
  have hr := e_1234_1
  simp only [e_1234_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11259)) := by
  have c75951 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75952 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75953 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75954 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75955 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75956 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_2
  simp only [← c75951, ← c75952, ← c75953] at e_1234_2
  have e_1234_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_3
  simp only [← c75954, ← c75955, k2, ← c75956] at e_1234_3
  have hr := e_1234_3
  simp only [e_1234_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11260)) := by
  have c75957 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75958 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75959 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75960 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75961 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75962 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_4
  simp only [← c75957, ← c75958, ← c75959] at e_1234_4
  have e_1234_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_5
  simp only [← c75960, ← c75961, k2, ← c75962] at e_1234_5
  have hr := e_1234_5
  simp only [e_1234_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11261)) := by
  have c75963 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75964 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75965 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75966 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75967 := (publicBatchWrapper86_copies2373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c75968 := (publicBatchWrapper86_copies2374 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_6
  simp only [← c75963, ← c75964, ← c75965] at e_1234_6
  have e_1234_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_7
  simp only [← c75966, ← c75967, k2, ← c75968] at e_1234_7
  have hr := e_1234_7
  simp only [e_1234_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11262)) := by
  have c75969 := (publicBatchWrapper86_copies2374 a h).2.1
  have c75970 := (publicBatchWrapper86_copies2374 a h).2.2.1
  have c75971 := (publicBatchWrapper86_copies2374 a h).2.2.2.1
  have c75972 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.1
  have c75973 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.1
  have c75974 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_8
  simp only [← c75969, ← c75970, ← c75971] at e_1234_8
  have e_1234_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_9
  simp only [← c75972, ← c75973, k2, ← c75974] at e_1234_9
  have hr := e_1234_9
  simp only [e_1234_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11263)) := by
  have c75975 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.1
  have c75976 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.1
  have c75977 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.1
  have c75978 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c75979 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c75980 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_10
  simp only [← c75975, ← c75976, ← c75977] at e_1234_10
  have e_1234_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_11
  simp only [← c75978, ← c75979, k2, ← c75980] at e_1234_11
  have hr := e_1234_11
  simp only [e_1234_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11264)) := by
  have c75981 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75982 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75983 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75984 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75985 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75986 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_12
  simp only [← c75981, ← c75982, ← c75983] at e_1234_12
  have e_1234_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_13
  simp only [← c75984, ← c75985, k2, ← c75986] at e_1234_13
  have hr := e_1234_13
  simp only [e_1234_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11265)) := by
  have c75987 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75988 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75989 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75990 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75991 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75992 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_14
  simp only [← c75987, ← c75988, ← c75989] at e_1234_14
  have e_1234_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_15
  simp only [← c75990, ← c75991, k2, ← c75992] at e_1234_15
  have hr := e_1234_15
  simp only [e_1234_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11266)) := by
  have c75993 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75994 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75995 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75996 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75997 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c75998 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_16
  simp only [← c75993, ← c75994, ← c75995] at e_1234_16
  have e_1234_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_17
  simp only [← c75996, ← c75997, k2, ← c75998] at e_1234_17
  have hr := e_1234_17
  simp only [e_1234_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1234 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11267)) := by
  have c75999 := (publicBatchWrapper86_copies2374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76000 := (publicBatchWrapper86_copies2375 a h).1
  have c76001 := (publicBatchWrapper86_copies2375 a h).2.1
  have c76002 := (publicBatchWrapper86_copies2375 a h).2.2.1
  have c76003 := (publicBatchWrapper86_copies2375 a h).2.2.2.1
  have c76004 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1234_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_18
  simp only [← c75999, ← c76000, ← c76001] at e_1234_18
  have e_1234_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1234 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1234_19
  simp only [← c76002, ← c76003, k2, ← c76004] at e_1234_19
  have hr := e_1234_19
  simp only [e_1234_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11268)) := by
  have c76005 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.1
  have c76006 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.1
  have c76007 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.1
  have c76008 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.1
  have c76009 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.1
  have c76010 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_0
  simp only [← c76005, ← c76006, ← c76007] at e_1235_0
  have e_1235_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_1
  simp only [← c76008, ← c76009, k2, ← c76010] at e_1235_1
  have hr := e_1235_1
  simp only [e_1235_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11269)) := by
  have c76011 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76012 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76013 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76014 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76015 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76016 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_2
  simp only [← c76011, ← c76012, ← c76013] at e_1235_2
  have e_1235_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_3
  simp only [← c76014, ← c76015, k2, ← c76016] at e_1235_3
  have hr := e_1235_3
  simp only [e_1235_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11270)) := by
  have c76017 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76018 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76019 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76020 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76021 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76022 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_4
  simp only [← c76017, ← c76018, ← c76019] at e_1235_4
  have e_1235_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_5
  simp only [← c76020, ← c76021, k2, ← c76022] at e_1235_5
  have hr := e_1235_5
  simp only [e_1235_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11271)) := by
  have c76023 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76024 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76025 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76026 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76027 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76028 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_6
  simp only [← c76023, ← c76024, ← c76025] at e_1235_6
  have e_1235_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_7
  simp only [← c76026, ← c76027, k2, ← c76028] at e_1235_7
  have hr := e_1235_7
  simp only [e_1235_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11272)) := by
  have c76029 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76030 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76031 := (publicBatchWrapper86_copies2375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76032 := (publicBatchWrapper86_copies2376 a h).1
  have c76033 := (publicBatchWrapper86_copies2376 a h).2.1
  have c76034 := (publicBatchWrapper86_copies2376 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_8
  simp only [← c76029, ← c76030, ← c76031] at e_1235_8
  have e_1235_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_9
  simp only [← c76032, ← c76033, k2, ← c76034] at e_1235_9
  have hr := e_1235_9
  simp only [e_1235_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11273)) := by
  have c76035 := (publicBatchWrapper86_copies2376 a h).2.2.2.1
  have c76036 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.1
  have c76037 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.1
  have c76038 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.1
  have c76039 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.1
  have c76040 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_10
  simp only [← c76035, ← c76036, ← c76037] at e_1235_10
  have e_1235_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_11
  simp only [← c76038, ← c76039, k2, ← c76040] at e_1235_11
  have hr := e_1235_11
  simp only [e_1235_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11274)) := by
  have c76041 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.1
  have c76042 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76043 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76044 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76045 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76046 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_12
  simp only [← c76041, ← c76042, ← c76043] at e_1235_12
  have e_1235_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_13
  simp only [← c76044, ← c76045, k2, ← c76046] at e_1235_13
  have hr := e_1235_13
  simp only [e_1235_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11275)) := by
  have c76047 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76048 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76049 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76050 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76051 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76052 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_14
  simp only [← c76047, ← c76048, ← c76049] at e_1235_14
  have e_1235_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_15
  simp only [← c76050, ← c76051, k2, ← c76052] at e_1235_15
  have hr := e_1235_15
  simp only [e_1235_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11276)) := by
  have c76053 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76054 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76055 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76056 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76057 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76058 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_16
  simp only [← c76053, ← c76054, ← c76055] at e_1235_16
  have e_1235_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_17
  simp only [← c76056, ← c76057, k2, ← c76058] at e_1235_17
  have hr := e_1235_17
  simp only [e_1235_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1235 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11277)) := by
  have c76059 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76060 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76061 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76062 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76063 := (publicBatchWrapper86_copies2376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76064 := (publicBatchWrapper86_copies2377 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1235_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_18
  simp only [← c76059, ← c76060, ← c76061] at e_1235_18
  have e_1235_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1235 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1235_19
  simp only [← c76062, ← c76063, k2, ← c76064] at e_1235_19
  have hr := e_1235_19
  simp only [e_1235_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11278)) := by
  have c76065 := (publicBatchWrapper86_copies2377 a h).2.1
  have c76066 := (publicBatchWrapper86_copies2377 a h).2.2.1
  have c76067 := (publicBatchWrapper86_copies2377 a h).2.2.2.1
  have c76068 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.1
  have c76069 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.1
  have c76070 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_0
  simp only [← c76065, ← c76066, ← c76067] at e_1236_0
  have e_1236_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_1
  simp only [← c76068, ← c76069, k2, ← c76070] at e_1236_1
  have hr := e_1236_1
  simp only [e_1236_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11279)) := by
  have c76071 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.1
  have c76072 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.1
  have c76073 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.1
  have c76074 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76075 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76076 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_2
  simp only [← c76071, ← c76072, ← c76073] at e_1236_2
  have e_1236_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_3
  simp only [← c76074, ← c76075, k2, ← c76076] at e_1236_3
  have hr := e_1236_3
  simp only [e_1236_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11280)) := by
  have c76077 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76078 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76079 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76080 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76081 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76082 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_4
  simp only [← c76077, ← c76078, ← c76079] at e_1236_4
  have e_1236_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_5
  simp only [← c76080, ← c76081, k2, ← c76082] at e_1236_5
  have hr := e_1236_5
  simp only [e_1236_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11281)) := by
  have c76083 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76084 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76085 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76086 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76087 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76088 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_6
  simp only [← c76083, ← c76084, ← c76085] at e_1236_6
  have e_1236_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_7
  simp only [← c76086, ← c76087, k2, ← c76088] at e_1236_7
  have hr := e_1236_7
  simp only [e_1236_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11282)) := by
  have c76089 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76090 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76091 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76092 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76093 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76094 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_8
  simp only [← c76089, ← c76090, ← c76091] at e_1236_8
  have e_1236_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_9
  simp only [← c76092, ← c76093, k2, ← c76094] at e_1236_9
  have hr := e_1236_9
  simp only [e_1236_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11283)) := by
  have c76095 := (publicBatchWrapper86_copies2377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76096 := (publicBatchWrapper86_copies2378 a h).1
  have c76097 := (publicBatchWrapper86_copies2378 a h).2.1
  have c76098 := (publicBatchWrapper86_copies2378 a h).2.2.1
  have c76099 := (publicBatchWrapper86_copies2378 a h).2.2.2.1
  have c76100 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_10
  simp only [← c76095, ← c76096, ← c76097] at e_1236_10
  have e_1236_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_11
  simp only [← c76098, ← c76099, k2, ← c76100] at e_1236_11
  have hr := e_1236_11
  simp only [e_1236_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11284)) := by
  have c76101 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.1
  have c76102 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.1
  have c76103 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.1
  have c76104 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.1
  have c76105 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.1
  have c76106 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_12
  simp only [← c76101, ← c76102, ← c76103] at e_1236_12
  have e_1236_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_13
  simp only [← c76104, ← c76105, k2, ← c76106] at e_1236_13
  have hr := e_1236_13
  simp only [e_1236_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11285)) := by
  have c76107 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76108 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76109 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76110 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76111 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76112 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_14
  simp only [← c76107, ← c76108, ← c76109] at e_1236_14
  have e_1236_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_15
  simp only [← c76110, ← c76111, k2, ← c76112] at e_1236_15
  have hr := e_1236_15
  simp only [e_1236_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11286)) := by
  have c76113 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76114 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76115 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76116 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76117 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76118 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_16
  simp only [← c76113, ← c76114, ← c76115] at e_1236_16
  have e_1236_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_17
  simp only [← c76116, ← c76117, k2, ← c76118] at e_1236_17
  have hr := e_1236_17
  simp only [e_1236_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1236 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11287)) := by
  have c76119 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76120 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76121 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76122 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76123 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76124 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1236_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_18
  simp only [← c76119, ← c76120, ← c76121] at e_1236_18
  have e_1236_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1236 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1236_19
  simp only [← c76122, ← c76123, k2, ← c76124] at e_1236_19
  have hr := e_1236_19
  simp only [e_1236_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11288)) := by
  have c76125 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76126 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76127 := (publicBatchWrapper86_copies2378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76128 := (publicBatchWrapper86_copies2379 a h).1
  have c76129 := (publicBatchWrapper86_copies2379 a h).2.1
  have c76130 := (publicBatchWrapper86_copies2379 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_0
  simp only [← c76125, ← c76126, ← c76127] at e_1237_0
  have e_1237_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_1
  simp only [← c76128, ← c76129, k2, ← c76130] at e_1237_1
  have hr := e_1237_1
  simp only [e_1237_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11289)) := by
  have c76131 := (publicBatchWrapper86_copies2379 a h).2.2.2.1
  have c76132 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.1
  have c76133 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.1
  have c76134 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.1
  have c76135 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.1
  have c76136 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_2
  simp only [← c76131, ← c76132, ← c76133] at e_1237_2
  have e_1237_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_3
  simp only [← c76134, ← c76135, k2, ← c76136] at e_1237_3
  have hr := e_1237_3
  simp only [e_1237_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11290)) := by
  have c76137 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.1
  have c76138 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76139 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76140 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76141 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76142 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_4
  simp only [← c76137, ← c76138, ← c76139] at e_1237_4
  have e_1237_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_5
  simp only [← c76140, ← c76141, k2, ← c76142] at e_1237_5
  have hr := e_1237_5
  simp only [e_1237_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11291)) := by
  have c76143 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76144 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76145 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76146 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76147 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76148 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_6
  simp only [← c76143, ← c76144, ← c76145] at e_1237_6
  have e_1237_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_7
  simp only [← c76146, ← c76147, k2, ← c76148] at e_1237_7
  have hr := e_1237_7
  simp only [e_1237_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11292)) := by
  have c76149 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76150 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76151 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76152 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76153 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76154 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_8
  simp only [← c76149, ← c76150, ← c76151] at e_1237_8
  have e_1237_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_9
  simp only [← c76152, ← c76153, k2, ← c76154] at e_1237_9
  have hr := e_1237_9
  simp only [e_1237_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11293)) := by
  have c76155 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76156 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76157 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76158 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76159 := (publicBatchWrapper86_copies2379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76160 := (publicBatchWrapper86_copies2380 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_10
  simp only [← c76155, ← c76156, ← c76157] at e_1237_10
  have e_1237_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_11
  simp only [← c76158, ← c76159, k2, ← c76160] at e_1237_11
  have hr := e_1237_11
  simp only [e_1237_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11294)) := by
  have c76161 := (publicBatchWrapper86_copies2380 a h).2.1
  have c76162 := (publicBatchWrapper86_copies2380 a h).2.2.1
  have c76163 := (publicBatchWrapper86_copies2380 a h).2.2.2.1
  have c76164 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.1
  have c76165 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.1
  have c76166 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_12
  simp only [← c76161, ← c76162, ← c76163] at e_1237_12
  have e_1237_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_13
  simp only [← c76164, ← c76165, k2, ← c76166] at e_1237_13
  have hr := e_1237_13
  simp only [e_1237_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11295)) := by
  have c76167 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.1
  have c76168 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.1
  have c76169 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.1
  have c76170 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76171 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76172 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_14
  simp only [← c76167, ← c76168, ← c76169] at e_1237_14
  have e_1237_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_15
  simp only [← c76170, ← c76171, k2, ← c76172] at e_1237_15
  have hr := e_1237_15
  simp only [e_1237_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11296)) := by
  have c76173 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76174 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76175 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76176 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76177 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76178 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_16
  simp only [← c76173, ← c76174, ← c76175] at e_1237_16
  have e_1237_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_17
  simp only [← c76176, ← c76177, k2, ← c76178] at e_1237_17
  have hr := e_1237_17
  simp only [e_1237_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1237 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11297)) := by
  have c76179 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76180 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76181 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76182 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76183 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76184 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1237_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_18
  simp only [← c76179, ← c76180, ← c76181] at e_1237_18
  have e_1237_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1237 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1237_19
  simp only [← c76182, ← c76183, k2, ← c76184] at e_1237_19
  have hr := e_1237_19
  simp only [e_1237_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11298)) := by
  have c76185 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76186 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76187 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76188 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76189 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76190 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_0
  simp only [← c76185, ← c76186, ← c76187] at e_1238_0
  have e_1238_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_1
  simp only [← c76188, ← c76189, k2, ← c76190] at e_1238_1
  have hr := e_1238_1
  simp only [e_1238_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11299)) := by
  have c76191 := (publicBatchWrapper86_copies2380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76192 := (publicBatchWrapper86_copies2381 a h).1
  have c76193 := (publicBatchWrapper86_copies2381 a h).2.1
  have c76194 := (publicBatchWrapper86_copies2381 a h).2.2.1
  have c76195 := (publicBatchWrapper86_copies2381 a h).2.2.2.1
  have c76196 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_2
  simp only [← c76191, ← c76192, ← c76193] at e_1238_2
  have e_1238_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_3
  simp only [← c76194, ← c76195, k2, ← c76196] at e_1238_3
  have hr := e_1238_3
  simp only [e_1238_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11300)) := by
  have c76197 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.1
  have c76198 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.1
  have c76199 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.1
  have c76200 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.1
  have c76201 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.1
  have c76202 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_4
  simp only [← c76197, ← c76198, ← c76199] at e_1238_4
  have e_1238_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_5
  simp only [← c76200, ← c76201, k2, ← c76202] at e_1238_5
  have hr := e_1238_5
  simp only [e_1238_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11301)) := by
  have c76203 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76204 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76205 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76206 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76207 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76208 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_6
  simp only [← c76203, ← c76204, ← c76205] at e_1238_6
  have e_1238_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_7
  simp only [← c76206, ← c76207, k2, ← c76208] at e_1238_7
  have hr := e_1238_7
  simp only [e_1238_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11302)) := by
  have c76209 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76210 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76211 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76212 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76213 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76214 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_8
  simp only [← c76209, ← c76210, ← c76211] at e_1238_8
  have e_1238_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_9
  simp only [← c76212, ← c76213, k2, ← c76214] at e_1238_9
  have hr := e_1238_9
  simp only [e_1238_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11303)) := by
  have c76215 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76216 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76217 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76218 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76219 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76220 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_10
  simp only [← c76215, ← c76216, ← c76217] at e_1238_10
  have e_1238_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_11
  simp only [← c76218, ← c76219, k2, ← c76220] at e_1238_11
  have hr := e_1238_11
  simp only [e_1238_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11304)) := by
  have c76221 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76222 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76223 := (publicBatchWrapper86_copies2381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76224 := (publicBatchWrapper86_copies2382 a h).1
  have c76225 := (publicBatchWrapper86_copies2382 a h).2.1
  have c76226 := (publicBatchWrapper86_copies2382 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_12
  simp only [← c76221, ← c76222, ← c76223] at e_1238_12
  have e_1238_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_13
  simp only [← c76224, ← c76225, k2, ← c76226] at e_1238_13
  have hr := e_1238_13
  simp only [e_1238_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11305)) := by
  have c76227 := (publicBatchWrapper86_copies2382 a h).2.2.2.1
  have c76228 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.1
  have c76229 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.1
  have c76230 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.1
  have c76231 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.1
  have c76232 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_14
  simp only [← c76227, ← c76228, ← c76229] at e_1238_14
  have e_1238_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_15
  simp only [← c76230, ← c76231, k2, ← c76232] at e_1238_15
  have hr := e_1238_15
  simp only [e_1238_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11306)) := by
  have c76233 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.1
  have c76234 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76235 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76236 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76237 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76238 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_16
  simp only [← c76233, ← c76234, ← c76235] at e_1238_16
  have e_1238_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_17
  simp only [← c76236, ← c76237, k2, ← c76238] at e_1238_17
  have hr := e_1238_17
  simp only [e_1238_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1238 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11307)) := by
  have c76239 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76240 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76241 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76242 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76243 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76244 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1238_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_18
  simp only [← c76239, ← c76240, ← c76241] at e_1238_18
  have e_1238_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1238 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1238_19
  simp only [← c76242, ← c76243, k2, ← c76244] at e_1238_19
  have hr := e_1238_19
  simp only [e_1238_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11308)) := by
  have c76245 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76246 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76247 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76248 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76249 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76250 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_0
  simp only [← c76245, ← c76246, ← c76247] at e_1239_0
  have e_1239_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_1
  simp only [← c76248, ← c76249, k2, ← c76250] at e_1239_1
  have hr := e_1239_1
  simp only [e_1239_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11309)) := by
  have c76251 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76252 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76253 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76254 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76255 := (publicBatchWrapper86_copies2382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76256 := (publicBatchWrapper86_copies2383 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_2
  simp only [← c76251, ← c76252, ← c76253] at e_1239_2
  have e_1239_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_3
  simp only [← c76254, ← c76255, k2, ← c76256] at e_1239_3
  have hr := e_1239_3
  simp only [e_1239_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11310)) := by
  have c76257 := (publicBatchWrapper86_copies2383 a h).2.1
  have c76258 := (publicBatchWrapper86_copies2383 a h).2.2.1
  have c76259 := (publicBatchWrapper86_copies2383 a h).2.2.2.1
  have c76260 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.1
  have c76261 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.1
  have c76262 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_4
  simp only [← c76257, ← c76258, ← c76259] at e_1239_4
  have e_1239_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_5
  simp only [← c76260, ← c76261, k2, ← c76262] at e_1239_5
  have hr := e_1239_5
  simp only [e_1239_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11311)) := by
  have c76263 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.1
  have c76264 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.1
  have c76265 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.1
  have c76266 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76267 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76268 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_6
  simp only [← c76263, ← c76264, ← c76265] at e_1239_6
  have e_1239_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_7
  simp only [← c76266, ← c76267, k2, ← c76268] at e_1239_7
  have hr := e_1239_7
  simp only [e_1239_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11312)) := by
  have c76269 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76270 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76271 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76272 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76273 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76274 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_8
  simp only [← c76269, ← c76270, ← c76271] at e_1239_8
  have e_1239_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_9
  simp only [← c76272, ← c76273, k2, ← c76274] at e_1239_9
  have hr := e_1239_9
  simp only [e_1239_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11313)) := by
  have c76275 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76276 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76277 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76278 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76279 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76280 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_10
  simp only [← c76275, ← c76276, ← c76277] at e_1239_10
  have e_1239_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_11
  simp only [← c76278, ← c76279, k2, ← c76280] at e_1239_11
  have hr := e_1239_11
  simp only [e_1239_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11314)) := by
  have c76281 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76282 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76283 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76284 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76285 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76286 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_12
  simp only [← c76281, ← c76282, ← c76283] at e_1239_12
  have e_1239_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_13
  simp only [← c76284, ← c76285, k2, ← c76286] at e_1239_13
  have hr := e_1239_13
  simp only [e_1239_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11315)) := by
  have c76287 := (publicBatchWrapper86_copies2383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76288 := (publicBatchWrapper86_copies2384 a h).1
  have c76289 := (publicBatchWrapper86_copies2384 a h).2.1
  have c76290 := (publicBatchWrapper86_copies2384 a h).2.2.1
  have c76291 := (publicBatchWrapper86_copies2384 a h).2.2.2.1
  have c76292 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_14
  simp only [← c76287, ← c76288, ← c76289] at e_1239_14
  have e_1239_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_15
  simp only [← c76290, ← c76291, k2, ← c76292] at e_1239_15
  have hr := e_1239_15
  simp only [e_1239_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11316)) := by
  have c76293 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.1
  have c76294 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.1
  have c76295 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.1
  have c76296 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.1
  have c76297 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.1
  have c76298 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_16
  simp only [← c76293, ← c76294, ← c76295] at e_1239_16
  have e_1239_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_17
  simp only [← c76296, ← c76297, k2, ← c76298] at e_1239_17
  have hr := e_1239_17
  simp only [e_1239_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1239 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11317)) := by
  have c76299 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76300 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76301 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76302 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76303 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76304 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1239_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_18
  simp only [← c76299, ← c76300, ← c76301] at e_1239_18
  have e_1239_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1239 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1239_19
  simp only [← c76302, ← c76303, k2, ← c76304] at e_1239_19
  have hr := e_1239_19
  simp only [e_1239_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11318)) := by
  have c76305 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76306 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76307 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76308 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76309 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76310 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_0
  simp only [← c76305, ← c76306, ← c76307] at e_1240_0
  have e_1240_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_1
  simp only [← c76308, ← c76309, k2, ← c76310] at e_1240_1
  have hr := e_1240_1
  simp only [e_1240_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11319)) := by
  have c76311 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76312 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76313 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76314 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76315 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76316 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_2
  simp only [← c76311, ← c76312, ← c76313] at e_1240_2
  have e_1240_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_3
  simp only [← c76314, ← c76315, k2, ← c76316] at e_1240_3
  have hr := e_1240_3
  simp only [e_1240_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11320)) := by
  have c76317 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76318 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76319 := (publicBatchWrapper86_copies2384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76320 := (publicBatchWrapper86_copies2385 a h).1
  have c76321 := (publicBatchWrapper86_copies2385 a h).2.1
  have c76322 := (publicBatchWrapper86_copies2385 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_4
  simp only [← c76317, ← c76318, ← c76319] at e_1240_4
  have e_1240_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_5
  simp only [← c76320, ← c76321, k2, ← c76322] at e_1240_5
  have hr := e_1240_5
  simp only [e_1240_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11321)) := by
  have c76323 := (publicBatchWrapper86_copies2385 a h).2.2.2.1
  have c76324 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.1
  have c76325 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.1
  have c76326 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.1
  have c76327 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.1
  have c76328 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_6
  simp only [← c76323, ← c76324, ← c76325] at e_1240_6
  have e_1240_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_7
  simp only [← c76326, ← c76327, k2, ← c76328] at e_1240_7
  have hr := e_1240_7
  simp only [e_1240_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11322)) := by
  have c76329 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.1
  have c76330 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76331 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76332 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76333 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76334 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_8
  simp only [← c76329, ← c76330, ← c76331] at e_1240_8
  have e_1240_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_9
  simp only [← c76332, ← c76333, k2, ← c76334] at e_1240_9
  have hr := e_1240_9
  simp only [e_1240_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11323)) := by
  have c76335 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76336 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76337 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76338 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76339 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76340 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_10
  simp only [← c76335, ← c76336, ← c76337] at e_1240_10
  have e_1240_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_11
  simp only [← c76338, ← c76339, k2, ← c76340] at e_1240_11
  have hr := e_1240_11
  simp only [e_1240_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11324)) := by
  have c76341 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76342 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76343 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76344 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76345 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76346 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_12
  simp only [← c76341, ← c76342, ← c76343] at e_1240_12
  have e_1240_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_13
  simp only [← c76344, ← c76345, k2, ← c76346] at e_1240_13
  have hr := e_1240_13
  simp only [e_1240_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f11999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11325)) := by
  have c76347 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76348 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76349 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76350 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76351 := (publicBatchWrapper86_copies2385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76352 := (publicBatchWrapper86_copies2386 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_14
  simp only [← c76347, ← c76348, ← c76349] at e_1240_14
  have e_1240_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_15
  simp only [← c76350, ← c76351, k2, ← c76352] at e_1240_15
  have hr := e_1240_15
  simp only [e_1240_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11326)) := by
  have c76353 := (publicBatchWrapper86_copies2386 a h).2.1
  have c76354 := (publicBatchWrapper86_copies2386 a h).2.2.1
  have c76355 := (publicBatchWrapper86_copies2386 a h).2.2.2.1
  have c76356 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.1
  have c76357 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.1
  have c76358 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_16
  simp only [← c76353, ← c76354, ← c76355] at e_1240_16
  have e_1240_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_17
  simp only [← c76356, ← c76357, k2, ← c76358] at e_1240_17
  have hr := e_1240_17
  simp only [e_1240_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1240 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11327)) := by
  have c76359 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.1
  have c76360 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.1
  have c76361 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.1
  have c76362 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76363 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76364 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1240_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_18
  simp only [← c76359, ← c76360, ← c76361] at e_1240_18
  have e_1240_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1240 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1240_19
  simp only [← c76362, ← c76363, k2, ← c76364] at e_1240_19
  have hr := e_1240_19
  simp only [e_1240_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11328)) := by
  have c76365 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76366 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76367 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76368 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76369 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76370 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_0
  simp only [← c76365, ← c76366, ← c76367] at e_1241_0
  have e_1241_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_1
  simp only [← c76368, ← c76369, k2, ← c76370] at e_1241_1
  have hr := e_1241_1
  simp only [e_1241_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11329)) := by
  have c76371 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76372 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76373 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76374 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76375 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76376 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_2
  simp only [← c76371, ← c76372, ← c76373] at e_1241_2
  have e_1241_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_3
  simp only [← c76374, ← c76375, k2, ← c76376] at e_1241_3
  have hr := e_1241_3
  simp only [e_1241_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11330)) := by
  have c76377 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76378 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76379 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76380 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76381 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76382 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_4
  simp only [← c76377, ← c76378, ← c76379] at e_1241_4
  have e_1241_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_5
  simp only [← c76380, ← c76381, k2, ← c76382] at e_1241_5
  have hr := e_1241_5
  simp only [e_1241_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11331)) := by
  have c76383 := (publicBatchWrapper86_copies2386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76384 := (publicBatchWrapper86_copies2387 a h).1
  have c76385 := (publicBatchWrapper86_copies2387 a h).2.1
  have c76386 := (publicBatchWrapper86_copies2387 a h).2.2.1
  have c76387 := (publicBatchWrapper86_copies2387 a h).2.2.2.1
  have c76388 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_6
  simp only [← c76383, ← c76384, ← c76385] at e_1241_6
  have e_1241_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_7
  simp only [← c76386, ← c76387, k2, ← c76388] at e_1241_7
  have hr := e_1241_7
  simp only [e_1241_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11332)) := by
  have c76389 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.1
  have c76390 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.1
  have c76391 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.1
  have c76392 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.1
  have c76393 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.1
  have c76394 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_8
  simp only [← c76389, ← c76390, ← c76391] at e_1241_8
  have e_1241_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_9
  simp only [← c76392, ← c76393, k2, ← c76394] at e_1241_9
  have hr := e_1241_9
  simp only [e_1241_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11333)) := by
  have c76395 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76396 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76397 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76398 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76399 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76400 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_10
  simp only [← c76395, ← c76396, ← c76397] at e_1241_10
  have e_1241_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_11
  simp only [← c76398, ← c76399, k2, ← c76400] at e_1241_11
  have hr := e_1241_11
  simp only [e_1241_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11334)) := by
  have c76401 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76402 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76403 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76404 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76405 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76406 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_12
  simp only [← c76401, ← c76402, ← c76403] at e_1241_12
  have e_1241_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_13
  simp only [← c76404, ← c76405, k2, ← c76406] at e_1241_13
  have hr := e_1241_13
  simp only [e_1241_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11335)) := by
  have c76407 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76408 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76409 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76410 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76411 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76412 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_14
  simp only [← c76407, ← c76408, ← c76409] at e_1241_14
  have e_1241_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_15
  simp only [← c76410, ← c76411, k2, ← c76412] at e_1241_15
  have hr := e_1241_15
  simp only [e_1241_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11336)) := by
  have c76413 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76414 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76415 := (publicBatchWrapper86_copies2387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76416 := (publicBatchWrapper86_copies2388 a h).1
  have c76417 := (publicBatchWrapper86_copies2388 a h).2.1
  have c76418 := (publicBatchWrapper86_copies2388 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_16
  simp only [← c76413, ← c76414, ← c76415] at e_1241_16
  have e_1241_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_17
  simp only [← c76416, ← c76417, k2, ← c76418] at e_1241_17
  have hr := e_1241_17
  simp only [e_1241_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1241 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11337)) := by
  have c76419 := (publicBatchWrapper86_copies2388 a h).2.2.2.1
  have c76420 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.1
  have c76421 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.1
  have c76422 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.1
  have c76423 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.1
  have c76424 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1241_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_18
  simp only [← c76419, ← c76420, ← c76421] at e_1241_18
  have e_1241_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1241 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1241_19
  simp only [← c76422, ← c76423, k2, ← c76424] at e_1241_19
  have hr := e_1241_19
  simp only [e_1241_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11338)) := by
  have c76425 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.1
  have c76426 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76427 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76428 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76429 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76430 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_0
  simp only [← c76425, ← c76426, ← c76427] at e_1242_0
  have e_1242_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_1
  simp only [← c76428, ← c76429, k2, ← c76430] at e_1242_1
  have hr := e_1242_1
  simp only [e_1242_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11339)) := by
  have c76431 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76432 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76433 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76434 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76435 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76436 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_2
  simp only [← c76431, ← c76432, ← c76433] at e_1242_2
  have e_1242_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_3
  simp only [← c76434, ← c76435, k2, ← c76436] at e_1242_3
  have hr := e_1242_3
  simp only [e_1242_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11340)) := by
  have c76437 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76438 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76439 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76440 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76441 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76442 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_4
  simp only [← c76437, ← c76438, ← c76439] at e_1242_4
  have e_1242_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_5
  simp only [← c76440, ← c76441, k2, ← c76442] at e_1242_5
  have hr := e_1242_5
  simp only [e_1242_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11341)) := by
  have c76443 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76444 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76445 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76446 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76447 := (publicBatchWrapper86_copies2388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76448 := (publicBatchWrapper86_copies2389 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_6
  simp only [← c76443, ← c76444, ← c76445] at e_1242_6
  have e_1242_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_7
  simp only [← c76446, ← c76447, k2, ← c76448] at e_1242_7
  have hr := e_1242_7
  simp only [e_1242_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11342)) := by
  have c76449 := (publicBatchWrapper86_copies2389 a h).2.1
  have c76450 := (publicBatchWrapper86_copies2389 a h).2.2.1
  have c76451 := (publicBatchWrapper86_copies2389 a h).2.2.2.1
  have c76452 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.1
  have c76453 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.1
  have c76454 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_8
  simp only [← c76449, ← c76450, ← c76451] at e_1242_8
  have e_1242_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_9
  simp only [← c76452, ← c76453, k2, ← c76454] at e_1242_9
  have hr := e_1242_9
  simp only [e_1242_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11343)) := by
  have c76455 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.1
  have c76456 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.1
  have c76457 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.1
  have c76458 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76459 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76460 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_10
  simp only [← c76455, ← c76456, ← c76457] at e_1242_10
  have e_1242_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_11
  simp only [← c76458, ← c76459, k2, ← c76460] at e_1242_11
  have hr := e_1242_11
  simp only [e_1242_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11344)) := by
  have c76461 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76462 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76463 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76464 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76465 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76466 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_12
  simp only [← c76461, ← c76462, ← c76463] at e_1242_12
  have e_1242_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_13
  simp only [← c76464, ← c76465, k2, ← c76466] at e_1242_13
  have hr := e_1242_13
  simp only [e_1242_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11345)) := by
  have c76467 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76468 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76469 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76470 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76471 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76472 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_14
  simp only [← c76467, ← c76468, ← c76469] at e_1242_14
  have e_1242_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_15
  simp only [← c76470, ← c76471, k2, ← c76472] at e_1242_15
  have hr := e_1242_15
  simp only [e_1242_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11346)) := by
  have c76473 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76474 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76475 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76476 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76477 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76478 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_16
  simp only [← c76473, ← c76474, ← c76475] at e_1242_16
  have e_1242_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_17
  simp only [← c76476, ← c76477, k2, ← c76478] at e_1242_17
  have hr := e_1242_17
  simp only [e_1242_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1242 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11347)) := by
  have c76479 := (publicBatchWrapper86_copies2389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76480 := (publicBatchWrapper86_copies2390 a h).1
  have c76481 := (publicBatchWrapper86_copies2390 a h).2.1
  have c76482 := (publicBatchWrapper86_copies2390 a h).2.2.1
  have c76483 := (publicBatchWrapper86_copies2390 a h).2.2.2.1
  have c76484 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1242_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_18
  simp only [← c76479, ← c76480, ← c76481] at e_1242_18
  have e_1242_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1242 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1242_19
  simp only [← c76482, ← c76483, k2, ← c76484] at e_1242_19
  have hr := e_1242_19
  simp only [e_1242_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11348)) := by
  have c76485 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.1
  have c76486 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.1
  have c76487 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.1
  have c76488 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.1
  have c76489 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.1
  have c76490 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_0
  simp only [← c76485, ← c76486, ← c76487] at e_1243_0
  have e_1243_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_1
  simp only [← c76488, ← c76489, k2, ← c76490] at e_1243_1
  have hr := e_1243_1
  simp only [e_1243_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11349)) := by
  have c76491 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76492 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76493 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76494 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76495 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76496 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_2
  simp only [← c76491, ← c76492, ← c76493] at e_1243_2
  have e_1243_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_3
  simp only [← c76494, ← c76495, k2, ← c76496] at e_1243_3
  have hr := e_1243_3
  simp only [e_1243_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11350)) := by
  have c76497 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76498 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76499 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76500 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76501 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76502 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_4
  simp only [← c76497, ← c76498, ← c76499] at e_1243_4
  have e_1243_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_5
  simp only [← c76500, ← c76501, k2, ← c76502] at e_1243_5
  have hr := e_1243_5
  simp only [e_1243_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11351)) := by
  have c76503 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76504 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76505 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76506 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76507 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76508 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_6
  simp only [← c76503, ← c76504, ← c76505] at e_1243_6
  have e_1243_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_7
  simp only [← c76506, ← c76507, k2, ← c76508] at e_1243_7
  have hr := e_1243_7
  simp only [e_1243_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11352)) := by
  have c76509 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76510 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76511 := (publicBatchWrapper86_copies2390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76512 := (publicBatchWrapper86_copies2391 a h).1
  have c76513 := (publicBatchWrapper86_copies2391 a h).2.1
  have c76514 := (publicBatchWrapper86_copies2391 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_8
  simp only [← c76509, ← c76510, ← c76511] at e_1243_8
  have e_1243_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_9
  simp only [← c76512, ← c76513, k2, ← c76514] at e_1243_9
  have hr := e_1243_9
  simp only [e_1243_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11353)) := by
  have c76515 := (publicBatchWrapper86_copies2391 a h).2.2.2.1
  have c76516 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.1
  have c76517 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.1
  have c76518 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.1
  have c76519 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.1
  have c76520 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_10
  simp only [← c76515, ← c76516, ← c76517] at e_1243_10
  have e_1243_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_11
  simp only [← c76518, ← c76519, k2, ← c76520] at e_1243_11
  have hr := e_1243_11
  simp only [e_1243_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11354)) := by
  have c76521 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.1
  have c76522 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76523 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76524 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76525 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76526 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_12
  simp only [← c76521, ← c76522, ← c76523] at e_1243_12
  have e_1243_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_13
  simp only [← c76524, ← c76525, k2, ← c76526] at e_1243_13
  have hr := e_1243_13
  simp only [e_1243_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11355)) := by
  have c76527 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76528 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76529 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76530 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76531 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76532 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_14
  simp only [← c76527, ← c76528, ← c76529] at e_1243_14
  have e_1243_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_15
  simp only [← c76530, ← c76531, k2, ← c76532] at e_1243_15
  have hr := e_1243_15
  simp only [e_1243_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11356)) := by
  have c76533 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76534 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76535 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76536 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76537 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76538 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_16
  simp only [← c76533, ← c76534, ← c76535] at e_1243_16
  have e_1243_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_17
  simp only [← c76536, ← c76537, k2, ← c76538] at e_1243_17
  have hr := e_1243_17
  simp only [e_1243_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1243 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11357)) := by
  have c76539 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76540 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76541 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76542 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76543 := (publicBatchWrapper86_copies2391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76544 := (publicBatchWrapper86_copies2392 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1243_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_18
  simp only [← c76539, ← c76540, ← c76541] at e_1243_18
  have e_1243_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1243 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1243_19
  simp only [← c76542, ← c76543, k2, ← c76544] at e_1243_19
  have hr := e_1243_19
  simp only [e_1243_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11358)) := by
  have c76545 := (publicBatchWrapper86_copies2392 a h).2.1
  have c76546 := (publicBatchWrapper86_copies2392 a h).2.2.1
  have c76547 := (publicBatchWrapper86_copies2392 a h).2.2.2.1
  have c76548 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.1
  have c76549 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.1
  have c76550 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_0
  simp only [← c76545, ← c76546, ← c76547] at e_1244_0
  have e_1244_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_1
  simp only [← c76548, ← c76549, k2, ← c76550] at e_1244_1
  have hr := e_1244_1
  simp only [e_1244_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11359)) := by
  have c76551 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.1
  have c76552 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.1
  have c76553 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.1
  have c76554 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76555 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76556 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_2
  simp only [← c76551, ← c76552, ← c76553] at e_1244_2
  have e_1244_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_3
  simp only [← c76554, ← c76555, k2, ← c76556] at e_1244_3
  have hr := e_1244_3
  simp only [e_1244_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 23) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11360)) := by
  have c76557 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76558 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76559 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76560 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76561 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76562 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_4
  simp only [← c76557, ← c76558, ← c76559] at e_1244_4
  have e_1244_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_5
  simp only [← c76560, ← c76561, k2, ← c76562] at e_1244_5
  have hr := e_1244_5
  simp only [e_1244_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 31) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11361)) := by
  have c76563 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76564 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76565 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76566 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76567 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76568 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_6
  simp only [← c76563, ← c76564, ← c76565] at e_1244_6
  have e_1244_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_7
  simp only [← c76566, ← c76567, k2, ← c76568] at e_1244_7
  have hr := e_1244_7
  simp only [e_1244_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 39) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11362)) := by
  have c76569 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76570 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76571 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76572 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76573 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76574 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_8
  simp only [← c76569, ← c76570, ← c76571] at e_1244_8
  have e_1244_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_9
  simp only [← c76572, ← c76573, k2, ← c76574] at e_1244_9
  have hr := e_1244_9
  simp only [e_1244_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 47) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11363)) := by
  have c76575 := (publicBatchWrapper86_copies2392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76576 := (publicBatchWrapper86_copies2393 a h).1
  have c76577 := (publicBatchWrapper86_copies2393 a h).2.1
  have c76578 := (publicBatchWrapper86_copies2393 a h).2.2.1
  have c76579 := (publicBatchWrapper86_copies2393 a h).2.2.2.1
  have c76580 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_10
  simp only [← c76575, ← c76576, ← c76577] at e_1244_10
  have e_1244_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_11
  simp only [← c76578, ← c76579, k2, ← c76580] at e_1244_11
  have hr := e_1244_11
  simp only [e_1244_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 55) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11364)) := by
  have c76581 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.1
  have c76582 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.1
  have c76583 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.1
  have c76584 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.1
  have c76585 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.1
  have c76586 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_12
  simp only [← c76581, ← c76582, ← c76583] at e_1244_12
  have e_1244_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_13
  simp only [← c76584, ← c76585, k2, ← c76586] at e_1244_13
  have hr := e_1244_13
  simp only [e_1244_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 63) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11365)) := by
  have c76587 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76588 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76589 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76590 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76591 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76592 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_14
  simp only [← c76587, ← c76588, ← c76589] at e_1244_14
  have e_1244_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_15
  simp only [← c76590, ← c76591, k2, ← c76592] at e_1244_15
  have hr := e_1244_15
  simp only [e_1244_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 71) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11366)) := by
  have c76593 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76594 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76595 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76596 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76597 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76598 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_16
  simp only [← c76593, ← c76594, ← c76595] at e_1244_16
  have e_1244_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_17
  simp only [← c76596, ← c76597, k2, ← c76598] at e_1244_17
  have hr := e_1244_17
  simp only [e_1244_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1244 79) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11367)) := by
  have c76599 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76600 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76601 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76602 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76603 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76604 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1244_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_18
  simp only [← c76599, ← c76600, ← c76601] at e_1244_18
  have e_1244_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1244 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1244_19
  simp only [← c76602, ← c76603, k2, ← c76604] at e_1244_19
  have hr := e_1244_19
  simp only [e_1244_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 7) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11368)) := by
  have c76605 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76606 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76607 := (publicBatchWrapper86_copies2393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76608 := (publicBatchWrapper86_copies2394 a h).1
  have c76609 := (publicBatchWrapper86_copies2394 a h).2.1
  have c76610 := (publicBatchWrapper86_copies2394 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_0
  simp only [← c76605, ← c76606, ← c76607] at e_1245_0
  have e_1245_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_1
  simp only [← c76608, ← c76609, k2, ← c76610] at e_1245_1
  have hr := e_1245_1
  simp only [e_1245_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 15) = bselect (a (.wire 56 11)) (a (.virt 200)) (a (.virt 11369)) := by
  have c76611 := (publicBatchWrapper86_copies2394 a h).2.2.2.1
  have c76612 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.1
  have c76613 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.1
  have c76614 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.1
  have c76615 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.1
  have c76616 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_2
  simp only [← c76611, ← c76612, ← c76613] at e_1245_2
  have e_1245_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_3
  simp only [← c76614, ← c76615, k2, ← c76616] at e_1245_3
  have hr := e_1245_3
  simp only [e_1245_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11410)) := by
  have c76617 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.1
  have c76618 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76619 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76620 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76621 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76622 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_4
  simp only [← c76617, ← c76618, ← c76619] at e_1245_4
  have e_1245_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_5
  simp only [← c76620, ← c76621, k2, ← c76622] at e_1245_5
  have hr := e_1245_5
  simp only [e_1245_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11411)) := by
  have c76623 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76624 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76625 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76626 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76627 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76628 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_6
  simp only [← c76623, ← c76624, ← c76625] at e_1245_6
  have e_1245_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_7
  simp only [← c76626, ← c76627, k2, ← c76628] at e_1245_7
  have hr := e_1245_7
  simp only [e_1245_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11412)) := by
  have c76629 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76630 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76631 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76632 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76633 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76634 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_8
  simp only [← c76629, ← c76630, ← c76631] at e_1245_8
  have e_1245_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_9
  simp only [← c76632, ← c76633, k2, ← c76634] at e_1245_9
  have hr := e_1245_9
  simp only [e_1245_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11413)) := by
  have c76635 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76636 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76637 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76638 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76639 := (publicBatchWrapper86_copies2394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76640 := (publicBatchWrapper86_copies2395 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_10
  simp only [← c76635, ← c76636, ← c76637] at e_1245_10
  have e_1245_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_11
  simp only [← c76638, ← c76639, k2, ← c76640] at e_1245_11
  have hr := e_1245_11
  simp only [e_1245_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11414)) := by
  have c76641 := (publicBatchWrapper86_copies2395 a h).2.1
  have c76642 := (publicBatchWrapper86_copies2395 a h).2.2.1
  have c76643 := (publicBatchWrapper86_copies2395 a h).2.2.2.1
  have c76644 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.1
  have c76645 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.1
  have c76646 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_12
  simp only [← c76641, ← c76642, ← c76643] at e_1245_12
  have e_1245_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_13
  simp only [← c76644, ← c76645, k2, ← c76646] at e_1245_13
  have hr := e_1245_13
  simp only [e_1245_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11415)) := by
  have c76647 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.1
  have c76648 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.1
  have c76649 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.1
  have c76650 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76651 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76652 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_14
  simp only [← c76647, ← c76648, ← c76649] at e_1245_14
  have e_1245_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_15
  simp only [← c76650, ← c76651, k2, ← c76652] at e_1245_15
  have hr := e_1245_15
  simp only [e_1245_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11416)) := by
  have c76653 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76654 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76655 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76656 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76657 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76658 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_16
  simp only [← c76653, ← c76654, ← c76655] at e_1245_16
  have e_1245_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_17
  simp only [← c76656, ← c76657, k2, ← c76658] at e_1245_17
  have hr := e_1245_17
  simp only [e_1245_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1245 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11417)) := by
  have c76659 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76660 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76661 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76662 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76663 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76664 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1245_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_18
  simp only [← c76659, ← c76660, ← c76661] at e_1245_18
  have e_1245_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1245 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1245_19
  simp only [← c76662, ← c76663, k2, ← c76664] at e_1245_19
  have hr := e_1245_19
  simp only [e_1245_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11418)) := by
  have c76665 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76666 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76667 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76668 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76669 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76670 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_0
  simp only [← c76665, ← c76666, ← c76667] at e_1246_0
  have e_1246_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_1
  simp only [← c76668, ← c76669, k2, ← c76670] at e_1246_1
  have hr := e_1246_1
  simp only [e_1246_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11419)) := by
  have c76671 := (publicBatchWrapper86_copies2395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76672 := (publicBatchWrapper86_copies2396 a h).1
  have c76673 := (publicBatchWrapper86_copies2396 a h).2.1
  have c76674 := (publicBatchWrapper86_copies2396 a h).2.2.1
  have c76675 := (publicBatchWrapper86_copies2396 a h).2.2.2.1
  have c76676 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_2
  simp only [← c76671, ← c76672, ← c76673] at e_1246_2
  have e_1246_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_3
  simp only [← c76674, ← c76675, k2, ← c76676] at e_1246_3
  have hr := e_1246_3
  simp only [e_1246_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11420)) := by
  have c76677 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.1
  have c76678 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.1
  have c76679 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.1
  have c76680 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.1
  have c76681 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.1
  have c76682 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_4
  simp only [← c76677, ← c76678, ← c76679] at e_1246_4
  have e_1246_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_5
  simp only [← c76680, ← c76681, k2, ← c76682] at e_1246_5
  have hr := e_1246_5
  simp only [e_1246_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11421)) := by
  have c76683 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76684 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76685 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76686 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76687 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76688 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_6
  simp only [← c76683, ← c76684, ← c76685] at e_1246_6
  have e_1246_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_7
  simp only [← c76686, ← c76687, k2, ← c76688] at e_1246_7
  have hr := e_1246_7
  simp only [e_1246_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11422)) := by
  have c76689 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76690 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76691 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76692 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76693 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76694 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_8
  simp only [← c76689, ← c76690, ← c76691] at e_1246_8
  have e_1246_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_9
  simp only [← c76692, ← c76693, k2, ← c76694] at e_1246_9
  have hr := e_1246_9
  simp only [e_1246_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11423)) := by
  have c76695 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76696 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76697 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76698 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76699 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76700 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_10
  simp only [← c76695, ← c76696, ← c76697] at e_1246_10
  have e_1246_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_11
  simp only [← c76698, ← c76699, k2, ← c76700] at e_1246_11
  have hr := e_1246_11
  simp only [e_1246_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11424)) := by
  have c76701 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76702 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76703 := (publicBatchWrapper86_copies2396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76704 := (publicBatchWrapper86_copies2397 a h).1
  have c76705 := (publicBatchWrapper86_copies2397 a h).2.1
  have c76706 := (publicBatchWrapper86_copies2397 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_12
  simp only [← c76701, ← c76702, ← c76703] at e_1246_12
  have e_1246_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_13
  simp only [← c76704, ← c76705, k2, ← c76706] at e_1246_13
  have hr := e_1246_13
  simp only [e_1246_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11425)) := by
  have c76707 := (publicBatchWrapper86_copies2397 a h).2.2.2.1
  have c76708 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.1
  have c76709 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.1
  have c76710 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.1
  have c76711 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.1
  have c76712 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_14
  simp only [← c76707, ← c76708, ← c76709] at e_1246_14
  have e_1246_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_15
  simp only [← c76710, ← c76711, k2, ← c76712] at e_1246_15
  have hr := e_1246_15
  simp only [e_1246_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11426)) := by
  have c76713 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.1
  have c76714 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76715 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76716 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76717 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76718 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_16
  simp only [← c76713, ← c76714, ← c76715] at e_1246_16
  have e_1246_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_17
  simp only [← c76716, ← c76717, k2, ← c76718] at e_1246_17
  have hr := e_1246_17
  simp only [e_1246_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1246 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11427)) := by
  have c76719 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76720 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76721 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76722 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76723 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76724 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1246_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_18
  simp only [← c76719, ← c76720, ← c76721] at e_1246_18
  have e_1246_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1246 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1246_19
  simp only [← c76722, ← c76723, k2, ← c76724] at e_1246_19
  have hr := e_1246_19
  simp only [e_1246_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11428)) := by
  have c76725 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76726 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76727 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76728 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76729 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76730 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_0
  simp only [← c76725, ← c76726, ← c76727] at e_1247_0
  have e_1247_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_1
  simp only [← c76728, ← c76729, k2, ← c76730] at e_1247_1
  have hr := e_1247_1
  simp only [e_1247_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11429)) := by
  have c76731 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76732 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76733 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76734 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76735 := (publicBatchWrapper86_copies2397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76736 := (publicBatchWrapper86_copies2398 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_2
  simp only [← c76731, ← c76732, ← c76733] at e_1247_2
  have e_1247_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_3
  simp only [← c76734, ← c76735, k2, ← c76736] at e_1247_3
  have hr := e_1247_3
  simp only [e_1247_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11430)) := by
  have c76737 := (publicBatchWrapper86_copies2398 a h).2.1
  have c76738 := (publicBatchWrapper86_copies2398 a h).2.2.1
  have c76739 := (publicBatchWrapper86_copies2398 a h).2.2.2.1
  have c76740 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.1
  have c76741 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.1
  have c76742 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_4
  simp only [← c76737, ← c76738, ← c76739] at e_1247_4
  have e_1247_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_5
  simp only [← c76740, ← c76741, k2, ← c76742] at e_1247_5
  have hr := e_1247_5
  simp only [e_1247_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11431)) := by
  have c76743 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.1
  have c76744 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.1
  have c76745 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.1
  have c76746 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76747 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76748 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_6
  simp only [← c76743, ← c76744, ← c76745] at e_1247_6
  have e_1247_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_7
  simp only [← c76746, ← c76747, k2, ← c76748] at e_1247_7
  have hr := e_1247_7
  simp only [e_1247_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11432)) := by
  have c76749 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76750 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76751 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76752 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76753 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76754 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_8
  simp only [← c76749, ← c76750, ← c76751] at e_1247_8
  have e_1247_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_9
  simp only [← c76752, ← c76753, k2, ← c76754] at e_1247_9
  have hr := e_1247_9
  simp only [e_1247_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11433)) := by
  have c76755 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76756 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76757 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76758 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76759 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76760 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_10
  simp only [← c76755, ← c76756, ← c76757] at e_1247_10
  have e_1247_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_11
  simp only [← c76758, ← c76759, k2, ← c76760] at e_1247_11
  have hr := e_1247_11
  simp only [e_1247_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11434)) := by
  have c76761 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76762 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76763 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76764 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76765 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76766 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_12
  simp only [← c76761, ← c76762, ← c76763] at e_1247_12
  have e_1247_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_13
  simp only [← c76764, ← c76765, k2, ← c76766] at e_1247_13
  have hr := e_1247_13
  simp only [e_1247_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11435)) := by
  have c76767 := (publicBatchWrapper86_copies2398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76768 := (publicBatchWrapper86_copies2399 a h).1
  have c76769 := (publicBatchWrapper86_copies2399 a h).2.1
  have c76770 := (publicBatchWrapper86_copies2399 a h).2.2.1
  have c76771 := (publicBatchWrapper86_copies2399 a h).2.2.2.1
  have c76772 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_14
  simp only [← c76767, ← c76768, ← c76769] at e_1247_14
  have e_1247_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_15
  simp only [← c76770, ← c76771, k2, ← c76772] at e_1247_15
  have hr := e_1247_15
  simp only [e_1247_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11436)) := by
  have c76773 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.1
  have c76774 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.1
  have c76775 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.1
  have c76776 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.1
  have c76777 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.1
  have c76778 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_16
  simp only [← c76773, ← c76774, ← c76775] at e_1247_16
  have e_1247_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_17
  simp only [← c76776, ← c76777, k2, ← c76778] at e_1247_17
  have hr := e_1247_17
  simp only [e_1247_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1247 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11437)) := by
  have c76779 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76780 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76781 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76782 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76783 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76784 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1247_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_18
  simp only [← c76779, ← c76780, ← c76781] at e_1247_18
  have e_1247_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1247 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1247_19
  simp only [← c76782, ← c76783, k2, ← c76784] at e_1247_19
  have hr := e_1247_19
  simp only [e_1247_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11438)) := by
  have c76785 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76786 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76787 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76788 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76789 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76790 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_0
  simp only [← c76785, ← c76786, ← c76787] at e_1248_0
  have e_1248_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_1
  simp only [← c76788, ← c76789, k2, ← c76790] at e_1248_1
  have hr := e_1248_1
  simp only [e_1248_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11439)) := by
  have c76791 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76792 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76793 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76794 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76795 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76796 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_2
  simp only [← c76791, ← c76792, ← c76793] at e_1248_2
  have e_1248_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_3
  simp only [← c76794, ← c76795, k2, ← c76796] at e_1248_3
  have hr := e_1248_3
  simp only [e_1248_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11440)) := by
  have c76797 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76798 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76799 := (publicBatchWrapper86_copies2399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76800 := (publicBatchWrapper86_copies2400 a h).1
  have c76801 := (publicBatchWrapper86_copies2400 a h).2.1
  have c76802 := (publicBatchWrapper86_copies2400 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_4
  simp only [← c76797, ← c76798, ← c76799] at e_1248_4
  have e_1248_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_5
  simp only [← c76800, ← c76801, k2, ← c76802] at e_1248_5
  have hr := e_1248_5
  simp only [e_1248_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11441)) := by
  have c76803 := (publicBatchWrapper86_copies2400 a h).2.2.2.1
  have c76804 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.1
  have c76805 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.1
  have c76806 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.1
  have c76807 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.1
  have c76808 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_6
  simp only [← c76803, ← c76804, ← c76805] at e_1248_6
  have e_1248_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_7
  simp only [← c76806, ← c76807, k2, ← c76808] at e_1248_7
  have hr := e_1248_7
  simp only [e_1248_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11442)) := by
  have c76809 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.1
  have c76810 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76811 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76812 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76813 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76814 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_8
  simp only [← c76809, ← c76810, ← c76811] at e_1248_8
  have e_1248_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_9
  simp only [← c76812, ← c76813, k2, ← c76814] at e_1248_9
  have hr := e_1248_9
  simp only [e_1248_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11443)) := by
  have c76815 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76816 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76817 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76818 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76819 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76820 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_10
  simp only [← c76815, ← c76816, ← c76817] at e_1248_10
  have e_1248_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_11
  simp only [← c76818, ← c76819, k2, ← c76820] at e_1248_11
  have hr := e_1248_11
  simp only [e_1248_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11444)) := by
  have c76821 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76822 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76823 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76824 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76825 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76826 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_12
  simp only [← c76821, ← c76822, ← c76823] at e_1248_12
  have e_1248_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_13
  simp only [← c76824, ← c76825, k2, ← c76826] at e_1248_13
  have hr := e_1248_13
  simp only [e_1248_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11445)) := by
  have c76827 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76828 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76829 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76830 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76831 := (publicBatchWrapper86_copies2400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76832 := (publicBatchWrapper86_copies2401 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_14
  simp only [← c76827, ← c76828, ← c76829] at e_1248_14
  have e_1248_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_15
  simp only [← c76830, ← c76831, k2, ← c76832] at e_1248_15
  have hr := e_1248_15
  simp only [e_1248_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11446)) := by
  have c76833 := (publicBatchWrapper86_copies2401 a h).2.1
  have c76834 := (publicBatchWrapper86_copies2401 a h).2.2.1
  have c76835 := (publicBatchWrapper86_copies2401 a h).2.2.2.1
  have c76836 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.1
  have c76837 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.1
  have c76838 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_16
  simp only [← c76833, ← c76834, ← c76835] at e_1248_16
  have e_1248_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_17
  simp only [← c76836, ← c76837, k2, ← c76838] at e_1248_17
  have hr := e_1248_17
  simp only [e_1248_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1248 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11447)) := by
  have c76839 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.1
  have c76840 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.1
  have c76841 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.1
  have c76842 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76843 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76844 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1248_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_18
  simp only [← c76839, ← c76840, ← c76841] at e_1248_18
  have e_1248_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1248 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1248_19
  simp only [← c76842, ← c76843, k2, ← c76844] at e_1248_19
  have hr := e_1248_19
  simp only [e_1248_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11448)) := by
  have c76845 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76846 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76847 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76848 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76849 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76850 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_0
  simp only [← c76845, ← c76846, ← c76847] at e_1249_0
  have e_1249_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_1
  simp only [← c76848, ← c76849, k2, ← c76850] at e_1249_1
  have hr := e_1249_1
  simp only [e_1249_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11449)) := by
  have c76851 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76852 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76853 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76854 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76855 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76856 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_2
  simp only [← c76851, ← c76852, ← c76853] at e_1249_2
  have e_1249_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_3
  simp only [← c76854, ← c76855, k2, ← c76856] at e_1249_3
  have hr := e_1249_3
  simp only [e_1249_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11450)) := by
  have c76857 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76858 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76859 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76860 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76861 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76862 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_4
  simp only [← c76857, ← c76858, ← c76859] at e_1249_4
  have e_1249_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_5
  simp only [← c76860, ← c76861, k2, ← c76862] at e_1249_5
  have hr := e_1249_5
  simp only [e_1249_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11451)) := by
  have c76863 := (publicBatchWrapper86_copies2401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76864 := (publicBatchWrapper86_copies2402 a h).1
  have c76865 := (publicBatchWrapper86_copies2402 a h).2.1
  have c76866 := (publicBatchWrapper86_copies2402 a h).2.2.1
  have c76867 := (publicBatchWrapper86_copies2402 a h).2.2.2.1
  have c76868 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_6
  simp only [← c76863, ← c76864, ← c76865] at e_1249_6
  have e_1249_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_7
  simp only [← c76866, ← c76867, k2, ← c76868] at e_1249_7
  have hr := e_1249_7
  simp only [e_1249_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11452)) := by
  have c76869 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.1
  have c76870 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.1
  have c76871 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.1
  have c76872 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.1
  have c76873 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.1
  have c76874 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_8
  simp only [← c76869, ← c76870, ← c76871] at e_1249_8
  have e_1249_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_9
  simp only [← c76872, ← c76873, k2, ← c76874] at e_1249_9
  have hr := e_1249_9
  simp only [e_1249_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11453)) := by
  have c76875 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76876 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76877 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76878 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76879 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76880 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_10
  simp only [← c76875, ← c76876, ← c76877] at e_1249_10
  have e_1249_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_11
  simp only [← c76878, ← c76879, k2, ← c76880] at e_1249_11
  have hr := e_1249_11
  simp only [e_1249_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11454)) := by
  have c76881 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76882 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76883 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76884 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76885 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76886 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_12
  simp only [← c76881, ← c76882, ← c76883] at e_1249_12
  have e_1249_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_13
  simp only [← c76884, ← c76885, k2, ← c76886] at e_1249_13
  have hr := e_1249_13
  simp only [e_1249_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11455)) := by
  have c76887 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76888 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76889 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76890 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76891 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76892 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_14
  simp only [← c76887, ← c76888, ← c76889] at e_1249_14
  have e_1249_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_15
  simp only [← c76890, ← c76891, k2, ← c76892] at e_1249_15
  have hr := e_1249_15
  simp only [e_1249_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11456)) := by
  have c76893 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76894 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76895 := (publicBatchWrapper86_copies2402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76896 := (publicBatchWrapper86_copies2403 a h).1
  have c76897 := (publicBatchWrapper86_copies2403 a h).2.1
  have c76898 := (publicBatchWrapper86_copies2403 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_16
  simp only [← c76893, ← c76894, ← c76895] at e_1249_16
  have e_1249_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_17
  simp only [← c76896, ← c76897, k2, ← c76898] at e_1249_17
  have hr := e_1249_17
  simp only [e_1249_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1249 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11457)) := by
  have c76899 := (publicBatchWrapper86_copies2403 a h).2.2.2.1
  have c76900 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.1
  have c76901 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.1
  have c76902 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.1
  have c76903 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.1
  have c76904 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1249_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_18
  simp only [← c76899, ← c76900, ← c76901] at e_1249_18
  have e_1249_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1249 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1249_19
  simp only [← c76902, ← c76903, k2, ← c76904] at e_1249_19
  have hr := e_1249_19
  simp only [e_1249_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11458)) := by
  have c76905 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.1
  have c76906 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76907 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76908 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76909 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76910 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_0
  simp only [← c76905, ← c76906, ← c76907] at e_1250_0
  have e_1250_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_1
  simp only [← c76908, ← c76909, k2, ← c76910] at e_1250_1
  have hr := e_1250_1
  simp only [e_1250_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11459)) := by
  have c76911 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76912 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76913 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76914 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76915 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76916 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_2
  simp only [← c76911, ← c76912, ← c76913] at e_1250_2
  have e_1250_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_3
  simp only [← c76914, ← c76915, k2, ← c76916] at e_1250_3
  have hr := e_1250_3
  simp only [e_1250_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11460)) := by
  have c76917 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76918 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76919 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76920 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76921 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76922 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_4
  simp only [← c76917, ← c76918, ← c76919] at e_1250_4
  have e_1250_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_5
  simp only [← c76920, ← c76921, k2, ← c76922] at e_1250_5
  have hr := e_1250_5
  simp only [e_1250_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11461)) := by
  have c76923 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76924 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76925 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76926 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76927 := (publicBatchWrapper86_copies2403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76928 := (publicBatchWrapper86_copies2404 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_6
  simp only [← c76923, ← c76924, ← c76925] at e_1250_6
  have e_1250_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_7
  simp only [← c76926, ← c76927, k2, ← c76928] at e_1250_7
  have hr := e_1250_7
  simp only [e_1250_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11462)) := by
  have c76929 := (publicBatchWrapper86_copies2404 a h).2.1
  have c76930 := (publicBatchWrapper86_copies2404 a h).2.2.1
  have c76931 := (publicBatchWrapper86_copies2404 a h).2.2.2.1
  have c76932 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.1
  have c76933 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.1
  have c76934 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_8
  simp only [← c76929, ← c76930, ← c76931] at e_1250_8
  have e_1250_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_9
  simp only [← c76932, ← c76933, k2, ← c76934] at e_1250_9
  have hr := e_1250_9
  simp only [e_1250_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11463)) := by
  have c76935 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.1
  have c76936 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.1
  have c76937 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.1
  have c76938 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.1
  have c76939 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76940 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_10
  simp only [← c76935, ← c76936, ← c76937] at e_1250_10
  have e_1250_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_11
  simp only [← c76938, ← c76939, k2, ← c76940] at e_1250_11
  have hr := e_1250_11
  simp only [e_1250_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 55) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11464)) := by
  have c76941 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76942 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76943 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76944 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76945 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76946 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_12
  simp only [← c76941, ← c76942, ← c76943] at e_1250_12
  have e_1250_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_13
  simp only [← c76944, ← c76945, k2, ← c76946] at e_1250_13
  have hr := e_1250_13
  simp only [e_1250_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 63) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11465)) := by
  have c76947 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76948 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76949 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76950 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76951 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76952 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_14
  simp only [← c76947, ← c76948, ← c76949] at e_1250_14
  have e_1250_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_15
  simp only [← c76950, ← c76951, k2, ← c76952] at e_1250_15
  have hr := e_1250_15
  simp only [e_1250_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 71) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11466)) := by
  have c76953 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76954 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76955 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76956 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76957 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76958 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_16
  simp only [← c76953, ← c76954, ← c76955] at e_1250_16
  have e_1250_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_17
  simp only [← c76956, ← c76957, k2, ← c76958] at e_1250_17
  have hr := e_1250_17
  simp only [e_1250_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1250 79) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11467)) := by
  have c76959 := (publicBatchWrapper86_copies2404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76960 := (publicBatchWrapper86_copies2405 a h).1
  have c76961 := (publicBatchWrapper86_copies2405 a h).2.1
  have c76962 := (publicBatchWrapper86_copies2405 a h).2.2.1
  have c76963 := (publicBatchWrapper86_copies2405 a h).2.2.2.1
  have c76964 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1250_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_18
  simp only [← c76959, ← c76960, ← c76961] at e_1250_18
  have e_1250_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1250 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1250_19
  simp only [← c76962, ← c76963, k2, ← c76964] at e_1250_19
  have hr := e_1250_19
  simp only [e_1250_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 7) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11468)) := by
  have c76965 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.1
  have c76966 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.1
  have c76967 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.1
  have c76968 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.1
  have c76969 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.1
  have c76970 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_0
  simp only [← c76965, ← c76966, ← c76967] at e_1251_0
  have e_1251_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_1
  simp only [← c76968, ← c76969, k2, ← c76970] at e_1251_1
  have hr := e_1251_1
  simp only [e_1251_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 15) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11469)) := by
  have c76971 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c76972 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76973 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76974 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76975 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76976 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_2
  simp only [← c76971, ← c76972, ← c76973] at e_1251_2
  have e_1251_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_3
  simp only [← c76974, ← c76975, k2, ← c76976] at e_1251_3
  have hr := e_1251_3
  simp only [e_1251_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 23) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11470)) := by
  have c76977 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76978 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76979 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76980 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76981 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76982 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_4
  simp only [← c76977, ← c76978, ← c76979] at e_1251_4
  have e_1251_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_5
  simp only [← c76980, ← c76981, k2, ← c76982] at e_1251_5
  have hr := e_1251_5
  simp only [e_1251_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 31) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11471)) := by
  have c76983 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76984 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76985 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76986 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76987 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76988 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_6
  simp only [← c76983, ← c76984, ← c76985] at e_1251_6
  have e_1251_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_7
  simp only [← c76986, ← c76987, k2, ← c76988] at e_1251_7
  have hr := e_1251_7
  simp only [e_1251_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 39) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11472)) := by
  have c76989 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76990 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c76991 := (publicBatchWrapper86_copies2405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c76992 := (publicBatchWrapper86_copies2406 a h).1
  have c76993 := (publicBatchWrapper86_copies2406 a h).2.1
  have c76994 := (publicBatchWrapper86_copies2406 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_8
  simp only [← c76989, ← c76990, ← c76991] at e_1251_8
  have e_1251_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_9
  simp only [← c76992, ← c76993, k2, ← c76994] at e_1251_9
  have hr := e_1251_9
  simp only [e_1251_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f12107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1251 47) = bselect (a (.wire 56 55)) (a (.virt 200)) (a (.virt 11473)) := by
  have c76995 := (publicBatchWrapper86_copies2406 a h).2.2.2.1
  have c76996 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.1
  have c76997 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.1
  have c76998 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.1
  have c76999 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.1
  have c77000 := (publicBatchWrapper86_copies2406 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1251_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_10
  simp only [← c76995, ← c76996, ← c76997] at e_1251_10
  have e_1251_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1251 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1251_11
  simp only [← c76998, ← c76999, k2, ← c77000] at e_1251_11
  have hr := e_1251_11
  simp only [e_1251_10] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
