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

theorem publicBatchWrapper89_f19318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12974)) := by
  have c120414 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120415 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120416 := (publicBatchWrapper89_copies3763 a h).1
  have c120417 := (publicBatchWrapper89_copies3763 a h).2.1
  have c120418 := (publicBatchWrapper89_copies3763 a h).2.2.1
  have c120419 := (publicBatchWrapper89_copies3763 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_14 := arithEq_of_rows h (row := 1973) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_14
  simp only [← c120414, ← c120415, ← c120416] at e_1973_14
  have e_1973_15 := arithEq_of_rows h (row := 1973) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_15
  simp only [← c120417, ← c120418, k2, ← c120419] at e_1973_15
  have hr := e_1973_15
  simp only [e_1973_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12975)) := by
  have c120420 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.1
  have c120421 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.1
  have c120422 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.1
  have c120423 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.1
  have c120424 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.1
  have c120425 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_16 := arithEq_of_rows h (row := 1973) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_16
  simp only [← c120420, ← c120421, ← c120422] at e_1973_16
  have e_1973_17 := arithEq_of_rows h (row := 1973) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_17
  simp only [← c120423, ← c120424, k2, ← c120425] at e_1973_17
  have hr := e_1973_17
  simp only [e_1973_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12976)) := by
  have c120426 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120427 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120428 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120429 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120430 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120431 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_18 := arithEq_of_rows h (row := 1973) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_18
  simp only [← c120426, ← c120427, ← c120428] at e_1973_18
  have e_1973_19 := arithEq_of_rows h (row := 1973) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_19
  simp only [← c120429, ← c120430, k2, ← c120431] at e_1973_19
  have hr := e_1973_19
  simp only [e_1973_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12977)) := by
  have c120432 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120433 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120434 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120435 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120436 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120437 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_0 := arithEq_of_rows h (row := 1974) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_0
  simp only [← c120432, ← c120433, ← c120434] at e_1974_0
  have e_1974_1 := arithEq_of_rows h (row := 1974) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_1
  simp only [← c120435, ← c120436, k2, ← c120437] at e_1974_1
  have hr := e_1974_1
  simp only [e_1974_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12978)) := by
  have c120438 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120439 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120440 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120441 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120442 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120443 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_2 := arithEq_of_rows h (row := 1974) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_2
  simp only [← c120438, ← c120439, ← c120440] at e_1974_2
  have e_1974_3 := arithEq_of_rows h (row := 1974) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_3
  simp only [← c120441, ← c120442, k2, ← c120443] at e_1974_3
  have hr := e_1974_3
  simp only [e_1974_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12979)) := by
  have c120444 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120445 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120446 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120447 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120448 := (publicBatchWrapper89_copies3764 a h).1
  have c120449 := (publicBatchWrapper89_copies3764 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_4 := arithEq_of_rows h (row := 1974) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_4
  simp only [← c120444, ← c120445, ← c120446] at e_1974_4
  have e_1974_5 := arithEq_of_rows h (row := 1974) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_5
  simp only [← c120447, ← c120448, k2, ← c120449] at e_1974_5
  have hr := e_1974_5
  simp only [e_1974_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12980)) := by
  have c120450 := (publicBatchWrapper89_copies3764 a h).2.2.1
  have c120451 := (publicBatchWrapper89_copies3764 a h).2.2.2.1
  have c120452 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.1
  have c120453 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.1
  have c120454 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.1
  have c120455 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_6 := arithEq_of_rows h (row := 1974) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_6
  simp only [← c120450, ← c120451, ← c120452] at e_1974_6
  have e_1974_7 := arithEq_of_rows h (row := 1974) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_7
  simp only [← c120453, ← c120454, k2, ← c120455] at e_1974_7
  have hr := e_1974_7
  simp only [e_1974_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12981)) := by
  have c120456 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.1
  have c120457 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.1
  have c120458 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120459 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120460 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120461 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_8 := arithEq_of_rows h (row := 1974) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_8
  simp only [← c120456, ← c120457, ← c120458] at e_1974_8
  have e_1974_9 := arithEq_of_rows h (row := 1974) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_9
  simp only [← c120459, ← c120460, k2, ← c120461] at e_1974_9
  have hr := e_1974_9
  simp only [e_1974_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12982)) := by
  have c120462 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120463 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120464 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120465 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120466 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120467 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_10 := arithEq_of_rows h (row := 1974) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_10
  simp only [← c120462, ← c120463, ← c120464] at e_1974_10
  have e_1974_11 := arithEq_of_rows h (row := 1974) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_11
  simp only [← c120465, ← c120466, k2, ← c120467] at e_1974_11
  have hr := e_1974_11
  simp only [e_1974_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12983)) := by
  have c120468 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120469 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120470 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120471 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120472 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120473 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_12 := arithEq_of_rows h (row := 1974) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_12
  simp only [← c120468, ← c120469, ← c120470] at e_1974_12
  have e_1974_13 := arithEq_of_rows h (row := 1974) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_13
  simp only [← c120471, ← c120472, k2, ← c120473] at e_1974_13
  have hr := e_1974_13
  simp only [e_1974_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12984)) := by
  have c120474 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120475 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120476 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120477 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120478 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120479 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_14 := arithEq_of_rows h (row := 1974) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_14
  simp only [← c120474, ← c120475, ← c120476] at e_1974_14
  have e_1974_15 := arithEq_of_rows h (row := 1974) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_15
  simp only [← c120477, ← c120478, k2, ← c120479] at e_1974_15
  have hr := e_1974_15
  simp only [e_1974_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12985)) := by
  have c120480 := (publicBatchWrapper89_copies3765 a h).1
  have c120481 := (publicBatchWrapper89_copies3765 a h).2.1
  have c120482 := (publicBatchWrapper89_copies3765 a h).2.2.1
  have c120483 := (publicBatchWrapper89_copies3765 a h).2.2.2.1
  have c120484 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.1
  have c120485 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_16 := arithEq_of_rows h (row := 1974) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_16
  simp only [← c120480, ← c120481, ← c120482] at e_1974_16
  have e_1974_17 := arithEq_of_rows h (row := 1974) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_17
  simp only [← c120483, ← c120484, k2, ← c120485] at e_1974_17
  have hr := e_1974_17
  simp only [e_1974_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12986)) := by
  have c120486 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.1
  have c120487 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.1
  have c120488 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.1
  have c120489 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.1
  have c120490 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120491 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_18 := arithEq_of_rows h (row := 1974) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_18
  simp only [← c120486, ← c120487, ← c120488] at e_1974_18
  have e_1974_19 := arithEq_of_rows h (row := 1974) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_19
  simp only [← c120489, ← c120490, k2, ← c120491] at e_1974_19
  have hr := e_1974_19
  simp only [e_1974_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12987)) := by
  have c120492 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120493 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120494 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120495 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120496 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120497 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_0 := arithEq_of_rows h (row := 1975) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_0
  simp only [← c120492, ← c120493, ← c120494] at e_1975_0
  have e_1975_1 := arithEq_of_rows h (row := 1975) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_1
  simp only [← c120495, ← c120496, k2, ← c120497] at e_1975_1
  have hr := e_1975_1
  simp only [e_1975_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12988)) := by
  have c120498 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120499 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120500 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120501 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120502 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120503 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_2 := arithEq_of_rows h (row := 1975) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_2
  simp only [← c120498, ← c120499, ← c120500] at e_1975_2
  have e_1975_3 := arithEq_of_rows h (row := 1975) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_3
  simp only [← c120501, ← c120502, k2, ← c120503] at e_1975_3
  have hr := e_1975_3
  simp only [e_1975_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12989)) := by
  have c120504 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120505 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120506 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120507 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120508 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120509 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_4 := arithEq_of_rows h (row := 1975) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_4
  simp only [← c120504, ← c120505, ← c120506] at e_1975_4
  have e_1975_5 := arithEq_of_rows h (row := 1975) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_5
  simp only [← c120507, ← c120508, k2, ← c120509] at e_1975_5
  have hr := e_1975_5
  simp only [e_1975_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12990)) := by
  have c120510 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120511 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120512 := (publicBatchWrapper89_copies3766 a h).1
  have c120513 := (publicBatchWrapper89_copies3766 a h).2.1
  have c120514 := (publicBatchWrapper89_copies3766 a h).2.2.1
  have c120515 := (publicBatchWrapper89_copies3766 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_6 := arithEq_of_rows h (row := 1975) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_6
  simp only [← c120510, ← c120511, ← c120512] at e_1975_6
  have e_1975_7 := arithEq_of_rows h (row := 1975) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_7
  simp only [← c120513, ← c120514, k2, ← c120515] at e_1975_7
  have hr := e_1975_7
  simp only [e_1975_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12991)) := by
  have c120516 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.1
  have c120517 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.1
  have c120518 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.1
  have c120519 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.1
  have c120520 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.1
  have c120521 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_8 := arithEq_of_rows h (row := 1975) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_8
  simp only [← c120516, ← c120517, ← c120518] at e_1975_8
  have e_1975_9 := arithEq_of_rows h (row := 1975) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_9
  simp only [← c120519, ← c120520, k2, ← c120521] at e_1975_9
  have hr := e_1975_9
  simp only [e_1975_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12992)) := by
  have c120522 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120523 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120524 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120525 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120526 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120527 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_10 := arithEq_of_rows h (row := 1975) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_10
  simp only [← c120522, ← c120523, ← c120524] at e_1975_10
  have e_1975_11 := arithEq_of_rows h (row := 1975) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_11
  simp only [← c120525, ← c120526, k2, ← c120527] at e_1975_11
  have hr := e_1975_11
  simp only [e_1975_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12993)) := by
  have c120528 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120529 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120530 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120531 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120532 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120533 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_12 := arithEq_of_rows h (row := 1975) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_12
  simp only [← c120528, ← c120529, ← c120530] at e_1975_12
  have e_1975_13 := arithEq_of_rows h (row := 1975) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_13
  simp only [← c120531, ← c120532, k2, ← c120533] at e_1975_13
  have hr := e_1975_13
  simp only [e_1975_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12994)) := by
  have c120534 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120535 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120536 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120537 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120538 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120539 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_14 := arithEq_of_rows h (row := 1975) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_14
  simp only [← c120534, ← c120535, ← c120536] at e_1975_14
  have e_1975_15 := arithEq_of_rows h (row := 1975) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_15
  simp only [← c120537, ← c120538, k2, ← c120539] at e_1975_15
  have hr := e_1975_15
  simp only [e_1975_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12995)) := by
  have c120540 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120541 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120542 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120543 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120544 := (publicBatchWrapper89_copies3767 a h).1
  have c120545 := (publicBatchWrapper89_copies3767 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_16 := arithEq_of_rows h (row := 1975) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_16
  simp only [← c120540, ← c120541, ← c120542] at e_1975_16
  have e_1975_17 := arithEq_of_rows h (row := 1975) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_17
  simp only [← c120543, ← c120544, k2, ← c120545] at e_1975_17
  have hr := e_1975_17
  simp only [e_1975_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12996)) := by
  have c120546 := (publicBatchWrapper89_copies3767 a h).2.2.1
  have c120547 := (publicBatchWrapper89_copies3767 a h).2.2.2.1
  have c120548 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.1
  have c120549 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.1
  have c120550 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.1
  have c120551 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_18 := arithEq_of_rows h (row := 1975) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_18
  simp only [← c120546, ← c120547, ← c120548] at e_1975_18
  have e_1975_19 := arithEq_of_rows h (row := 1975) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_19
  simp only [← c120549, ← c120550, k2, ← c120551] at e_1975_19
  have hr := e_1975_19
  simp only [e_1975_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12997)) := by
  have c120552 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.1
  have c120553 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.1
  have c120554 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120555 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120556 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120557 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_0 := arithEq_of_rows h (row := 1976) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_0
  simp only [← c120552, ← c120553, ← c120554] at e_1976_0
  have e_1976_1 := arithEq_of_rows h (row := 1976) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_1
  simp only [← c120555, ← c120556, k2, ← c120557] at e_1976_1
  have hr := e_1976_1
  simp only [e_1976_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12998)) := by
  have c120558 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120559 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120560 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120561 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120562 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120563 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_2 := arithEq_of_rows h (row := 1976) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_2
  simp only [← c120558, ← c120559, ← c120560] at e_1976_2
  have e_1976_3 := arithEq_of_rows h (row := 1976) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_3
  simp only [← c120561, ← c120562, k2, ← c120563] at e_1976_3
  have hr := e_1976_3
  simp only [e_1976_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12999)) := by
  have c120564 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120565 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120566 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120567 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120568 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120569 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_4 := arithEq_of_rows h (row := 1976) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_4
  simp only [← c120564, ← c120565, ← c120566] at e_1976_4
  have e_1976_5 := arithEq_of_rows h (row := 1976) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_5
  simp only [← c120567, ← c120568, k2, ← c120569] at e_1976_5
  have hr := e_1976_5
  simp only [e_1976_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 13000)) := by
  have c120570 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120571 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120572 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120573 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120574 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120575 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_6 := arithEq_of_rows h (row := 1976) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_6
  simp only [← c120570, ← c120571, ← c120572] at e_1976_6
  have e_1976_7 := arithEq_of_rows h (row := 1976) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_7
  simp only [← c120573, ← c120574, k2, ← c120575] at e_1976_7
  have hr := e_1976_7
  simp only [e_1976_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 13001)) := by
  have c120576 := (publicBatchWrapper89_copies3768 a h).1
  have c120577 := (publicBatchWrapper89_copies3768 a h).2.1
  have c120578 := (publicBatchWrapper89_copies3768 a h).2.2.1
  have c120579 := (publicBatchWrapper89_copies3768 a h).2.2.2.1
  have c120580 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.1
  have c120581 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_8 := arithEq_of_rows h (row := 1976) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_8
  simp only [← c120576, ← c120577, ← c120578] at e_1976_8
  have e_1976_9 := arithEq_of_rows h (row := 1976) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_9
  simp only [← c120579, ← c120580, k2, ← c120581] at e_1976_9
  have hr := e_1976_9
  simp only [e_1976_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13170)) := by
  have c120582 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.1
  have c120583 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.1
  have c120584 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.1
  have c120585 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.1
  have c120586 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120587 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_10 := arithEq_of_rows h (row := 1976) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_10
  simp only [← c120582, ← c120583, ← c120584] at e_1976_10
  have e_1976_11 := arithEq_of_rows h (row := 1976) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_11
  simp only [← c120585, ← c120586, k2, ← c120587] at e_1976_11
  have hr := e_1976_11
  simp only [e_1976_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13171)) := by
  have c120588 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120589 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120590 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120591 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120592 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120593 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_12 := arithEq_of_rows h (row := 1976) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_12
  simp only [← c120588, ← c120589, ← c120590] at e_1976_12
  have e_1976_13 := arithEq_of_rows h (row := 1976) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_13
  simp only [← c120591, ← c120592, k2, ← c120593] at e_1976_13
  have hr := e_1976_13
  simp only [e_1976_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13172)) := by
  have c120594 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120595 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120596 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120597 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120598 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120599 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_14 := arithEq_of_rows h (row := 1976) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_14
  simp only [← c120594, ← c120595, ← c120596] at e_1976_14
  have e_1976_15 := arithEq_of_rows h (row := 1976) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_15
  simp only [← c120597, ← c120598, k2, ← c120599] at e_1976_15
  have hr := e_1976_15
  simp only [e_1976_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13173)) := by
  have c120600 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120601 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120602 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120603 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120604 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120605 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_16 := arithEq_of_rows h (row := 1976) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_16
  simp only [← c120600, ← c120601, ← c120602] at e_1976_16
  have e_1976_17 := arithEq_of_rows h (row := 1976) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_17
  simp only [← c120603, ← c120604, k2, ← c120605] at e_1976_17
  have hr := e_1976_17
  simp only [e_1976_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13174)) := by
  have c120606 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120607 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120608 := (publicBatchWrapper89_copies3769 a h).1
  have c120609 := (publicBatchWrapper89_copies3769 a h).2.1
  have c120610 := (publicBatchWrapper89_copies3769 a h).2.2.1
  have c120611 := (publicBatchWrapper89_copies3769 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_18 := arithEq_of_rows h (row := 1976) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_18
  simp only [← c120606, ← c120607, ← c120608] at e_1976_18
  have e_1976_19 := arithEq_of_rows h (row := 1976) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_19
  simp only [← c120609, ← c120610, k2, ← c120611] at e_1976_19
  have hr := e_1976_19
  simp only [e_1976_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13175)) := by
  have c120612 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.1
  have c120613 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.1
  have c120614 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.1
  have c120615 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.1
  have c120616 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.1
  have c120617 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_0 := arithEq_of_rows h (row := 1977) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_0
  simp only [← c120612, ← c120613, ← c120614] at e_1977_0
  have e_1977_1 := arithEq_of_rows h (row := 1977) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_1
  simp only [← c120615, ← c120616, k2, ← c120617] at e_1977_1
  have hr := e_1977_1
  simp only [e_1977_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13176)) := by
  have c120618 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120619 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120620 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120621 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120622 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120623 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_2 := arithEq_of_rows h (row := 1977) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_2
  simp only [← c120618, ← c120619, ← c120620] at e_1977_2
  have e_1977_3 := arithEq_of_rows h (row := 1977) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_3
  simp only [← c120621, ← c120622, k2, ← c120623] at e_1977_3
  have hr := e_1977_3
  simp only [e_1977_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13177)) := by
  have c120624 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120625 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120626 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120627 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120628 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120629 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_4 := arithEq_of_rows h (row := 1977) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_4
  simp only [← c120624, ← c120625, ← c120626] at e_1977_4
  have e_1977_5 := arithEq_of_rows h (row := 1977) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_5
  simp only [← c120627, ← c120628, k2, ← c120629] at e_1977_5
  have hr := e_1977_5
  simp only [e_1977_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13178)) := by
  have c120630 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120631 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120632 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120633 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120634 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120635 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_6 := arithEq_of_rows h (row := 1977) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_6
  simp only [← c120630, ← c120631, ← c120632] at e_1977_6
  have e_1977_7 := arithEq_of_rows h (row := 1977) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_7
  simp only [← c120633, ← c120634, k2, ← c120635] at e_1977_7
  have hr := e_1977_7
  simp only [e_1977_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13179)) := by
  have c120636 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120637 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120638 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120639 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120640 := (publicBatchWrapper89_copies3770 a h).1
  have c120641 := (publicBatchWrapper89_copies3770 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_8 := arithEq_of_rows h (row := 1977) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_8
  simp only [← c120636, ← c120637, ← c120638] at e_1977_8
  have e_1977_9 := arithEq_of_rows h (row := 1977) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_9
  simp only [← c120639, ← c120640, k2, ← c120641] at e_1977_9
  have hr := e_1977_9
  simp only [e_1977_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13180)) := by
  have c120642 := (publicBatchWrapper89_copies3770 a h).2.2.1
  have c120643 := (publicBatchWrapper89_copies3770 a h).2.2.2.1
  have c120644 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.1
  have c120645 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.1
  have c120646 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.1
  have c120647 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_10 := arithEq_of_rows h (row := 1977) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_10
  simp only [← c120642, ← c120643, ← c120644] at e_1977_10
  have e_1977_11 := arithEq_of_rows h (row := 1977) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_11
  simp only [← c120645, ← c120646, k2, ← c120647] at e_1977_11
  have hr := e_1977_11
  simp only [e_1977_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13181)) := by
  have c120648 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.1
  have c120649 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.1
  have c120650 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120651 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120652 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120653 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_12 := arithEq_of_rows h (row := 1977) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_12
  simp only [← c120648, ← c120649, ← c120650] at e_1977_12
  have e_1977_13 := arithEq_of_rows h (row := 1977) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_13
  simp only [← c120651, ← c120652, k2, ← c120653] at e_1977_13
  have hr := e_1977_13
  simp only [e_1977_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13182)) := by
  have c120654 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120655 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120656 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120657 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120658 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120659 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_14 := arithEq_of_rows h (row := 1977) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_14
  simp only [← c120654, ← c120655, ← c120656] at e_1977_14
  have e_1977_15 := arithEq_of_rows h (row := 1977) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_15
  simp only [← c120657, ← c120658, k2, ← c120659] at e_1977_15
  have hr := e_1977_15
  simp only [e_1977_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13183)) := by
  have c120660 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120661 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120662 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120663 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120664 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120665 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_16 := arithEq_of_rows h (row := 1977) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_16
  simp only [← c120660, ← c120661, ← c120662] at e_1977_16
  have e_1977_17 := arithEq_of_rows h (row := 1977) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_17
  simp only [← c120663, ← c120664, k2, ← c120665] at e_1977_17
  have hr := e_1977_17
  simp only [e_1977_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13184)) := by
  have c120666 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120667 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120668 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120669 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120670 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120671 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_18 := arithEq_of_rows h (row := 1977) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_18
  simp only [← c120666, ← c120667, ← c120668] at e_1977_18
  have e_1977_19 := arithEq_of_rows h (row := 1977) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_19
  simp only [← c120669, ← c120670, k2, ← c120671] at e_1977_19
  have hr := e_1977_19
  simp only [e_1977_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13185)) := by
  have c120672 := (publicBatchWrapper89_copies3771 a h).1
  have c120673 := (publicBatchWrapper89_copies3771 a h).2.1
  have c120674 := (publicBatchWrapper89_copies3771 a h).2.2.1
  have c120675 := (publicBatchWrapper89_copies3771 a h).2.2.2.1
  have c120676 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.1
  have c120677 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_0 := arithEq_of_rows h (row := 1978) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_0
  simp only [← c120672, ← c120673, ← c120674] at e_1978_0
  have e_1978_1 := arithEq_of_rows h (row := 1978) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_1
  simp only [← c120675, ← c120676, k2, ← c120677] at e_1978_1
  have hr := e_1978_1
  simp only [e_1978_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13186)) := by
  have c120678 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.1
  have c120679 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.1
  have c120680 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.1
  have c120681 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.1
  have c120682 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120683 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_2 := arithEq_of_rows h (row := 1978) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_2
  simp only [← c120678, ← c120679, ← c120680] at e_1978_2
  have e_1978_3 := arithEq_of_rows h (row := 1978) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_3
  simp only [← c120681, ← c120682, k2, ← c120683] at e_1978_3
  have hr := e_1978_3
  simp only [e_1978_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13187)) := by
  have c120684 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120685 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120686 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120687 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120688 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120689 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_4 := arithEq_of_rows h (row := 1978) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_4
  simp only [← c120684, ← c120685, ← c120686] at e_1978_4
  have e_1978_5 := arithEq_of_rows h (row := 1978) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_5
  simp only [← c120687, ← c120688, k2, ← c120689] at e_1978_5
  have hr := e_1978_5
  simp only [e_1978_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13188)) := by
  have c120690 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120691 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120692 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120693 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120694 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120695 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_6 := arithEq_of_rows h (row := 1978) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_6
  simp only [← c120690, ← c120691, ← c120692] at e_1978_6
  have e_1978_7 := arithEq_of_rows h (row := 1978) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_7
  simp only [← c120693, ← c120694, k2, ← c120695] at e_1978_7
  have hr := e_1978_7
  simp only [e_1978_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13189)) := by
  have c120696 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120697 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120698 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120699 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120700 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120701 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_8 := arithEq_of_rows h (row := 1978) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_8
  simp only [← c120696, ← c120697, ← c120698] at e_1978_8
  have e_1978_9 := arithEq_of_rows h (row := 1978) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_9
  simp only [← c120699, ← c120700, k2, ← c120701] at e_1978_9
  have hr := e_1978_9
  simp only [e_1978_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13190)) := by
  have c120702 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120703 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120704 := (publicBatchWrapper89_copies3772 a h).1
  have c120705 := (publicBatchWrapper89_copies3772 a h).2.1
  have c120706 := (publicBatchWrapper89_copies3772 a h).2.2.1
  have c120707 := (publicBatchWrapper89_copies3772 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_10 := arithEq_of_rows h (row := 1978) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_10
  simp only [← c120702, ← c120703, ← c120704] at e_1978_10
  have e_1978_11 := arithEq_of_rows h (row := 1978) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_11
  simp only [← c120705, ← c120706, k2, ← c120707] at e_1978_11
  have hr := e_1978_11
  simp only [e_1978_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13191)) := by
  have c120708 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.1
  have c120709 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.1
  have c120710 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.1
  have c120711 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.1
  have c120712 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.1
  have c120713 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_12 := arithEq_of_rows h (row := 1978) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_12
  simp only [← c120708, ← c120709, ← c120710] at e_1978_12
  have e_1978_13 := arithEq_of_rows h (row := 1978) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_13
  simp only [← c120711, ← c120712, k2, ← c120713] at e_1978_13
  have hr := e_1978_13
  simp only [e_1978_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13192)) := by
  have c120714 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120715 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120716 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120717 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120718 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120719 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_14 := arithEq_of_rows h (row := 1978) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_14
  simp only [← c120714, ← c120715, ← c120716] at e_1978_14
  have e_1978_15 := arithEq_of_rows h (row := 1978) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_15
  simp only [← c120717, ← c120718, k2, ← c120719] at e_1978_15
  have hr := e_1978_15
  simp only [e_1978_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13193)) := by
  have c120720 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120721 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120722 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120723 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120724 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120725 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_16 := arithEq_of_rows h (row := 1978) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_16
  simp only [← c120720, ← c120721, ← c120722] at e_1978_16
  have e_1978_17 := arithEq_of_rows h (row := 1978) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_17
  simp only [← c120723, ← c120724, k2, ← c120725] at e_1978_17
  have hr := e_1978_17
  simp only [e_1978_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13194)) := by
  have c120726 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120727 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120728 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120729 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120730 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120731 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_18 := arithEq_of_rows h (row := 1978) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_18
  simp only [← c120726, ← c120727, ← c120728] at e_1978_18
  have e_1978_19 := arithEq_of_rows h (row := 1978) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_19
  simp only [← c120729, ← c120730, k2, ← c120731] at e_1978_19
  have hr := e_1978_19
  simp only [e_1978_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13195)) := by
  have c120732 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120733 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120734 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120735 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120736 := (publicBatchWrapper89_copies3773 a h).1
  have c120737 := (publicBatchWrapper89_copies3773 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_0 := arithEq_of_rows h (row := 1979) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_0
  simp only [← c120732, ← c120733, ← c120734] at e_1979_0
  have e_1979_1 := arithEq_of_rows h (row := 1979) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_1
  simp only [← c120735, ← c120736, k2, ← c120737] at e_1979_1
  have hr := e_1979_1
  simp only [e_1979_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13196)) := by
  have c120738 := (publicBatchWrapper89_copies3773 a h).2.2.1
  have c120739 := (publicBatchWrapper89_copies3773 a h).2.2.2.1
  have c120740 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.1
  have c120741 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.1
  have c120742 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.1
  have c120743 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_2 := arithEq_of_rows h (row := 1979) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_2
  simp only [← c120738, ← c120739, ← c120740] at e_1979_2
  have e_1979_3 := arithEq_of_rows h (row := 1979) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_3
  simp only [← c120741, ← c120742, k2, ← c120743] at e_1979_3
  have hr := e_1979_3
  simp only [e_1979_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13197)) := by
  have c120744 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.1
  have c120745 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.1
  have c120746 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120747 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120748 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120749 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_4 := arithEq_of_rows h (row := 1979) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_4
  simp only [← c120744, ← c120745, ← c120746] at e_1979_4
  have e_1979_5 := arithEq_of_rows h (row := 1979) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_5
  simp only [← c120747, ← c120748, k2, ← c120749] at e_1979_5
  have hr := e_1979_5
  simp only [e_1979_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13198)) := by
  have c120750 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120751 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120752 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120753 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120754 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120755 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_6 := arithEq_of_rows h (row := 1979) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_6
  simp only [← c120750, ← c120751, ← c120752] at e_1979_6
  have e_1979_7 := arithEq_of_rows h (row := 1979) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_7
  simp only [← c120753, ← c120754, k2, ← c120755] at e_1979_7
  have hr := e_1979_7
  simp only [e_1979_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13199)) := by
  have c120756 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120757 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120758 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120759 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120760 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120761 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_8 := arithEq_of_rows h (row := 1979) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_8
  simp only [← c120756, ← c120757, ← c120758] at e_1979_8
  have e_1979_9 := arithEq_of_rows h (row := 1979) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_9
  simp only [← c120759, ← c120760, k2, ← c120761] at e_1979_9
  have hr := e_1979_9
  simp only [e_1979_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13200)) := by
  have c120762 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120763 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120764 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120765 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120766 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120767 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_10 := arithEq_of_rows h (row := 1979) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_10
  simp only [← c120762, ← c120763, ← c120764] at e_1979_10
  have e_1979_11 := arithEq_of_rows h (row := 1979) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_11
  simp only [← c120765, ← c120766, k2, ← c120767] at e_1979_11
  have hr := e_1979_11
  simp only [e_1979_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13201)) := by
  have c120768 := (publicBatchWrapper89_copies3774 a h).1
  have c120769 := (publicBatchWrapper89_copies3774 a h).2.1
  have c120770 := (publicBatchWrapper89_copies3774 a h).2.2.1
  have c120771 := (publicBatchWrapper89_copies3774 a h).2.2.2.1
  have c120772 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.1
  have c120773 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_12 := arithEq_of_rows h (row := 1979) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_12
  simp only [← c120768, ← c120769, ← c120770] at e_1979_12
  have e_1979_13 := arithEq_of_rows h (row := 1979) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_13
  simp only [← c120771, ← c120772, k2, ← c120773] at e_1979_13
  have hr := e_1979_13
  simp only [e_1979_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13370)) := by
  have c120774 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.1
  have c120775 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.1
  have c120776 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.1
  have c120777 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.1
  have c120778 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120779 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_14 := arithEq_of_rows h (row := 1979) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_14
  simp only [← c120774, ← c120775, ← c120776] at e_1979_14
  have e_1979_15 := arithEq_of_rows h (row := 1979) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_15
  simp only [← c120777, ← c120778, k2, ← c120779] at e_1979_15
  have hr := e_1979_15
  simp only [e_1979_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13371)) := by
  have c120780 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120781 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120782 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120783 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120784 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120785 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_16 := arithEq_of_rows h (row := 1979) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_16
  simp only [← c120780, ← c120781, ← c120782] at e_1979_16
  have e_1979_17 := arithEq_of_rows h (row := 1979) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_17
  simp only [← c120783, ← c120784, k2, ← c120785] at e_1979_17
  have hr := e_1979_17
  simp only [e_1979_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13372)) := by
  have c120786 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120787 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120788 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120789 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120790 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120791 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_18 := arithEq_of_rows h (row := 1979) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_18
  simp only [← c120786, ← c120787, ← c120788] at e_1979_18
  have e_1979_19 := arithEq_of_rows h (row := 1979) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_19
  simp only [← c120789, ← c120790, k2, ← c120791] at e_1979_19
  have hr := e_1979_19
  simp only [e_1979_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13373)) := by
  have c120792 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120793 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120794 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120795 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120796 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120797 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_0 := arithEq_of_rows h (row := 1980) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_0
  simp only [← c120792, ← c120793, ← c120794] at e_1980_0
  have e_1980_1 := arithEq_of_rows h (row := 1980) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_1
  simp only [← c120795, ← c120796, k2, ← c120797] at e_1980_1
  have hr := e_1980_1
  simp only [e_1980_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13374)) := by
  have c120798 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120799 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120800 := (publicBatchWrapper89_copies3775 a h).1
  have c120801 := (publicBatchWrapper89_copies3775 a h).2.1
  have c120802 := (publicBatchWrapper89_copies3775 a h).2.2.1
  have c120803 := (publicBatchWrapper89_copies3775 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_2 := arithEq_of_rows h (row := 1980) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_2
  simp only [← c120798, ← c120799, ← c120800] at e_1980_2
  have e_1980_3 := arithEq_of_rows h (row := 1980) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_3
  simp only [← c120801, ← c120802, k2, ← c120803] at e_1980_3
  have hr := e_1980_3
  simp only [e_1980_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13375)) := by
  have c120804 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.1
  have c120805 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.1
  have c120806 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.1
  have c120807 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.1
  have c120808 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.1
  have c120809 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_4 := arithEq_of_rows h (row := 1980) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_4
  simp only [← c120804, ← c120805, ← c120806] at e_1980_4
  have e_1980_5 := arithEq_of_rows h (row := 1980) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_5
  simp only [← c120807, ← c120808, k2, ← c120809] at e_1980_5
  have hr := e_1980_5
  simp only [e_1980_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13376)) := by
  have c120810 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120811 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120812 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120813 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120814 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120815 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_6 := arithEq_of_rows h (row := 1980) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_6
  simp only [← c120810, ← c120811, ← c120812] at e_1980_6
  have e_1980_7 := arithEq_of_rows h (row := 1980) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_7
  simp only [← c120813, ← c120814, k2, ← c120815] at e_1980_7
  have hr := e_1980_7
  simp only [e_1980_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13377)) := by
  have c120816 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120817 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120818 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120819 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120820 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120821 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_8 := arithEq_of_rows h (row := 1980) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_8
  simp only [← c120816, ← c120817, ← c120818] at e_1980_8
  have e_1980_9 := arithEq_of_rows h (row := 1980) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_9
  simp only [← c120819, ← c120820, k2, ← c120821] at e_1980_9
  have hr := e_1980_9
  simp only [e_1980_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13378)) := by
  have c120822 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120823 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120824 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120825 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120826 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120827 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_10 := arithEq_of_rows h (row := 1980) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_10
  simp only [← c120822, ← c120823, ← c120824] at e_1980_10
  have e_1980_11 := arithEq_of_rows h (row := 1980) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_11
  simp only [← c120825, ← c120826, k2, ← c120827] at e_1980_11
  have hr := e_1980_11
  simp only [e_1980_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13379)) := by
  have c120828 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120829 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120830 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120831 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120832 := (publicBatchWrapper89_copies3776 a h).1
  have c120833 := (publicBatchWrapper89_copies3776 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_12 := arithEq_of_rows h (row := 1980) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_12
  simp only [← c120828, ← c120829, ← c120830] at e_1980_12
  have e_1980_13 := arithEq_of_rows h (row := 1980) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_13
  simp only [← c120831, ← c120832, k2, ← c120833] at e_1980_13
  have hr := e_1980_13
  simp only [e_1980_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13380)) := by
  have c120834 := (publicBatchWrapper89_copies3776 a h).2.2.1
  have c120835 := (publicBatchWrapper89_copies3776 a h).2.2.2.1
  have c120836 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.1
  have c120837 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.1
  have c120838 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.1
  have c120839 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_14 := arithEq_of_rows h (row := 1980) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_14
  simp only [← c120834, ← c120835, ← c120836] at e_1980_14
  have e_1980_15 := arithEq_of_rows h (row := 1980) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_15
  simp only [← c120837, ← c120838, k2, ← c120839] at e_1980_15
  have hr := e_1980_15
  simp only [e_1980_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13381)) := by
  have c120840 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.1
  have c120841 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.1
  have c120842 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120843 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120844 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120845 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_16 := arithEq_of_rows h (row := 1980) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_16
  simp only [← c120840, ← c120841, ← c120842] at e_1980_16
  have e_1980_17 := arithEq_of_rows h (row := 1980) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_17
  simp only [← c120843, ← c120844, k2, ← c120845] at e_1980_17
  have hr := e_1980_17
  simp only [e_1980_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13382)) := by
  have c120846 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120847 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120848 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120849 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120850 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120851 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_18 := arithEq_of_rows h (row := 1980) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_18
  simp only [← c120846, ← c120847, ← c120848] at e_1980_18
  have e_1980_19 := arithEq_of_rows h (row := 1980) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_19
  simp only [← c120849, ← c120850, k2, ← c120851] at e_1980_19
  have hr := e_1980_19
  simp only [e_1980_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13383)) := by
  have c120852 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120853 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120854 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120855 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120856 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120857 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_0 := arithEq_of_rows h (row := 1981) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_0
  simp only [← c120852, ← c120853, ← c120854] at e_1981_0
  have e_1981_1 := arithEq_of_rows h (row := 1981) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_1
  simp only [← c120855, ← c120856, k2, ← c120857] at e_1981_1
  have hr := e_1981_1
  simp only [e_1981_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13384)) := by
  have c120858 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120859 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120860 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120861 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120862 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120863 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_2 := arithEq_of_rows h (row := 1981) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_2
  simp only [← c120858, ← c120859, ← c120860] at e_1981_2
  have e_1981_3 := arithEq_of_rows h (row := 1981) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_3
  simp only [← c120861, ← c120862, k2, ← c120863] at e_1981_3
  have hr := e_1981_3
  simp only [e_1981_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13385)) := by
  have c120864 := (publicBatchWrapper89_copies3777 a h).1
  have c120865 := (publicBatchWrapper89_copies3777 a h).2.1
  have c120866 := (publicBatchWrapper89_copies3777 a h).2.2.1
  have c120867 := (publicBatchWrapper89_copies3777 a h).2.2.2.1
  have c120868 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.1
  have c120869 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_4 := arithEq_of_rows h (row := 1981) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_4
  simp only [← c120864, ← c120865, ← c120866] at e_1981_4
  have e_1981_5 := arithEq_of_rows h (row := 1981) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_5
  simp only [← c120867, ← c120868, k2, ← c120869] at e_1981_5
  have hr := e_1981_5
  simp only [e_1981_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13386)) := by
  have c120870 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.1
  have c120871 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.1
  have c120872 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.1
  have c120873 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.1
  have c120874 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120875 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_6 := arithEq_of_rows h (row := 1981) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_6
  simp only [← c120870, ← c120871, ← c120872] at e_1981_6
  have e_1981_7 := arithEq_of_rows h (row := 1981) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_7
  simp only [← c120873, ← c120874, k2, ← c120875] at e_1981_7
  have hr := e_1981_7
  simp only [e_1981_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13387)) := by
  have c120876 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120877 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120878 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120879 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120880 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120881 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_8 := arithEq_of_rows h (row := 1981) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_8
  simp only [← c120876, ← c120877, ← c120878] at e_1981_8
  have e_1981_9 := arithEq_of_rows h (row := 1981) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_9
  simp only [← c120879, ← c120880, k2, ← c120881] at e_1981_9
  have hr := e_1981_9
  simp only [e_1981_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13388)) := by
  have c120882 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120883 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120884 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120885 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120886 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120887 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_10 := arithEq_of_rows h (row := 1981) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_10
  simp only [← c120882, ← c120883, ← c120884] at e_1981_10
  have e_1981_11 := arithEq_of_rows h (row := 1981) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_11
  simp only [← c120885, ← c120886, k2, ← c120887] at e_1981_11
  have hr := e_1981_11
  simp only [e_1981_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13389)) := by
  have c120888 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120889 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120890 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120891 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120892 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120893 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_12 := arithEq_of_rows h (row := 1981) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_12
  simp only [← c120888, ← c120889, ← c120890] at e_1981_12
  have e_1981_13 := arithEq_of_rows h (row := 1981) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_13
  simp only [← c120891, ← c120892, k2, ← c120893] at e_1981_13
  have hr := e_1981_13
  simp only [e_1981_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13390)) := by
  have c120894 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120895 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120896 := (publicBatchWrapper89_copies3778 a h).1
  have c120897 := (publicBatchWrapper89_copies3778 a h).2.1
  have c120898 := (publicBatchWrapper89_copies3778 a h).2.2.1
  have c120899 := (publicBatchWrapper89_copies3778 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_14 := arithEq_of_rows h (row := 1981) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_14
  simp only [← c120894, ← c120895, ← c120896] at e_1981_14
  have e_1981_15 := arithEq_of_rows h (row := 1981) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_15
  simp only [← c120897, ← c120898, k2, ← c120899] at e_1981_15
  have hr := e_1981_15
  simp only [e_1981_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13391)) := by
  have c120900 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.1
  have c120901 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.1
  have c120902 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.1
  have c120903 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.1
  have c120904 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.1
  have c120905 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_16 := arithEq_of_rows h (row := 1981) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_16
  simp only [← c120900, ← c120901, ← c120902] at e_1981_16
  have e_1981_17 := arithEq_of_rows h (row := 1981) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_17
  simp only [← c120903, ← c120904, k2, ← c120905] at e_1981_17
  have hr := e_1981_17
  simp only [e_1981_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13392)) := by
  have c120906 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120907 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120908 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120909 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120910 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120911 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_18 := arithEq_of_rows h (row := 1981) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_18
  simp only [← c120906, ← c120907, ← c120908] at e_1981_18
  have e_1981_19 := arithEq_of_rows h (row := 1981) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_19
  simp only [← c120909, ← c120910, k2, ← c120911] at e_1981_19
  have hr := e_1981_19
  simp only [e_1981_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13393)) := by
  have c120912 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120913 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120914 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120915 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120916 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120917 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_0 := arithEq_of_rows h (row := 1982) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_0
  simp only [← c120912, ← c120913, ← c120914] at e_1982_0
  have e_1982_1 := arithEq_of_rows h (row := 1982) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_1
  simp only [← c120915, ← c120916, k2, ← c120917] at e_1982_1
  have hr := e_1982_1
  simp only [e_1982_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13394)) := by
  have c120918 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120919 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120920 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120921 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120922 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120923 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_2 := arithEq_of_rows h (row := 1982) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_2
  simp only [← c120918, ← c120919, ← c120920] at e_1982_2
  have e_1982_3 := arithEq_of_rows h (row := 1982) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_3
  simp only [← c120921, ← c120922, k2, ← c120923] at e_1982_3
  have hr := e_1982_3
  simp only [e_1982_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13395)) := by
  have c120924 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120925 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120926 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120927 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120928 := (publicBatchWrapper89_copies3779 a h).1
  have c120929 := (publicBatchWrapper89_copies3779 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_4 := arithEq_of_rows h (row := 1982) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_4
  simp only [← c120924, ← c120925, ← c120926] at e_1982_4
  have e_1982_5 := arithEq_of_rows h (row := 1982) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_5
  simp only [← c120927, ← c120928, k2, ← c120929] at e_1982_5
  have hr := e_1982_5
  simp only [e_1982_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13396)) := by
  have c120930 := (publicBatchWrapper89_copies3779 a h).2.2.1
  have c120931 := (publicBatchWrapper89_copies3779 a h).2.2.2.1
  have c120932 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.1
  have c120933 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.1
  have c120934 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.1
  have c120935 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_6 := arithEq_of_rows h (row := 1982) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_6
  simp only [← c120930, ← c120931, ← c120932] at e_1982_6
  have e_1982_7 := arithEq_of_rows h (row := 1982) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_7
  simp only [← c120933, ← c120934, k2, ← c120935] at e_1982_7
  have hr := e_1982_7
  simp only [e_1982_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13397)) := by
  have c120936 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.1
  have c120937 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.1
  have c120938 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120939 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120940 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120941 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_8 := arithEq_of_rows h (row := 1982) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_8
  simp only [← c120936, ← c120937, ← c120938] at e_1982_8
  have e_1982_9 := arithEq_of_rows h (row := 1982) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_9
  simp only [← c120939, ← c120940, k2, ← c120941] at e_1982_9
  have hr := e_1982_9
  simp only [e_1982_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13398)) := by
  have c120942 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120943 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120944 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120945 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120946 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120947 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_10 := arithEq_of_rows h (row := 1982) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_10
  simp only [← c120942, ← c120943, ← c120944] at e_1982_10
  have e_1982_11 := arithEq_of_rows h (row := 1982) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_11
  simp only [← c120945, ← c120946, k2, ← c120947] at e_1982_11
  have hr := e_1982_11
  simp only [e_1982_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13399)) := by
  have c120948 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120949 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120950 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120951 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120952 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120953 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_12 := arithEq_of_rows h (row := 1982) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_12
  simp only [← c120948, ← c120949, ← c120950] at e_1982_12
  have e_1982_13 := arithEq_of_rows h (row := 1982) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_13
  simp only [← c120951, ← c120952, k2, ← c120953] at e_1982_13
  have hr := e_1982_13
  simp only [e_1982_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13400)) := by
  have c120954 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120955 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120956 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120957 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120958 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120959 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_14 := arithEq_of_rows h (row := 1982) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_14
  simp only [← c120954, ← c120955, ← c120956] at e_1982_14
  have e_1982_15 := arithEq_of_rows h (row := 1982) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_15
  simp only [← c120957, ← c120958, k2, ← c120959] at e_1982_15
  have hr := e_1982_15
  simp only [e_1982_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13401)) := by
  have c120960 := (publicBatchWrapper89_copies3780 a h).1
  have c120961 := (publicBatchWrapper89_copies3780 a h).2.1
  have c120962 := (publicBatchWrapper89_copies3780 a h).2.2.1
  have c120963 := (publicBatchWrapper89_copies3780 a h).2.2.2.1
  have c120964 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.1
  have c120965 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_16 := arithEq_of_rows h (row := 1982) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_16
  simp only [← c120960, ← c120961, ← c120962] at e_1982_16
  have e_1982_17 := arithEq_of_rows h (row := 1982) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_17
  simp only [← c120963, ← c120964, k2, ← c120965] at e_1982_17
  have hr := e_1982_17
  simp only [e_1982_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13570)) := by
  have c120966 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.1
  have c120967 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.1
  have c120968 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.1
  have c120969 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.1
  have c120970 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120971 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_18 := arithEq_of_rows h (row := 1982) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_18
  simp only [← c120966, ← c120967, ← c120968] at e_1982_18
  have e_1982_19 := arithEq_of_rows h (row := 1982) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_19
  simp only [← c120969, ← c120970, k2, ← c120971] at e_1982_19
  have hr := e_1982_19
  simp only [e_1982_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13571)) := by
  have c120972 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120973 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120974 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120975 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120976 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120977 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_0 := arithEq_of_rows h (row := 1983) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_0
  simp only [← c120972, ← c120973, ← c120974] at e_1983_0
  have e_1983_1 := arithEq_of_rows h (row := 1983) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_1
  simp only [← c120975, ← c120976, k2, ← c120977] at e_1983_1
  have hr := e_1983_1
  simp only [e_1983_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13572)) := by
  have c120978 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120979 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120980 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120981 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120982 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120983 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_2 := arithEq_of_rows h (row := 1983) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_2
  simp only [← c120978, ← c120979, ← c120980] at e_1983_2
  have e_1983_3 := arithEq_of_rows h (row := 1983) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_3
  simp only [← c120981, ← c120982, k2, ← c120983] at e_1983_3
  have hr := e_1983_3
  simp only [e_1983_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13573)) := by
  have c120984 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120985 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120986 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120987 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120988 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120989 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_4 := arithEq_of_rows h (row := 1983) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_4
  simp only [← c120984, ← c120985, ← c120986] at e_1983_4
  have e_1983_5 := arithEq_of_rows h (row := 1983) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_5
  simp only [← c120987, ← c120988, k2, ← c120989] at e_1983_5
  have hr := e_1983_5
  simp only [e_1983_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13574)) := by
  have c120990 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120991 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120992 := (publicBatchWrapper89_copies3781 a h).1
  have c120993 := (publicBatchWrapper89_copies3781 a h).2.1
  have c120994 := (publicBatchWrapper89_copies3781 a h).2.2.1
  have c120995 := (publicBatchWrapper89_copies3781 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_6 := arithEq_of_rows h (row := 1983) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_6
  simp only [← c120990, ← c120991, ← c120992] at e_1983_6
  have e_1983_7 := arithEq_of_rows h (row := 1983) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_7
  simp only [← c120993, ← c120994, k2, ← c120995] at e_1983_7
  have hr := e_1983_7
  simp only [e_1983_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13575)) := by
  have c120996 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.1
  have c120997 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.1
  have c120998 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.1
  have c120999 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.1
  have c121000 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.1
  have c121001 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_8 := arithEq_of_rows h (row := 1983) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_8
  simp only [← c120996, ← c120997, ← c120998] at e_1983_8
  have e_1983_9 := arithEq_of_rows h (row := 1983) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_9
  simp only [← c120999, ← c121000, k2, ← c121001] at e_1983_9
  have hr := e_1983_9
  simp only [e_1983_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13576)) := by
  have c121002 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121003 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121004 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121005 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121006 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121007 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_10 := arithEq_of_rows h (row := 1983) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_10
  simp only [← c121002, ← c121003, ← c121004] at e_1983_10
  have e_1983_11 := arithEq_of_rows h (row := 1983) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_11
  simp only [← c121005, ← c121006, k2, ← c121007] at e_1983_11
  have hr := e_1983_11
  simp only [e_1983_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13577)) := by
  have c121008 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121009 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121010 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121011 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121012 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121013 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_12 := arithEq_of_rows h (row := 1983) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_12
  simp only [← c121008, ← c121009, ← c121010] at e_1983_12
  have e_1983_13 := arithEq_of_rows h (row := 1983) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_13
  simp only [← c121011, ← c121012, k2, ← c121013] at e_1983_13
  have hr := e_1983_13
  simp only [e_1983_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13578)) := by
  have c121014 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121015 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121016 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121017 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121018 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121019 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_14 := arithEq_of_rows h (row := 1983) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_14
  simp only [← c121014, ← c121015, ← c121016] at e_1983_14
  have e_1983_15 := arithEq_of_rows h (row := 1983) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_15
  simp only [← c121017, ← c121018, k2, ← c121019] at e_1983_15
  have hr := e_1983_15
  simp only [e_1983_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13579)) := by
  have c121020 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121021 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121022 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121023 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121024 := (publicBatchWrapper89_copies3782 a h).1
  have c121025 := (publicBatchWrapper89_copies3782 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_16 := arithEq_of_rows h (row := 1983) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_16
  simp only [← c121020, ← c121021, ← c121022] at e_1983_16
  have e_1983_17 := arithEq_of_rows h (row := 1983) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_17
  simp only [← c121023, ← c121024, k2, ← c121025] at e_1983_17
  have hr := e_1983_17
  simp only [e_1983_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13580)) := by
  have c121026 := (publicBatchWrapper89_copies3782 a h).2.2.1
  have c121027 := (publicBatchWrapper89_copies3782 a h).2.2.2.1
  have c121028 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.1
  have c121029 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.1
  have c121030 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.1
  have c121031 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_18 := arithEq_of_rows h (row := 1983) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_18
  simp only [← c121026, ← c121027, ← c121028] at e_1983_18
  have e_1983_19 := arithEq_of_rows h (row := 1983) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_19
  simp only [← c121029, ← c121030, k2, ← c121031] at e_1983_19
  have hr := e_1983_19
  simp only [e_1983_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13581)) := by
  have c121032 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.1
  have c121033 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.1
  have c121034 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121035 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121036 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121037 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_0 := arithEq_of_rows h (row := 1984) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_0
  simp only [← c121032, ← c121033, ← c121034] at e_1984_0
  have e_1984_1 := arithEq_of_rows h (row := 1984) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_1
  simp only [← c121035, ← c121036, k2, ← c121037] at e_1984_1
  have hr := e_1984_1
  simp only [e_1984_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13582)) := by
  have c121038 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121039 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121040 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121041 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121042 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121043 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_2 := arithEq_of_rows h (row := 1984) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_2
  simp only [← c121038, ← c121039, ← c121040] at e_1984_2
  have e_1984_3 := arithEq_of_rows h (row := 1984) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_3
  simp only [← c121041, ← c121042, k2, ← c121043] at e_1984_3
  have hr := e_1984_3
  simp only [e_1984_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13583)) := by
  have c121044 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121045 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121046 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121047 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121048 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121049 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_4 := arithEq_of_rows h (row := 1984) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_4
  simp only [← c121044, ← c121045, ← c121046] at e_1984_4
  have e_1984_5 := arithEq_of_rows h (row := 1984) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_5
  simp only [← c121047, ← c121048, k2, ← c121049] at e_1984_5
  have hr := e_1984_5
  simp only [e_1984_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13584)) := by
  have c121050 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121051 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121052 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121053 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121054 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121055 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_6 := arithEq_of_rows h (row := 1984) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_6
  simp only [← c121050, ← c121051, ← c121052] at e_1984_6
  have e_1984_7 := arithEq_of_rows h (row := 1984) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_7
  simp only [← c121053, ← c121054, k2, ← c121055] at e_1984_7
  have hr := e_1984_7
  simp only [e_1984_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13585)) := by
  have c121056 := (publicBatchWrapper89_copies3783 a h).1
  have c121057 := (publicBatchWrapper89_copies3783 a h).2.1
  have c121058 := (publicBatchWrapper89_copies3783 a h).2.2.1
  have c121059 := (publicBatchWrapper89_copies3783 a h).2.2.2.1
  have c121060 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.1
  have c121061 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_8 := arithEq_of_rows h (row := 1984) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_8
  simp only [← c121056, ← c121057, ← c121058] at e_1984_8
  have e_1984_9 := arithEq_of_rows h (row := 1984) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_9
  simp only [← c121059, ← c121060, k2, ← c121061] at e_1984_9
  have hr := e_1984_9
  simp only [e_1984_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13586)) := by
  have c121062 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.1
  have c121063 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.1
  have c121064 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.1
  have c121065 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.1
  have c121066 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121067 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_10 := arithEq_of_rows h (row := 1984) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_10
  simp only [← c121062, ← c121063, ← c121064] at e_1984_10
  have e_1984_11 := arithEq_of_rows h (row := 1984) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_11
  simp only [← c121065, ← c121066, k2, ← c121067] at e_1984_11
  have hr := e_1984_11
  simp only [e_1984_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13587)) := by
  have c121068 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121069 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121070 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121071 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121072 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121073 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_12 := arithEq_of_rows h (row := 1984) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_12
  simp only [← c121068, ← c121069, ← c121070] at e_1984_12
  have e_1984_13 := arithEq_of_rows h (row := 1984) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_13
  simp only [← c121071, ← c121072, k2, ← c121073] at e_1984_13
  have hr := e_1984_13
  simp only [e_1984_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13588)) := by
  have c121074 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121075 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121076 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121077 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121078 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121079 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_14 := arithEq_of_rows h (row := 1984) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_14
  simp only [← c121074, ← c121075, ← c121076] at e_1984_14
  have e_1984_15 := arithEq_of_rows h (row := 1984) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_15
  simp only [← c121077, ← c121078, k2, ← c121079] at e_1984_15
  have hr := e_1984_15
  simp only [e_1984_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13589)) := by
  have c121080 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121081 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121082 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121083 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121084 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121085 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_16 := arithEq_of_rows h (row := 1984) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_16
  simp only [← c121080, ← c121081, ← c121082] at e_1984_16
  have e_1984_17 := arithEq_of_rows h (row := 1984) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_17
  simp only [← c121083, ← c121084, k2, ← c121085] at e_1984_17
  have hr := e_1984_17
  simp only [e_1984_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13590)) := by
  have c121086 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121087 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121088 := (publicBatchWrapper89_copies3784 a h).1
  have c121089 := (publicBatchWrapper89_copies3784 a h).2.1
  have c121090 := (publicBatchWrapper89_copies3784 a h).2.2.1
  have c121091 := (publicBatchWrapper89_copies3784 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_18 := arithEq_of_rows h (row := 1984) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_18
  simp only [← c121086, ← c121087, ← c121088] at e_1984_18
  have e_1984_19 := arithEq_of_rows h (row := 1984) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_19
  simp only [← c121089, ← c121090, k2, ← c121091] at e_1984_19
  have hr := e_1984_19
  simp only [e_1984_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13591)) := by
  have c121092 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.1
  have c121093 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.1
  have c121094 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.1
  have c121095 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.1
  have c121096 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.1
  have c121097 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_0 := arithEq_of_rows h (row := 1985) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_0
  simp only [← c121092, ← c121093, ← c121094] at e_1985_0
  have e_1985_1 := arithEq_of_rows h (row := 1985) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_1
  simp only [← c121095, ← c121096, k2, ← c121097] at e_1985_1
  have hr := e_1985_1
  simp only [e_1985_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13592)) := by
  have c121098 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121099 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121100 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121101 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121102 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121103 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_2 := arithEq_of_rows h (row := 1985) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_2
  simp only [← c121098, ← c121099, ← c121100] at e_1985_2
  have e_1985_3 := arithEq_of_rows h (row := 1985) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_3
  simp only [← c121101, ← c121102, k2, ← c121103] at e_1985_3
  have hr := e_1985_3
  simp only [e_1985_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13593)) := by
  have c121104 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121105 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121106 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121107 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121108 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121109 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_4 := arithEq_of_rows h (row := 1985) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_4
  simp only [← c121104, ← c121105, ← c121106] at e_1985_4
  have e_1985_5 := arithEq_of_rows h (row := 1985) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_5
  simp only [← c121107, ← c121108, k2, ← c121109] at e_1985_5
  have hr := e_1985_5
  simp only [e_1985_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13594)) := by
  have c121110 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121111 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121112 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121113 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121114 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121115 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_6 := arithEq_of_rows h (row := 1985) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_6
  simp only [← c121110, ← c121111, ← c121112] at e_1985_6
  have e_1985_7 := arithEq_of_rows h (row := 1985) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_7
  simp only [← c121113, ← c121114, k2, ← c121115] at e_1985_7
  have hr := e_1985_7
  simp only [e_1985_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13595)) := by
  have c121116 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121117 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121118 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121119 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121120 := (publicBatchWrapper89_copies3785 a h).1
  have c121121 := (publicBatchWrapper89_copies3785 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_8 := arithEq_of_rows h (row := 1985) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_8
  simp only [← c121116, ← c121117, ← c121118] at e_1985_8
  have e_1985_9 := arithEq_of_rows h (row := 1985) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_9
  simp only [← c121119, ← c121120, k2, ← c121121] at e_1985_9
  have hr := e_1985_9
  simp only [e_1985_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13596)) := by
  have c121122 := (publicBatchWrapper89_copies3785 a h).2.2.1
  have c121123 := (publicBatchWrapper89_copies3785 a h).2.2.2.1
  have c121124 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.1
  have c121125 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.1
  have c121126 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.1
  have c121127 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_10 := arithEq_of_rows h (row := 1985) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_10
  simp only [← c121122, ← c121123, ← c121124] at e_1985_10
  have e_1985_11 := arithEq_of_rows h (row := 1985) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_11
  simp only [← c121125, ← c121126, k2, ← c121127] at e_1985_11
  have hr := e_1985_11
  simp only [e_1985_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13597)) := by
  have c121128 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.1
  have c121129 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.1
  have c121130 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121131 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121132 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121133 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_12 := arithEq_of_rows h (row := 1985) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_12
  simp only [← c121128, ← c121129, ← c121130] at e_1985_12
  have e_1985_13 := arithEq_of_rows h (row := 1985) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_13
  simp only [← c121131, ← c121132, k2, ← c121133] at e_1985_13
  have hr := e_1985_13
  simp only [e_1985_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13598)) := by
  have c121134 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121135 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121136 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121137 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121138 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121139 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_14 := arithEq_of_rows h (row := 1985) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_14
  simp only [← c121134, ← c121135, ← c121136] at e_1985_14
  have e_1985_15 := arithEq_of_rows h (row := 1985) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_15
  simp only [← c121137, ← c121138, k2, ← c121139] at e_1985_15
  have hr := e_1985_15
  simp only [e_1985_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13599)) := by
  have c121140 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121141 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121142 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121143 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121144 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121145 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_16 := arithEq_of_rows h (row := 1985) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_16
  simp only [← c121140, ← c121141, ← c121142] at e_1985_16
  have e_1985_17 := arithEq_of_rows h (row := 1985) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_17
  simp only [← c121143, ← c121144, k2, ← c121145] at e_1985_17
  have hr := e_1985_17
  simp only [e_1985_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13600)) := by
  have c121146 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121147 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121148 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121149 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121150 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121151 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_18 := arithEq_of_rows h (row := 1985) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_18
  simp only [← c121146, ← c121147, ← c121148] at e_1985_18
  have e_1985_19 := arithEq_of_rows h (row := 1985) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_19
  simp only [← c121149, ← c121150, k2, ← c121151] at e_1985_19
  have hr := e_1985_19
  simp only [e_1985_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13601)) := by
  have c121152 := (publicBatchWrapper89_copies3786 a h).1
  have c121153 := (publicBatchWrapper89_copies3786 a h).2.1
  have c121154 := (publicBatchWrapper89_copies3786 a h).2.2.1
  have c121155 := (publicBatchWrapper89_copies3786 a h).2.2.2.1
  have c121156 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.1
  have c121157 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_0 := arithEq_of_rows h (row := 1986) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_0
  simp only [← c121152, ← c121153, ← c121154] at e_1986_0
  have e_1986_1 := arithEq_of_rows h (row := 1986) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_1
  simp only [← c121155, ← c121156, k2, ← c121157] at e_1986_1
  have hr := e_1986_1
  simp only [e_1986_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13770)) := by
  have c121158 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.1
  have c121159 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.1
  have c121160 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.1
  have c121161 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.1
  have c121162 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121163 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_2 := arithEq_of_rows h (row := 1986) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_2
  simp only [← c121158, ← c121159, ← c121160] at e_1986_2
  have e_1986_3 := arithEq_of_rows h (row := 1986) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_3
  simp only [← c121161, ← c121162, k2, ← c121163] at e_1986_3
  have hr := e_1986_3
  simp only [e_1986_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13771)) := by
  have c121164 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121165 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121166 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121167 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121168 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121169 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_4 := arithEq_of_rows h (row := 1986) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_4
  simp only [← c121164, ← c121165, ← c121166] at e_1986_4
  have e_1986_5 := arithEq_of_rows h (row := 1986) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_5
  simp only [← c121167, ← c121168, k2, ← c121169] at e_1986_5
  have hr := e_1986_5
  simp only [e_1986_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13772)) := by
  have c121170 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121171 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121172 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121173 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121174 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121175 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_6 := arithEq_of_rows h (row := 1986) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_6
  simp only [← c121170, ← c121171, ← c121172] at e_1986_6
  have e_1986_7 := arithEq_of_rows h (row := 1986) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_7
  simp only [← c121173, ← c121174, k2, ← c121175] at e_1986_7
  have hr := e_1986_7
  simp only [e_1986_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13773)) := by
  have c121176 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121177 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121178 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121179 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121180 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121181 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_8 := arithEq_of_rows h (row := 1986) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_8
  simp only [← c121176, ← c121177, ← c121178] at e_1986_8
  have e_1986_9 := arithEq_of_rows h (row := 1986) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_9
  simp only [← c121179, ← c121180, k2, ← c121181] at e_1986_9
  have hr := e_1986_9
  simp only [e_1986_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13774)) := by
  have c121182 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121183 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121184 := (publicBatchWrapper89_copies3787 a h).1
  have c121185 := (publicBatchWrapper89_copies3787 a h).2.1
  have c121186 := (publicBatchWrapper89_copies3787 a h).2.2.1
  have c121187 := (publicBatchWrapper89_copies3787 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_10 := arithEq_of_rows h (row := 1986) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_10
  simp only [← c121182, ← c121183, ← c121184] at e_1986_10
  have e_1986_11 := arithEq_of_rows h (row := 1986) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_11
  simp only [← c121185, ← c121186, k2, ← c121187] at e_1986_11
  have hr := e_1986_11
  simp only [e_1986_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13775)) := by
  have c121188 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.1
  have c121189 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.1
  have c121190 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.1
  have c121191 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.1
  have c121192 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.1
  have c121193 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_12 := arithEq_of_rows h (row := 1986) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_12
  simp only [← c121188, ← c121189, ← c121190] at e_1986_12
  have e_1986_13 := arithEq_of_rows h (row := 1986) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_13
  simp only [← c121191, ← c121192, k2, ← c121193] at e_1986_13
  have hr := e_1986_13
  simp only [e_1986_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13776)) := by
  have c121194 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121195 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121196 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121197 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121198 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121199 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_14 := arithEq_of_rows h (row := 1986) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_14
  simp only [← c121194, ← c121195, ← c121196] at e_1986_14
  have e_1986_15 := arithEq_of_rows h (row := 1986) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_15
  simp only [← c121197, ← c121198, k2, ← c121199] at e_1986_15
  have hr := e_1986_15
  simp only [e_1986_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13777)) := by
  have c121200 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121201 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121202 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121203 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121204 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121205 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_16 := arithEq_of_rows h (row := 1986) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_16
  simp only [← c121200, ← c121201, ← c121202] at e_1986_16
  have e_1986_17 := arithEq_of_rows h (row := 1986) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_17
  simp only [← c121203, ← c121204, k2, ← c121205] at e_1986_17
  have hr := e_1986_17
  simp only [e_1986_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13778)) := by
  have c121206 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121207 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121208 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121209 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121210 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121211 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_18 := arithEq_of_rows h (row := 1986) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_18
  simp only [← c121206, ← c121207, ← c121208] at e_1986_18
  have e_1986_19 := arithEq_of_rows h (row := 1986) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_19
  simp only [← c121209, ← c121210, k2, ← c121211] at e_1986_19
  have hr := e_1986_19
  simp only [e_1986_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13779)) := by
  have c121212 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121213 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121214 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121215 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121216 := (publicBatchWrapper89_copies3788 a h).1
  have c121217 := (publicBatchWrapper89_copies3788 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_0 := arithEq_of_rows h (row := 1987) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_0
  simp only [← c121212, ← c121213, ← c121214] at e_1987_0
  have e_1987_1 := arithEq_of_rows h (row := 1987) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_1
  simp only [← c121215, ← c121216, k2, ← c121217] at e_1987_1
  have hr := e_1987_1
  simp only [e_1987_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13780)) := by
  have c121218 := (publicBatchWrapper89_copies3788 a h).2.2.1
  have c121219 := (publicBatchWrapper89_copies3788 a h).2.2.2.1
  have c121220 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.1
  have c121221 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.1
  have c121222 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.1
  have c121223 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_2 := arithEq_of_rows h (row := 1987) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_2
  simp only [← c121218, ← c121219, ← c121220] at e_1987_2
  have e_1987_3 := arithEq_of_rows h (row := 1987) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_3
  simp only [← c121221, ← c121222, k2, ← c121223] at e_1987_3
  have hr := e_1987_3
  simp only [e_1987_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13781)) := by
  have c121224 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.1
  have c121225 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.1
  have c121226 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121227 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121228 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121229 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_4 := arithEq_of_rows h (row := 1987) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_4
  simp only [← c121224, ← c121225, ← c121226] at e_1987_4
  have e_1987_5 := arithEq_of_rows h (row := 1987) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_5
  simp only [← c121227, ← c121228, k2, ← c121229] at e_1987_5
  have hr := e_1987_5
  simp only [e_1987_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13782)) := by
  have c121230 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121231 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121232 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121233 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121234 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121235 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_6 := arithEq_of_rows h (row := 1987) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_6
  simp only [← c121230, ← c121231, ← c121232] at e_1987_6
  have e_1987_7 := arithEq_of_rows h (row := 1987) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_7
  simp only [← c121233, ← c121234, k2, ← c121235] at e_1987_7
  have hr := e_1987_7
  simp only [e_1987_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13783)) := by
  have c121236 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121237 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121238 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121239 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121240 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121241 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_8 := arithEq_of_rows h (row := 1987) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_8
  simp only [← c121236, ← c121237, ← c121238] at e_1987_8
  have e_1987_9 := arithEq_of_rows h (row := 1987) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_9
  simp only [← c121239, ← c121240, k2, ← c121241] at e_1987_9
  have hr := e_1987_9
  simp only [e_1987_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13784)) := by
  have c121242 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121243 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121244 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121245 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121246 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121247 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_10 := arithEq_of_rows h (row := 1987) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_10
  simp only [← c121242, ← c121243, ← c121244] at e_1987_10
  have e_1987_11 := arithEq_of_rows h (row := 1987) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_11
  simp only [← c121245, ← c121246, k2, ← c121247] at e_1987_11
  have hr := e_1987_11
  simp only [e_1987_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13785)) := by
  have c121248 := (publicBatchWrapper89_copies3789 a h).1
  have c121249 := (publicBatchWrapper89_copies3789 a h).2.1
  have c121250 := (publicBatchWrapper89_copies3789 a h).2.2.1
  have c121251 := (publicBatchWrapper89_copies3789 a h).2.2.2.1
  have c121252 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.1
  have c121253 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_12 := arithEq_of_rows h (row := 1987) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_12
  simp only [← c121248, ← c121249, ← c121250] at e_1987_12
  have e_1987_13 := arithEq_of_rows h (row := 1987) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_13
  simp only [← c121251, ← c121252, k2, ← c121253] at e_1987_13
  have hr := e_1987_13
  simp only [e_1987_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13786)) := by
  have c121254 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.1
  have c121255 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.1
  have c121256 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.1
  have c121257 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.1
  have c121258 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121259 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_14 := arithEq_of_rows h (row := 1987) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_14
  simp only [← c121254, ← c121255, ← c121256] at e_1987_14
  have e_1987_15 := arithEq_of_rows h (row := 1987) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_15
  simp only [← c121257, ← c121258, k2, ← c121259] at e_1987_15
  have hr := e_1987_15
  simp only [e_1987_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13787)) := by
  have c121260 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121261 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121262 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121263 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121264 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121265 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_16 := arithEq_of_rows h (row := 1987) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_16
  simp only [← c121260, ← c121261, ← c121262] at e_1987_16
  have e_1987_17 := arithEq_of_rows h (row := 1987) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_17
  simp only [← c121263, ← c121264, k2, ← c121265] at e_1987_17
  have hr := e_1987_17
  simp only [e_1987_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13788)) := by
  have c121266 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121267 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121268 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121269 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121270 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121271 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_18 := arithEq_of_rows h (row := 1987) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_18
  simp only [← c121266, ← c121267, ← c121268] at e_1987_18
  have e_1987_19 := arithEq_of_rows h (row := 1987) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_19
  simp only [← c121269, ← c121270, k2, ← c121271] at e_1987_19
  have hr := e_1987_19
  simp only [e_1987_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13789)) := by
  have c121272 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121273 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121274 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121275 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121276 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121277 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_0 := arithEq_of_rows h (row := 1988) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_0
  simp only [← c121272, ← c121273, ← c121274] at e_1988_0
  have e_1988_1 := arithEq_of_rows h (row := 1988) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_1
  simp only [← c121275, ← c121276, k2, ← c121277] at e_1988_1
  have hr := e_1988_1
  simp only [e_1988_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13790)) := by
  have c121278 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121279 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121280 := (publicBatchWrapper89_copies3790 a h).1
  have c121281 := (publicBatchWrapper89_copies3790 a h).2.1
  have c121282 := (publicBatchWrapper89_copies3790 a h).2.2.1
  have c121283 := (publicBatchWrapper89_copies3790 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_2 := arithEq_of_rows h (row := 1988) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_2
  simp only [← c121278, ← c121279, ← c121280] at e_1988_2
  have e_1988_3 := arithEq_of_rows h (row := 1988) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_3
  simp only [← c121281, ← c121282, k2, ← c121283] at e_1988_3
  have hr := e_1988_3
  simp only [e_1988_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13791)) := by
  have c121284 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.1
  have c121285 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.1
  have c121286 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.1
  have c121287 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.1
  have c121288 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.1
  have c121289 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_4 := arithEq_of_rows h (row := 1988) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_4
  simp only [← c121284, ← c121285, ← c121286] at e_1988_4
  have e_1988_5 := arithEq_of_rows h (row := 1988) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_5
  simp only [← c121287, ← c121288, k2, ← c121289] at e_1988_5
  have hr := e_1988_5
  simp only [e_1988_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13792)) := by
  have c121290 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121291 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121292 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121293 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121294 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121295 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_6 := arithEq_of_rows h (row := 1988) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_6
  simp only [← c121290, ← c121291, ← c121292] at e_1988_6
  have e_1988_7 := arithEq_of_rows h (row := 1988) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_7
  simp only [← c121293, ← c121294, k2, ← c121295] at e_1988_7
  have hr := e_1988_7
  simp only [e_1988_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13793)) := by
  have c121296 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121297 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121298 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121299 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121300 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121301 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_8 := arithEq_of_rows h (row := 1988) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_8
  simp only [← c121296, ← c121297, ← c121298] at e_1988_8
  have e_1988_9 := arithEq_of_rows h (row := 1988) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_9
  simp only [← c121299, ← c121300, k2, ← c121301] at e_1988_9
  have hr := e_1988_9
  simp only [e_1988_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13794)) := by
  have c121302 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121303 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121304 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121305 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121306 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121307 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_10 := arithEq_of_rows h (row := 1988) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_10
  simp only [← c121302, ← c121303, ← c121304] at e_1988_10
  have e_1988_11 := arithEq_of_rows h (row := 1988) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_11
  simp only [← c121305, ← c121306, k2, ← c121307] at e_1988_11
  have hr := e_1988_11
  simp only [e_1988_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13795)) := by
  have c121308 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121309 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121310 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121311 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121312 := (publicBatchWrapper89_copies3791 a h).1
  have c121313 := (publicBatchWrapper89_copies3791 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_12 := arithEq_of_rows h (row := 1988) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_12
  simp only [← c121308, ← c121309, ← c121310] at e_1988_12
  have e_1988_13 := arithEq_of_rows h (row := 1988) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_13
  simp only [← c121311, ← c121312, k2, ← c121313] at e_1988_13
  have hr := e_1988_13
  simp only [e_1988_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13796)) := by
  have c121314 := (publicBatchWrapper89_copies3791 a h).2.2.1
  have c121315 := (publicBatchWrapper89_copies3791 a h).2.2.2.1
  have c121316 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.1
  have c121317 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.1
  have c121318 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.1
  have c121319 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_14 := arithEq_of_rows h (row := 1988) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_14
  simp only [← c121314, ← c121315, ← c121316] at e_1988_14
  have e_1988_15 := arithEq_of_rows h (row := 1988) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_15
  simp only [← c121317, ← c121318, k2, ← c121319] at e_1988_15
  have hr := e_1988_15
  simp only [e_1988_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13797)) := by
  have c121320 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.1
  have c121321 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.1
  have c121322 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121323 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121324 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121325 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_16 := arithEq_of_rows h (row := 1988) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_16
  simp only [← c121320, ← c121321, ← c121322] at e_1988_16
  have e_1988_17 := arithEq_of_rows h (row := 1988) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_17
  simp only [← c121323, ← c121324, k2, ← c121325] at e_1988_17
  have hr := e_1988_17
  simp only [e_1988_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13798)) := by
  have c121326 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121327 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121328 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121329 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121330 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121331 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_18 := arithEq_of_rows h (row := 1988) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_18
  simp only [← c121326, ← c121327, ← c121328] at e_1988_18
  have e_1988_19 := arithEq_of_rows h (row := 1988) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_19
  simp only [← c121329, ← c121330, k2, ← c121331] at e_1988_19
  have hr := e_1988_19
  simp only [e_1988_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13799)) := by
  have c121332 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121333 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121334 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121335 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121336 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121337 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_0 := arithEq_of_rows h (row := 1989) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_0
  simp only [← c121332, ← c121333, ← c121334] at e_1989_0
  have e_1989_1 := arithEq_of_rows h (row := 1989) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_1
  simp only [← c121335, ← c121336, k2, ← c121337] at e_1989_1
  have hr := e_1989_1
  simp only [e_1989_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13800)) := by
  have c121338 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121339 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121340 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121341 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121342 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121343 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_2 := arithEq_of_rows h (row := 1989) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_2
  simp only [← c121338, ← c121339, ← c121340] at e_1989_2
  have e_1989_3 := arithEq_of_rows h (row := 1989) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_3
  simp only [← c121341, ← c121342, k2, ← c121343] at e_1989_3
  have hr := e_1989_3
  simp only [e_1989_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13801)) := by
  have c121344 := (publicBatchWrapper89_copies3792 a h).1
  have c121345 := (publicBatchWrapper89_copies3792 a h).2.1
  have c121346 := (publicBatchWrapper89_copies3792 a h).2.2.1
  have c121347 := (publicBatchWrapper89_copies3792 a h).2.2.2.1
  have c121348 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.1
  have c121349 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_4 := arithEq_of_rows h (row := 1989) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_4
  simp only [← c121344, ← c121345, ← c121346] at e_1989_4
  have e_1989_5 := arithEq_of_rows h (row := 1989) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_5
  simp only [← c121347, ← c121348, k2, ← c121349] at e_1989_5
  have hr := e_1989_5
  simp only [e_1989_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13970)) := by
  have c121350 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.1
  have c121351 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.1
  have c121352 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.1
  have c121353 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.1
  have c121354 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121355 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_6 := arithEq_of_rows h (row := 1989) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_6
  simp only [← c121350, ← c121351, ← c121352] at e_1989_6
  have e_1989_7 := arithEq_of_rows h (row := 1989) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_7
  simp only [← c121353, ← c121354, k2, ← c121355] at e_1989_7
  have hr := e_1989_7
  simp only [e_1989_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13971)) := by
  have c121356 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121357 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121358 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121359 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121360 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121361 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_8 := arithEq_of_rows h (row := 1989) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_8
  simp only [← c121356, ← c121357, ← c121358] at e_1989_8
  have e_1989_9 := arithEq_of_rows h (row := 1989) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_9
  simp only [← c121359, ← c121360, k2, ← c121361] at e_1989_9
  have hr := e_1989_9
  simp only [e_1989_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13972)) := by
  have c121362 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121363 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121364 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121365 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121366 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121367 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_10 := arithEq_of_rows h (row := 1989) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_10
  simp only [← c121362, ← c121363, ← c121364] at e_1989_10
  have e_1989_11 := arithEq_of_rows h (row := 1989) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_11
  simp only [← c121365, ← c121366, k2, ← c121367] at e_1989_11
  have hr := e_1989_11
  simp only [e_1989_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13973)) := by
  have c121368 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121369 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121370 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121371 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121372 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121373 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_12 := arithEq_of_rows h (row := 1989) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_12
  simp only [← c121368, ← c121369, ← c121370] at e_1989_12
  have e_1989_13 := arithEq_of_rows h (row := 1989) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_13
  simp only [← c121371, ← c121372, k2, ← c121373] at e_1989_13
  have hr := e_1989_13
  simp only [e_1989_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13974)) := by
  have c121374 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121375 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121376 := (publicBatchWrapper89_copies3793 a h).1
  have c121377 := (publicBatchWrapper89_copies3793 a h).2.1
  have c121378 := (publicBatchWrapper89_copies3793 a h).2.2.1
  have c121379 := (publicBatchWrapper89_copies3793 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_14 := arithEq_of_rows h (row := 1989) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_14
  simp only [← c121374, ← c121375, ← c121376] at e_1989_14
  have e_1989_15 := arithEq_of_rows h (row := 1989) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_15
  simp only [← c121377, ← c121378, k2, ← c121379] at e_1989_15
  have hr := e_1989_15
  simp only [e_1989_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13975)) := by
  have c121380 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.1
  have c121381 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.1
  have c121382 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.1
  have c121383 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.1
  have c121384 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.1
  have c121385 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_16 := arithEq_of_rows h (row := 1989) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_16
  simp only [← c121380, ← c121381, ← c121382] at e_1989_16
  have e_1989_17 := arithEq_of_rows h (row := 1989) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_17
  simp only [← c121383, ← c121384, k2, ← c121385] at e_1989_17
  have hr := e_1989_17
  simp only [e_1989_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13976)) := by
  have c121386 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121387 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121388 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121389 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121390 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121391 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_18 := arithEq_of_rows h (row := 1989) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_18
  simp only [← c121386, ← c121387, ← c121388] at e_1989_18
  have e_1989_19 := arithEq_of_rows h (row := 1989) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_19
  simp only [← c121389, ← c121390, k2, ← c121391] at e_1989_19
  have hr := e_1989_19
  simp only [e_1989_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13977)) := by
  have c121392 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121393 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121394 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121395 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121396 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121397 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_0 := arithEq_of_rows h (row := 1990) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_0
  simp only [← c121392, ← c121393, ← c121394] at e_1990_0
  have e_1990_1 := arithEq_of_rows h (row := 1990) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_1
  simp only [← c121395, ← c121396, k2, ← c121397] at e_1990_1
  have hr := e_1990_1
  simp only [e_1990_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13978)) := by
  have c121398 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121399 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121400 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121401 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121402 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121403 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_2 := arithEq_of_rows h (row := 1990) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_2
  simp only [← c121398, ← c121399, ← c121400] at e_1990_2
  have e_1990_3 := arithEq_of_rows h (row := 1990) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_3
  simp only [← c121401, ← c121402, k2, ← c121403] at e_1990_3
  have hr := e_1990_3
  simp only [e_1990_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13979)) := by
  have c121404 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121405 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121406 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121407 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121408 := (publicBatchWrapper89_copies3794 a h).1
  have c121409 := (publicBatchWrapper89_copies3794 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_4 := arithEq_of_rows h (row := 1990) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_4
  simp only [← c121404, ← c121405, ← c121406] at e_1990_4
  have e_1990_5 := arithEq_of_rows h (row := 1990) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_5
  simp only [← c121407, ← c121408, k2, ← c121409] at e_1990_5
  have hr := e_1990_5
  simp only [e_1990_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13980)) := by
  have c121410 := (publicBatchWrapper89_copies3794 a h).2.2.1
  have c121411 := (publicBatchWrapper89_copies3794 a h).2.2.2.1
  have c121412 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.1
  have c121413 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.1
  have c121414 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.1
  have c121415 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_6 := arithEq_of_rows h (row := 1990) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_6
  simp only [← c121410, ← c121411, ← c121412] at e_1990_6
  have e_1990_7 := arithEq_of_rows h (row := 1990) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_7
  simp only [← c121413, ← c121414, k2, ← c121415] at e_1990_7
  have hr := e_1990_7
  simp only [e_1990_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13981)) := by
  have c121416 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.1
  have c121417 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.1
  have c121418 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121419 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121420 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121421 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_8 := arithEq_of_rows h (row := 1990) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_8
  simp only [← c121416, ← c121417, ← c121418] at e_1990_8
  have e_1990_9 := arithEq_of_rows h (row := 1990) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_9
  simp only [← c121419, ← c121420, k2, ← c121421] at e_1990_9
  have hr := e_1990_9
  simp only [e_1990_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13982)) := by
  have c121422 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121423 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121424 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121425 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121426 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121427 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_10 := arithEq_of_rows h (row := 1990) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_10
  simp only [← c121422, ← c121423, ← c121424] at e_1990_10
  have e_1990_11 := arithEq_of_rows h (row := 1990) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_11
  simp only [← c121425, ← c121426, k2, ← c121427] at e_1990_11
  have hr := e_1990_11
  simp only [e_1990_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13983)) := by
  have c121428 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121429 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121430 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121431 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121432 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121433 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_12 := arithEq_of_rows h (row := 1990) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_12
  simp only [← c121428, ← c121429, ← c121430] at e_1990_12
  have e_1990_13 := arithEq_of_rows h (row := 1990) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_13
  simp only [← c121431, ← c121432, k2, ← c121433] at e_1990_13
  have hr := e_1990_13
  simp only [e_1990_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13984)) := by
  have c121434 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121435 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121436 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121437 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121438 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121439 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_14 := arithEq_of_rows h (row := 1990) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_14
  simp only [← c121434, ← c121435, ← c121436] at e_1990_14
  have e_1990_15 := arithEq_of_rows h (row := 1990) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_15
  simp only [← c121437, ← c121438, k2, ← c121439] at e_1990_15
  have hr := e_1990_15
  simp only [e_1990_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13985)) := by
  have c121440 := (publicBatchWrapper89_copies3795 a h).1
  have c121441 := (publicBatchWrapper89_copies3795 a h).2.1
  have c121442 := (publicBatchWrapper89_copies3795 a h).2.2.1
  have c121443 := (publicBatchWrapper89_copies3795 a h).2.2.2.1
  have c121444 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.1
  have c121445 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_16 := arithEq_of_rows h (row := 1990) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_16
  simp only [← c121440, ← c121441, ← c121442] at e_1990_16
  have e_1990_17 := arithEq_of_rows h (row := 1990) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_17
  simp only [← c121443, ← c121444, k2, ← c121445] at e_1990_17
  have hr := e_1990_17
  simp only [e_1990_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13986)) := by
  have c121446 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.1
  have c121447 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.1
  have c121448 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.1
  have c121449 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.1
  have c121450 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121451 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_18 := arithEq_of_rows h (row := 1990) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_18
  simp only [← c121446, ← c121447, ← c121448] at e_1990_18
  have e_1990_19 := arithEq_of_rows h (row := 1990) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_19
  simp only [← c121449, ← c121450, k2, ← c121451] at e_1990_19
  have hr := e_1990_19
  simp only [e_1990_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13987)) := by
  have c121452 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121453 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121454 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121455 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121456 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121457 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_0 := arithEq_of_rows h (row := 1991) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_0
  simp only [← c121452, ← c121453, ← c121454] at e_1991_0
  have e_1991_1 := arithEq_of_rows h (row := 1991) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_1
  simp only [← c121455, ← c121456, k2, ← c121457] at e_1991_1
  have hr := e_1991_1
  simp only [e_1991_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13988)) := by
  have c121458 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121459 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121460 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121461 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121462 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121463 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_2 := arithEq_of_rows h (row := 1991) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_2
  simp only [← c121458, ← c121459, ← c121460] at e_1991_2
  have e_1991_3 := arithEq_of_rows h (row := 1991) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_3
  simp only [← c121461, ← c121462, k2, ← c121463] at e_1991_3
  have hr := e_1991_3
  simp only [e_1991_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13989)) := by
  have c121464 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121465 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121466 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121467 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121468 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121469 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_4 := arithEq_of_rows h (row := 1991) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_4
  simp only [← c121464, ← c121465, ← c121466] at e_1991_4
  have e_1991_5 := arithEq_of_rows h (row := 1991) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_5
  simp only [← c121467, ← c121468, k2, ← c121469] at e_1991_5
  have hr := e_1991_5
  simp only [e_1991_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13990)) := by
  have c121470 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121471 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121472 := (publicBatchWrapper89_copies3796 a h).1
  have c121473 := (publicBatchWrapper89_copies3796 a h).2.1
  have c121474 := (publicBatchWrapper89_copies3796 a h).2.2.1
  have c121475 := (publicBatchWrapper89_copies3796 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_6 := arithEq_of_rows h (row := 1991) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_6
  simp only [← c121470, ← c121471, ← c121472] at e_1991_6
  have e_1991_7 := arithEq_of_rows h (row := 1991) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_7
  simp only [← c121473, ← c121474, k2, ← c121475] at e_1991_7
  have hr := e_1991_7
  simp only [e_1991_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13991)) := by
  have c121476 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.1
  have c121477 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.1
  have c121478 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.1
  have c121479 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.1
  have c121480 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.1
  have c121481 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_8 := arithEq_of_rows h (row := 1991) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_8
  simp only [← c121476, ← c121477, ← c121478] at e_1991_8
  have e_1991_9 := arithEq_of_rows h (row := 1991) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_9
  simp only [← c121479, ← c121480, k2, ← c121481] at e_1991_9
  have hr := e_1991_9
  simp only [e_1991_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13992)) := by
  have c121482 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121483 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121484 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121485 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121486 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121487 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_10 := arithEq_of_rows h (row := 1991) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_10
  simp only [← c121482, ← c121483, ← c121484] at e_1991_10
  have e_1991_11 := arithEq_of_rows h (row := 1991) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_11
  simp only [← c121485, ← c121486, k2, ← c121487] at e_1991_11
  have hr := e_1991_11
  simp only [e_1991_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13993)) := by
  have c121488 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121489 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121490 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121491 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121492 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121493 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_12 := arithEq_of_rows h (row := 1991) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_12
  simp only [← c121488, ← c121489, ← c121490] at e_1991_12
  have e_1991_13 := arithEq_of_rows h (row := 1991) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_13
  simp only [← c121491, ← c121492, k2, ← c121493] at e_1991_13
  have hr := e_1991_13
  simp only [e_1991_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13994)) := by
  have c121494 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121495 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121496 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121497 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121498 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121499 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_14 := arithEq_of_rows h (row := 1991) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_14
  simp only [← c121494, ← c121495, ← c121496] at e_1991_14
  have e_1991_15 := arithEq_of_rows h (row := 1991) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_15
  simp only [← c121497, ← c121498, k2, ← c121499] at e_1991_15
  have hr := e_1991_15
  simp only [e_1991_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13995)) := by
  have c121500 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121501 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121502 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121503 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121504 := (publicBatchWrapper89_copies3797 a h).1
  have c121505 := (publicBatchWrapper89_copies3797 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_16 := arithEq_of_rows h (row := 1991) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_16
  simp only [← c121500, ← c121501, ← c121502] at e_1991_16
  have e_1991_17 := arithEq_of_rows h (row := 1991) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_17
  simp only [← c121503, ← c121504, k2, ← c121505] at e_1991_17
  have hr := e_1991_17
  simp only [e_1991_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13996)) := by
  have c121506 := (publicBatchWrapper89_copies3797 a h).2.2.1
  have c121507 := (publicBatchWrapper89_copies3797 a h).2.2.2.1
  have c121508 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.1
  have c121509 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.1
  have c121510 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.1
  have c121511 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_18 := arithEq_of_rows h (row := 1991) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_18
  simp only [← c121506, ← c121507, ← c121508] at e_1991_18
  have e_1991_19 := arithEq_of_rows h (row := 1991) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_19
  simp only [← c121509, ← c121510, k2, ← c121511] at e_1991_19
  have hr := e_1991_19
  simp only [e_1991_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13997)) := by
  have c121512 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.1
  have c121513 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.1
  have c121514 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121515 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121516 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121517 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_0 := arithEq_of_rows h (row := 1992) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_0
  simp only [← c121512, ← c121513, ← c121514] at e_1992_0
  have e_1992_1 := arithEq_of_rows h (row := 1992) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_1
  simp only [← c121515, ← c121516, k2, ← c121517] at e_1992_1
  have hr := e_1992_1
  simp only [e_1992_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13998)) := by
  have c121518 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121519 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121520 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121521 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121522 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121523 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_2 := arithEq_of_rows h (row := 1992) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_2
  simp only [← c121518, ← c121519, ← c121520] at e_1992_2
  have e_1992_3 := arithEq_of_rows h (row := 1992) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_3
  simp only [← c121521, ← c121522, k2, ← c121523] at e_1992_3
  have hr := e_1992_3
  simp only [e_1992_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13999)) := by
  have c121524 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121525 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121526 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121527 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121528 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121529 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_4 := arithEq_of_rows h (row := 1992) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_4
  simp only [← c121524, ← c121525, ← c121526] at e_1992_4
  have e_1992_5 := arithEq_of_rows h (row := 1992) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_5
  simp only [← c121527, ← c121528, k2, ← c121529] at e_1992_5
  have hr := e_1992_5
  simp only [e_1992_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 14000)) := by
  have c121530 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121531 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121532 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121533 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121534 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121535 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_6 := arithEq_of_rows h (row := 1992) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_6
  simp only [← c121530, ← c121531, ← c121532] at e_1992_6
  have e_1992_7 := arithEq_of_rows h (row := 1992) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_7
  simp only [← c121533, ← c121534, k2, ← c121535] at e_1992_7
  have hr := e_1992_7
  simp only [e_1992_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 14001)) := by
  have c121536 := (publicBatchWrapper89_copies3798 a h).1
  have c121537 := (publicBatchWrapper89_copies3798 a h).2.1
  have c121538 := (publicBatchWrapper89_copies3798 a h).2.2.1
  have c121539 := (publicBatchWrapper89_copies3798 a h).2.2.2.1
  have c121540 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.1
  have c121541 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_8 := arithEq_of_rows h (row := 1992) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_8
  simp only [← c121536, ← c121537, ← c121538] at e_1992_8
  have e_1992_9 := arithEq_of_rows h (row := 1992) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_9
  simp only [← c121539, ← c121540, k2, ← c121541] at e_1992_9
  have hr := e_1992_9
  simp only [e_1992_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14170)) := by
  have c121542 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.1
  have c121543 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.1
  have c121544 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.1
  have c121545 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.1
  have c121546 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121547 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_10 := arithEq_of_rows h (row := 1992) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_10
  simp only [← c121542, ← c121543, ← c121544] at e_1992_10
  have e_1992_11 := arithEq_of_rows h (row := 1992) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_11
  simp only [← c121545, ← c121546, k2, ← c121547] at e_1992_11
  have hr := e_1992_11
  simp only [e_1992_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14171)) := by
  have c121548 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121549 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121550 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121551 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121552 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121553 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_12 := arithEq_of_rows h (row := 1992) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_12
  simp only [← c121548, ← c121549, ← c121550] at e_1992_12
  have e_1992_13 := arithEq_of_rows h (row := 1992) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_13
  simp only [← c121551, ← c121552, k2, ← c121553] at e_1992_13
  have hr := e_1992_13
  simp only [e_1992_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14172)) := by
  have c121554 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121555 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121556 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121557 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121558 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121559 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_14 := arithEq_of_rows h (row := 1992) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_14
  simp only [← c121554, ← c121555, ← c121556] at e_1992_14
  have e_1992_15 := arithEq_of_rows h (row := 1992) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_15
  simp only [← c121557, ← c121558, k2, ← c121559] at e_1992_15
  have hr := e_1992_15
  simp only [e_1992_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14173)) := by
  have c121560 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121561 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121562 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121563 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121564 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121565 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_16 := arithEq_of_rows h (row := 1992) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_16
  simp only [← c121560, ← c121561, ← c121562] at e_1992_16
  have e_1992_17 := arithEq_of_rows h (row := 1992) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_17
  simp only [← c121563, ← c121564, k2, ← c121565] at e_1992_17
  have hr := e_1992_17
  simp only [e_1992_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14174)) := by
  have c121566 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121567 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121568 := (publicBatchWrapper89_copies3799 a h).1
  have c121569 := (publicBatchWrapper89_copies3799 a h).2.1
  have c121570 := (publicBatchWrapper89_copies3799 a h).2.2.1
  have c121571 := (publicBatchWrapper89_copies3799 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_18 := arithEq_of_rows h (row := 1992) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_18
  simp only [← c121566, ← c121567, ← c121568] at e_1992_18
  have e_1992_19 := arithEq_of_rows h (row := 1992) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_19
  simp only [← c121569, ← c121570, k2, ← c121571] at e_1992_19
  have hr := e_1992_19
  simp only [e_1992_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14175)) := by
  have c121572 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.1
  have c121573 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.1
  have c121574 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.1
  have c121575 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.1
  have c121576 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.1
  have c121577 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_0 := arithEq_of_rows h (row := 1993) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_0
  simp only [← c121572, ← c121573, ← c121574] at e_1993_0
  have e_1993_1 := arithEq_of_rows h (row := 1993) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_1
  simp only [← c121575, ← c121576, k2, ← c121577] at e_1993_1
  have hr := e_1993_1
  simp only [e_1993_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14176)) := by
  have c121578 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121579 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121580 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121581 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121582 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121583 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_2 := arithEq_of_rows h (row := 1993) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_2
  simp only [← c121578, ← c121579, ← c121580] at e_1993_2
  have e_1993_3 := arithEq_of_rows h (row := 1993) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_3
  simp only [← c121581, ← c121582, k2, ← c121583] at e_1993_3
  have hr := e_1993_3
  simp only [e_1993_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14177)) := by
  have c121584 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121585 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121586 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121587 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121588 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121589 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_4 := arithEq_of_rows h (row := 1993) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_4
  simp only [← c121584, ← c121585, ← c121586] at e_1993_4
  have e_1993_5 := arithEq_of_rows h (row := 1993) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_5
  simp only [← c121587, ← c121588, k2, ← c121589] at e_1993_5
  have hr := e_1993_5
  simp only [e_1993_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14178)) := by
  have c121590 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121591 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121592 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121593 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121594 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121595 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_6 := arithEq_of_rows h (row := 1993) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_6
  simp only [← c121590, ← c121591, ← c121592] at e_1993_6
  have e_1993_7 := arithEq_of_rows h (row := 1993) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_7
  simp only [← c121593, ← c121594, k2, ← c121595] at e_1993_7
  have hr := e_1993_7
  simp only [e_1993_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14179)) := by
  have c121596 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121597 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121598 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121599 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121600 := (publicBatchWrapper89_copies3800 a h).1
  have c121601 := (publicBatchWrapper89_copies3800 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_8 := arithEq_of_rows h (row := 1993) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_8
  simp only [← c121596, ← c121597, ← c121598] at e_1993_8
  have e_1993_9 := arithEq_of_rows h (row := 1993) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_9
  simp only [← c121599, ← c121600, k2, ← c121601] at e_1993_9
  have hr := e_1993_9
  simp only [e_1993_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14180)) := by
  have c121602 := (publicBatchWrapper89_copies3800 a h).2.2.1
  have c121603 := (publicBatchWrapper89_copies3800 a h).2.2.2.1
  have c121604 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.1
  have c121605 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.1
  have c121606 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.1
  have c121607 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_10 := arithEq_of_rows h (row := 1993) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_10
  simp only [← c121602, ← c121603, ← c121604] at e_1993_10
  have e_1993_11 := arithEq_of_rows h (row := 1993) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_11
  simp only [← c121605, ← c121606, k2, ← c121607] at e_1993_11
  have hr := e_1993_11
  simp only [e_1993_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14181)) := by
  have c121608 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.1
  have c121609 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.1
  have c121610 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121611 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121612 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121613 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_12 := arithEq_of_rows h (row := 1993) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_12
  simp only [← c121608, ← c121609, ← c121610] at e_1993_12
  have e_1993_13 := arithEq_of_rows h (row := 1993) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_13
  simp only [← c121611, ← c121612, k2, ← c121613] at e_1993_13
  have hr := e_1993_13
  simp only [e_1993_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14182)) := by
  have c121614 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121615 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121616 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121617 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121618 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121619 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_14 := arithEq_of_rows h (row := 1993) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_14
  simp only [← c121614, ← c121615, ← c121616] at e_1993_14
  have e_1993_15 := arithEq_of_rows h (row := 1993) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_15
  simp only [← c121617, ← c121618, k2, ← c121619] at e_1993_15
  have hr := e_1993_15
  simp only [e_1993_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14183)) := by
  have c121620 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121621 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121622 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121623 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121624 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121625 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_16 := arithEq_of_rows h (row := 1993) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_16
  simp only [← c121620, ← c121621, ← c121622] at e_1993_16
  have e_1993_17 := arithEq_of_rows h (row := 1993) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_17
  simp only [← c121623, ← c121624, k2, ← c121625] at e_1993_17
  have hr := e_1993_17
  simp only [e_1993_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14184)) := by
  have c121626 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121627 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121628 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121629 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121630 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121631 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_18 := arithEq_of_rows h (row := 1993) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_18
  simp only [← c121626, ← c121627, ← c121628] at e_1993_18
  have e_1993_19 := arithEq_of_rows h (row := 1993) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_19
  simp only [← c121629, ← c121630, k2, ← c121631] at e_1993_19
  have hr := e_1993_19
  simp only [e_1993_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14185)) := by
  have c121632 := (publicBatchWrapper89_copies3801 a h).1
  have c121633 := (publicBatchWrapper89_copies3801 a h).2.1
  have c121634 := (publicBatchWrapper89_copies3801 a h).2.2.1
  have c121635 := (publicBatchWrapper89_copies3801 a h).2.2.2.1
  have c121636 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.1
  have c121637 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_0 := arithEq_of_rows h (row := 1994) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_0
  simp only [← c121632, ← c121633, ← c121634] at e_1994_0
  have e_1994_1 := arithEq_of_rows h (row := 1994) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_1
  simp only [← c121635, ← c121636, k2, ← c121637] at e_1994_1
  have hr := e_1994_1
  simp only [e_1994_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14186)) := by
  have c121638 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.1
  have c121639 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.1
  have c121640 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.1
  have c121641 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.1
  have c121642 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121643 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_2 := arithEq_of_rows h (row := 1994) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_2
  simp only [← c121638, ← c121639, ← c121640] at e_1994_2
  have e_1994_3 := arithEq_of_rows h (row := 1994) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_3
  simp only [← c121641, ← c121642, k2, ← c121643] at e_1994_3
  have hr := e_1994_3
  simp only [e_1994_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14187)) := by
  have c121644 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121645 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121646 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121647 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121648 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121649 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_4 := arithEq_of_rows h (row := 1994) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_4
  simp only [← c121644, ← c121645, ← c121646] at e_1994_4
  have e_1994_5 := arithEq_of_rows h (row := 1994) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_5
  simp only [← c121647, ← c121648, k2, ← c121649] at e_1994_5
  have hr := e_1994_5
  simp only [e_1994_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14188)) := by
  have c121650 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121651 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121652 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121653 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121654 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121655 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_6 := arithEq_of_rows h (row := 1994) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_6
  simp only [← c121650, ← c121651, ← c121652] at e_1994_6
  have e_1994_7 := arithEq_of_rows h (row := 1994) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_7
  simp only [← c121653, ← c121654, k2, ← c121655] at e_1994_7
  have hr := e_1994_7
  simp only [e_1994_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14189)) := by
  have c121656 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121657 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121658 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121659 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121660 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121661 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_8 := arithEq_of_rows h (row := 1994) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_8
  simp only [← c121656, ← c121657, ← c121658] at e_1994_8
  have e_1994_9 := arithEq_of_rows h (row := 1994) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_9
  simp only [← c121659, ← c121660, k2, ← c121661] at e_1994_9
  have hr := e_1994_9
  simp only [e_1994_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14190)) := by
  have c121662 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121663 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121664 := (publicBatchWrapper89_copies3802 a h).1
  have c121665 := (publicBatchWrapper89_copies3802 a h).2.1
  have c121666 := (publicBatchWrapper89_copies3802 a h).2.2.1
  have c121667 := (publicBatchWrapper89_copies3802 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_10 := arithEq_of_rows h (row := 1994) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_10
  simp only [← c121662, ← c121663, ← c121664] at e_1994_10
  have e_1994_11 := arithEq_of_rows h (row := 1994) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_11
  simp only [← c121665, ← c121666, k2, ← c121667] at e_1994_11
  have hr := e_1994_11
  simp only [e_1994_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14191)) := by
  have c121668 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.1
  have c121669 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.1
  have c121670 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.1
  have c121671 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.1
  have c121672 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.1
  have c121673 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_12 := arithEq_of_rows h (row := 1994) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_12
  simp only [← c121668, ← c121669, ← c121670] at e_1994_12
  have e_1994_13 := arithEq_of_rows h (row := 1994) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_13
  simp only [← c121671, ← c121672, k2, ← c121673] at e_1994_13
  have hr := e_1994_13
  simp only [e_1994_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14192)) := by
  have c121674 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121675 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121676 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121677 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121678 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121679 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_14 := arithEq_of_rows h (row := 1994) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_14
  simp only [← c121674, ← c121675, ← c121676] at e_1994_14
  have e_1994_15 := arithEq_of_rows h (row := 1994) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_15
  simp only [← c121677, ← c121678, k2, ← c121679] at e_1994_15
  have hr := e_1994_15
  simp only [e_1994_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14193)) := by
  have c121680 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121681 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121682 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121683 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121684 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121685 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_16 := arithEq_of_rows h (row := 1994) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_16
  simp only [← c121680, ← c121681, ← c121682] at e_1994_16
  have e_1994_17 := arithEq_of_rows h (row := 1994) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_17
  simp only [← c121683, ← c121684, k2, ← c121685] at e_1994_17
  have hr := e_1994_17
  simp only [e_1994_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14194)) := by
  have c121686 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121687 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121688 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121689 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121690 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121691 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_18 := arithEq_of_rows h (row := 1994) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_18
  simp only [← c121686, ← c121687, ← c121688] at e_1994_18
  have e_1994_19 := arithEq_of_rows h (row := 1994) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_19
  simp only [← c121689, ← c121690, k2, ← c121691] at e_1994_19
  have hr := e_1994_19
  simp only [e_1994_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14195)) := by
  have c121692 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121693 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121694 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121695 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121696 := (publicBatchWrapper89_copies3803 a h).1
  have c121697 := (publicBatchWrapper89_copies3803 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_0 := arithEq_of_rows h (row := 1995) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_0
  simp only [← c121692, ← c121693, ← c121694] at e_1995_0
  have e_1995_1 := arithEq_of_rows h (row := 1995) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_1
  simp only [← c121695, ← c121696, k2, ← c121697] at e_1995_1
  have hr := e_1995_1
  simp only [e_1995_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14196)) := by
  have c121698 := (publicBatchWrapper89_copies3803 a h).2.2.1
  have c121699 := (publicBatchWrapper89_copies3803 a h).2.2.2.1
  have c121700 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.1
  have c121701 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.1
  have c121702 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.1
  have c121703 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_2 := arithEq_of_rows h (row := 1995) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_2
  simp only [← c121698, ← c121699, ← c121700] at e_1995_2
  have e_1995_3 := arithEq_of_rows h (row := 1995) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_3
  simp only [← c121701, ← c121702, k2, ← c121703] at e_1995_3
  have hr := e_1995_3
  simp only [e_1995_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14197)) := by
  have c121704 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.1
  have c121705 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.1
  have c121706 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121707 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121708 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121709 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_4 := arithEq_of_rows h (row := 1995) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_4
  simp only [← c121704, ← c121705, ← c121706] at e_1995_4
  have e_1995_5 := arithEq_of_rows h (row := 1995) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_5
  simp only [← c121707, ← c121708, k2, ← c121709] at e_1995_5
  have hr := e_1995_5
  simp only [e_1995_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14198)) := by
  have c121710 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121711 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121712 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121713 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121714 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121715 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_6 := arithEq_of_rows h (row := 1995) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_6
  simp only [← c121710, ← c121711, ← c121712] at e_1995_6
  have e_1995_7 := arithEq_of_rows h (row := 1995) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_7
  simp only [← c121713, ← c121714, k2, ← c121715] at e_1995_7
  have hr := e_1995_7
  simp only [e_1995_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14199)) := by
  have c121716 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121717 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121718 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121719 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121720 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121721 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_8 := arithEq_of_rows h (row := 1995) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_8
  simp only [← c121716, ← c121717, ← c121718] at e_1995_8
  have e_1995_9 := arithEq_of_rows h (row := 1995) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_9
  simp only [← c121719, ← c121720, k2, ← c121721] at e_1995_9
  have hr := e_1995_9
  simp only [e_1995_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14200)) := by
  have c121722 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121723 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121724 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121725 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121726 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121727 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_10 := arithEq_of_rows h (row := 1995) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_10
  simp only [← c121722, ← c121723, ← c121724] at e_1995_10
  have e_1995_11 := arithEq_of_rows h (row := 1995) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_11
  simp only [← c121725, ← c121726, k2, ← c121727] at e_1995_11
  have hr := e_1995_11
  simp only [e_1995_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14201)) := by
  have c121728 := (publicBatchWrapper89_copies3804 a h).1
  have c121729 := (publicBatchWrapper89_copies3804 a h).2.1
  have c121730 := (publicBatchWrapper89_copies3804 a h).2.2.1
  have c121731 := (publicBatchWrapper89_copies3804 a h).2.2.2.1
  have c121732 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.1
  have c121733 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_12 := arithEq_of_rows h (row := 1995) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_12
  simp only [← c121728, ← c121729, ← c121730] at e_1995_12
  have e_1995_13 := arithEq_of_rows h (row := 1995) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_13
  simp only [← c121731, ← c121732, k2, ← c121733] at e_1995_13
  have hr := e_1995_13
  simp only [e_1995_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14370)) := by
  have c121734 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.1
  have c121735 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.1
  have c121736 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.1
  have c121737 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.1
  have c121738 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121739 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_14 := arithEq_of_rows h (row := 1995) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_14
  simp only [← c121734, ← c121735, ← c121736] at e_1995_14
  have e_1995_15 := arithEq_of_rows h (row := 1995) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_15
  simp only [← c121737, ← c121738, k2, ← c121739] at e_1995_15
  have hr := e_1995_15
  simp only [e_1995_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14371)) := by
  have c121740 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121741 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121742 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121743 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121744 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121745 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_16 := arithEq_of_rows h (row := 1995) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_16
  simp only [← c121740, ← c121741, ← c121742] at e_1995_16
  have e_1995_17 := arithEq_of_rows h (row := 1995) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_17
  simp only [← c121743, ← c121744, k2, ← c121745] at e_1995_17
  have hr := e_1995_17
  simp only [e_1995_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14372)) := by
  have c121746 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121747 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121748 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121749 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121750 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121751 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_18 := arithEq_of_rows h (row := 1995) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_18
  simp only [← c121746, ← c121747, ← c121748] at e_1995_18
  have e_1995_19 := arithEq_of_rows h (row := 1995) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_19
  simp only [← c121749, ← c121750, k2, ← c121751] at e_1995_19
  have hr := e_1995_19
  simp only [e_1995_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14373)) := by
  have c121752 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121753 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121754 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121755 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121756 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121757 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_0 := arithEq_of_rows h (row := 1996) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_0
  simp only [← c121752, ← c121753, ← c121754] at e_1996_0
  have e_1996_1 := arithEq_of_rows h (row := 1996) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_1
  simp only [← c121755, ← c121756, k2, ← c121757] at e_1996_1
  have hr := e_1996_1
  simp only [e_1996_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14374)) := by
  have c121758 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121759 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121760 := (publicBatchWrapper89_copies3805 a h).1
  have c121761 := (publicBatchWrapper89_copies3805 a h).2.1
  have c121762 := (publicBatchWrapper89_copies3805 a h).2.2.1
  have c121763 := (publicBatchWrapper89_copies3805 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_2 := arithEq_of_rows h (row := 1996) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_2
  simp only [← c121758, ← c121759, ← c121760] at e_1996_2
  have e_1996_3 := arithEq_of_rows h (row := 1996) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_3
  simp only [← c121761, ← c121762, k2, ← c121763] at e_1996_3
  have hr := e_1996_3
  simp only [e_1996_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14375)) := by
  have c121764 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.1
  have c121765 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.1
  have c121766 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.1
  have c121767 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.1
  have c121768 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.1
  have c121769 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_4 := arithEq_of_rows h (row := 1996) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_4
  simp only [← c121764, ← c121765, ← c121766] at e_1996_4
  have e_1996_5 := arithEq_of_rows h (row := 1996) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_5
  simp only [← c121767, ← c121768, k2, ← c121769] at e_1996_5
  have hr := e_1996_5
  simp only [e_1996_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14376)) := by
  have c121770 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121771 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121772 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121773 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121774 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121775 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_6 := arithEq_of_rows h (row := 1996) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_6
  simp only [← c121770, ← c121771, ← c121772] at e_1996_6
  have e_1996_7 := arithEq_of_rows h (row := 1996) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_7
  simp only [← c121773, ← c121774, k2, ← c121775] at e_1996_7
  have hr := e_1996_7
  simp only [e_1996_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14377)) := by
  have c121776 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121777 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121778 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121779 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121780 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121781 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_8 := arithEq_of_rows h (row := 1996) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_8
  simp only [← c121776, ← c121777, ← c121778] at e_1996_8
  have e_1996_9 := arithEq_of_rows h (row := 1996) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_9
  simp only [← c121779, ← c121780, k2, ← c121781] at e_1996_9
  have hr := e_1996_9
  simp only [e_1996_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14378)) := by
  have c121782 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121783 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121784 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121785 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121786 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121787 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_10 := arithEq_of_rows h (row := 1996) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_10
  simp only [← c121782, ← c121783, ← c121784] at e_1996_10
  have e_1996_11 := arithEq_of_rows h (row := 1996) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_11
  simp only [← c121785, ← c121786, k2, ← c121787] at e_1996_11
  have hr := e_1996_11
  simp only [e_1996_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14379)) := by
  have c121788 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121789 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121790 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121791 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121792 := (publicBatchWrapper89_copies3806 a h).1
  have c121793 := (publicBatchWrapper89_copies3806 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_12 := arithEq_of_rows h (row := 1996) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_12
  simp only [← c121788, ← c121789, ← c121790] at e_1996_12
  have e_1996_13 := arithEq_of_rows h (row := 1996) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_13
  simp only [← c121791, ← c121792, k2, ← c121793] at e_1996_13
  have hr := e_1996_13
  simp only [e_1996_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14380)) := by
  have c121794 := (publicBatchWrapper89_copies3806 a h).2.2.1
  have c121795 := (publicBatchWrapper89_copies3806 a h).2.2.2.1
  have c121796 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.1
  have c121797 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.1
  have c121798 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.1
  have c121799 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_14 := arithEq_of_rows h (row := 1996) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_14
  simp only [← c121794, ← c121795, ← c121796] at e_1996_14
  have e_1996_15 := arithEq_of_rows h (row := 1996) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_15
  simp only [← c121797, ← c121798, k2, ← c121799] at e_1996_15
  have hr := e_1996_15
  simp only [e_1996_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14381)) := by
  have c121800 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.1
  have c121801 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.1
  have c121802 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121803 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121804 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121805 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_16 := arithEq_of_rows h (row := 1996) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_16
  simp only [← c121800, ← c121801, ← c121802] at e_1996_16
  have e_1996_17 := arithEq_of_rows h (row := 1996) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_17
  simp only [← c121803, ← c121804, k2, ← c121805] at e_1996_17
  have hr := e_1996_17
  simp only [e_1996_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14382)) := by
  have c121806 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121807 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121808 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121809 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121810 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121811 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_18 := arithEq_of_rows h (row := 1996) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_18
  simp only [← c121806, ← c121807, ← c121808] at e_1996_18
  have e_1996_19 := arithEq_of_rows h (row := 1996) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_19
  simp only [← c121809, ← c121810, k2, ← c121811] at e_1996_19
  have hr := e_1996_19
  simp only [e_1996_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14383)) := by
  have c121812 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121813 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121814 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121815 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121816 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121817 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_0 := arithEq_of_rows h (row := 1997) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_0
  simp only [← c121812, ← c121813, ← c121814] at e_1997_0
  have e_1997_1 := arithEq_of_rows h (row := 1997) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_1
  simp only [← c121815, ← c121816, k2, ← c121817] at e_1997_1
  have hr := e_1997_1
  simp only [e_1997_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14384)) := by
  have c121818 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121819 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121820 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121821 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121822 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121823 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_2 := arithEq_of_rows h (row := 1997) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_2
  simp only [← c121818, ← c121819, ← c121820] at e_1997_2
  have e_1997_3 := arithEq_of_rows h (row := 1997) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_3
  simp only [← c121821, ← c121822, k2, ← c121823] at e_1997_3
  have hr := e_1997_3
  simp only [e_1997_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14385)) := by
  have c121824 := (publicBatchWrapper89_copies3807 a h).1
  have c121825 := (publicBatchWrapper89_copies3807 a h).2.1
  have c121826 := (publicBatchWrapper89_copies3807 a h).2.2.1
  have c121827 := (publicBatchWrapper89_copies3807 a h).2.2.2.1
  have c121828 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.1
  have c121829 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_4 := arithEq_of_rows h (row := 1997) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_4
  simp only [← c121824, ← c121825, ← c121826] at e_1997_4
  have e_1997_5 := arithEq_of_rows h (row := 1997) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_5
  simp only [← c121827, ← c121828, k2, ← c121829] at e_1997_5
  have hr := e_1997_5
  simp only [e_1997_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14386)) := by
  have c121830 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.1
  have c121831 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.1
  have c121832 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.1
  have c121833 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.1
  have c121834 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121835 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_6 := arithEq_of_rows h (row := 1997) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_6
  simp only [← c121830, ← c121831, ← c121832] at e_1997_6
  have e_1997_7 := arithEq_of_rows h (row := 1997) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_7
  simp only [← c121833, ← c121834, k2, ← c121835] at e_1997_7
  have hr := e_1997_7
  simp only [e_1997_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14387)) := by
  have c121836 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121837 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121838 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121839 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121840 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121841 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_8 := arithEq_of_rows h (row := 1997) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_8
  simp only [← c121836, ← c121837, ← c121838] at e_1997_8
  have e_1997_9 := arithEq_of_rows h (row := 1997) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_9
  simp only [← c121839, ← c121840, k2, ← c121841] at e_1997_9
  have hr := e_1997_9
  simp only [e_1997_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14388)) := by
  have c121842 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121843 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121844 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121845 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121846 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121847 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_10 := arithEq_of_rows h (row := 1997) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_10
  simp only [← c121842, ← c121843, ← c121844] at e_1997_10
  have e_1997_11 := arithEq_of_rows h (row := 1997) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_11
  simp only [← c121845, ← c121846, k2, ← c121847] at e_1997_11
  have hr := e_1997_11
  simp only [e_1997_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14389)) := by
  have c121848 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121849 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121850 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121851 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121852 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121853 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_12 := arithEq_of_rows h (row := 1997) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_12
  simp only [← c121848, ← c121849, ← c121850] at e_1997_12
  have e_1997_13 := arithEq_of_rows h (row := 1997) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_13
  simp only [← c121851, ← c121852, k2, ← c121853] at e_1997_13
  have hr := e_1997_13
  simp only [e_1997_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14390)) := by
  have c121854 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121855 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121856 := (publicBatchWrapper89_copies3808 a h).1
  have c121857 := (publicBatchWrapper89_copies3808 a h).2.1
  have c121858 := (publicBatchWrapper89_copies3808 a h).2.2.1
  have c121859 := (publicBatchWrapper89_copies3808 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_14 := arithEq_of_rows h (row := 1997) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_14
  simp only [← c121854, ← c121855, ← c121856] at e_1997_14
  have e_1997_15 := arithEq_of_rows h (row := 1997) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_15
  simp only [← c121857, ← c121858, k2, ← c121859] at e_1997_15
  have hr := e_1997_15
  simp only [e_1997_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14391)) := by
  have c121860 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.1
  have c121861 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.1
  have c121862 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.1
  have c121863 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.1
  have c121864 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.1
  have c121865 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_16 := arithEq_of_rows h (row := 1997) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_16
  simp only [← c121860, ← c121861, ← c121862] at e_1997_16
  have e_1997_17 := arithEq_of_rows h (row := 1997) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_17
  simp only [← c121863, ← c121864, k2, ← c121865] at e_1997_17
  have hr := e_1997_17
  simp only [e_1997_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14392)) := by
  have c121866 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121867 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121868 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121869 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121870 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121871 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_18 := arithEq_of_rows h (row := 1997) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_18
  simp only [← c121866, ← c121867, ← c121868] at e_1997_18
  have e_1997_19 := arithEq_of_rows h (row := 1997) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_19
  simp only [← c121869, ← c121870, k2, ← c121871] at e_1997_19
  have hr := e_1997_19
  simp only [e_1997_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14393)) := by
  have c121872 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121873 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121874 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121875 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121876 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121877 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_0 := arithEq_of_rows h (row := 1998) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_0
  simp only [← c121872, ← c121873, ← c121874] at e_1998_0
  have e_1998_1 := arithEq_of_rows h (row := 1998) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_1
  simp only [← c121875, ← c121876, k2, ← c121877] at e_1998_1
  have hr := e_1998_1
  simp only [e_1998_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14394)) := by
  have c121878 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121879 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121880 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121881 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121882 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121883 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_2 := arithEq_of_rows h (row := 1998) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_2
  simp only [← c121878, ← c121879, ← c121880] at e_1998_2
  have e_1998_3 := arithEq_of_rows h (row := 1998) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_3
  simp only [← c121881, ← c121882, k2, ← c121883] at e_1998_3
  have hr := e_1998_3
  simp only [e_1998_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14395)) := by
  have c121884 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121885 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121886 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121887 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121888 := (publicBatchWrapper89_copies3809 a h).1
  have c121889 := (publicBatchWrapper89_copies3809 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_4 := arithEq_of_rows h (row := 1998) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_4
  simp only [← c121884, ← c121885, ← c121886] at e_1998_4
  have e_1998_5 := arithEq_of_rows h (row := 1998) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_5
  simp only [← c121887, ← c121888, k2, ← c121889] at e_1998_5
  have hr := e_1998_5
  simp only [e_1998_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14396)) := by
  have c121890 := (publicBatchWrapper89_copies3809 a h).2.2.1
  have c121891 := (publicBatchWrapper89_copies3809 a h).2.2.2.1
  have c121892 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.1
  have c121893 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.1
  have c121894 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.1
  have c121895 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_6 := arithEq_of_rows h (row := 1998) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_6
  simp only [← c121890, ← c121891, ← c121892] at e_1998_6
  have e_1998_7 := arithEq_of_rows h (row := 1998) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_7
  simp only [← c121893, ← c121894, k2, ← c121895] at e_1998_7
  have hr := e_1998_7
  simp only [e_1998_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14397)) := by
  have c121896 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.1
  have c121897 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.1
  have c121898 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121899 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121900 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121901 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_8 := arithEq_of_rows h (row := 1998) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_8
  simp only [← c121896, ← c121897, ← c121898] at e_1998_8
  have e_1998_9 := arithEq_of_rows h (row := 1998) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_9
  simp only [← c121899, ← c121900, k2, ← c121901] at e_1998_9
  have hr := e_1998_9
  simp only [e_1998_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14398)) := by
  have c121902 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121903 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121904 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121905 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121906 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121907 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_10 := arithEq_of_rows h (row := 1998) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_10
  simp only [← c121902, ← c121903, ← c121904] at e_1998_10
  have e_1998_11 := arithEq_of_rows h (row := 1998) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_11
  simp only [← c121905, ← c121906, k2, ← c121907] at e_1998_11
  have hr := e_1998_11
  simp only [e_1998_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14399)) := by
  have c121908 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121909 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121910 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121911 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121912 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121913 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_12 := arithEq_of_rows h (row := 1998) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_12
  simp only [← c121908, ← c121909, ← c121910] at e_1998_12
  have e_1998_13 := arithEq_of_rows h (row := 1998) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_13
  simp only [← c121911, ← c121912, k2, ← c121913] at e_1998_13
  have hr := e_1998_13
  simp only [e_1998_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14400)) := by
  have c121914 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121915 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121916 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121917 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121918 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121919 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_14 := arithEq_of_rows h (row := 1998) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_14
  simp only [← c121914, ← c121915, ← c121916] at e_1998_14
  have e_1998_15 := arithEq_of_rows h (row := 1998) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_15
  simp only [← c121917, ← c121918, k2, ← c121919] at e_1998_15
  have hr := e_1998_15
  simp only [e_1998_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14401)) := by
  have c121920 := (publicBatchWrapper89_copies3810 a h).1
  have c121921 := (publicBatchWrapper89_copies3810 a h).2.1
  have c121922 := (publicBatchWrapper89_copies3810 a h).2.2.1
  have c121923 := (publicBatchWrapper89_copies3810 a h).2.2.2.1
  have c121924 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.1
  have c121925 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_16 := arithEq_of_rows h (row := 1998) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_16
  simp only [← c121920, ← c121921, ← c121922] at e_1998_16
  have e_1998_17 := arithEq_of_rows h (row := 1998) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_17
  simp only [← c121923, ← c121924, k2, ← c121925] at e_1998_17
  have hr := e_1998_17
  simp only [e_1998_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14570)) := by
  have c121926 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.1
  have c121927 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.1
  have c121928 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.1
  have c121929 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.1
  have c121930 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121931 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_18 := arithEq_of_rows h (row := 1998) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_18
  simp only [← c121926, ← c121927, ← c121928] at e_1998_18
  have e_1998_19 := arithEq_of_rows h (row := 1998) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_19
  simp only [← c121929, ← c121930, k2, ← c121931] at e_1998_19
  have hr := e_1998_19
  simp only [e_1998_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14571)) := by
  have c121932 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121933 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121934 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121935 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121936 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121937 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_0 := arithEq_of_rows h (row := 1999) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_0
  simp only [← c121932, ← c121933, ← c121934] at e_1999_0
  have e_1999_1 := arithEq_of_rows h (row := 1999) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_1
  simp only [← c121935, ← c121936, k2, ← c121937] at e_1999_1
  have hr := e_1999_1
  simp only [e_1999_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14572)) := by
  have c121938 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121939 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121940 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121941 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121942 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121943 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_2 := arithEq_of_rows h (row := 1999) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_2
  simp only [← c121938, ← c121939, ← c121940] at e_1999_2
  have e_1999_3 := arithEq_of_rows h (row := 1999) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_3
  simp only [← c121941, ← c121942, k2, ← c121943] at e_1999_3
  have hr := e_1999_3
  simp only [e_1999_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14573)) := by
  have c121944 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121945 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121946 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121947 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121948 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121949 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_4 := arithEq_of_rows h (row := 1999) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_4
  simp only [← c121944, ← c121945, ← c121946] at e_1999_4
  have e_1999_5 := arithEq_of_rows h (row := 1999) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_5
  simp only [← c121947, ← c121948, k2, ← c121949] at e_1999_5
  have hr := e_1999_5
  simp only [e_1999_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14574)) := by
  have c121950 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121951 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121952 := (publicBatchWrapper89_copies3811 a h).1
  have c121953 := (publicBatchWrapper89_copies3811 a h).2.1
  have c121954 := (publicBatchWrapper89_copies3811 a h).2.2.1
  have c121955 := (publicBatchWrapper89_copies3811 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_6 := arithEq_of_rows h (row := 1999) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_6
  simp only [← c121950, ← c121951, ← c121952] at e_1999_6
  have e_1999_7 := arithEq_of_rows h (row := 1999) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_7
  simp only [← c121953, ← c121954, k2, ← c121955] at e_1999_7
  have hr := e_1999_7
  simp only [e_1999_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14575)) := by
  have c121956 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.1
  have c121957 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.1
  have c121958 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.1
  have c121959 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.1
  have c121960 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.1
  have c121961 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_8 := arithEq_of_rows h (row := 1999) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_8
  simp only [← c121956, ← c121957, ← c121958] at e_1999_8
  have e_1999_9 := arithEq_of_rows h (row := 1999) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_9
  simp only [← c121959, ← c121960, k2, ← c121961] at e_1999_9
  have hr := e_1999_9
  simp only [e_1999_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14576)) := by
  have c121962 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121963 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121964 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121965 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121966 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121967 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_10 := arithEq_of_rows h (row := 1999) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_10
  simp only [← c121962, ← c121963, ← c121964] at e_1999_10
  have e_1999_11 := arithEq_of_rows h (row := 1999) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_11
  simp only [← c121965, ← c121966, k2, ← c121967] at e_1999_11
  have hr := e_1999_11
  simp only [e_1999_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14577)) := by
  have c121968 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121969 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121970 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121971 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121972 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121973 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_12 := arithEq_of_rows h (row := 1999) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_12
  simp only [← c121968, ← c121969, ← c121970] at e_1999_12
  have e_1999_13 := arithEq_of_rows h (row := 1999) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_13
  simp only [← c121971, ← c121972, k2, ← c121973] at e_1999_13
  have hr := e_1999_13
  simp only [e_1999_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14578)) := by
  have c121974 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121975 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121976 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121977 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121978 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121979 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_14 := arithEq_of_rows h (row := 1999) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_14
  simp only [← c121974, ← c121975, ← c121976] at e_1999_14
  have e_1999_15 := arithEq_of_rows h (row := 1999) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_15
  simp only [← c121977, ← c121978, k2, ← c121979] at e_1999_15
  have hr := e_1999_15
  simp only [e_1999_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14579)) := by
  have c121980 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121981 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121982 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121983 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121984 := (publicBatchWrapper89_copies3812 a h).1
  have c121985 := (publicBatchWrapper89_copies3812 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_16 := arithEq_of_rows h (row := 1999) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_16
  simp only [← c121980, ← c121981, ← c121982] at e_1999_16
  have e_1999_17 := arithEq_of_rows h (row := 1999) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_17
  simp only [← c121983, ← c121984, k2, ← c121985] at e_1999_17
  have hr := e_1999_17
  simp only [e_1999_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14580)) := by
  have c121986 := (publicBatchWrapper89_copies3812 a h).2.2.1
  have c121987 := (publicBatchWrapper89_copies3812 a h).2.2.2.1
  have c121988 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.1
  have c121989 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.1
  have c121990 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.1
  have c121991 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_18 := arithEq_of_rows h (row := 1999) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_18
  simp only [← c121986, ← c121987, ← c121988] at e_1999_18
  have e_1999_19 := arithEq_of_rows h (row := 1999) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_19
  simp only [← c121989, ← c121990, k2, ← c121991] at e_1999_19
  have hr := e_1999_19
  simp only [e_1999_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14581)) := by
  have c121992 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.1
  have c121993 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.1
  have c121994 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121995 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121996 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121997 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_0 := arithEq_of_rows h (row := 2000) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_0
  simp only [← c121992, ← c121993, ← c121994] at e_2000_0
  have e_2000_1 := arithEq_of_rows h (row := 2000) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_1
  simp only [← c121995, ← c121996, k2, ← c121997] at e_2000_1
  have hr := e_2000_1
  simp only [e_2000_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14582)) := by
  have c121998 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121999 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122000 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122001 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122002 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122003 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_2 := arithEq_of_rows h (row := 2000) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_2
  simp only [← c121998, ← c121999, ← c122000] at e_2000_2
  have e_2000_3 := arithEq_of_rows h (row := 2000) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_3
  simp only [← c122001, ← c122002, k2, ← c122003] at e_2000_3
  have hr := e_2000_3
  simp only [e_2000_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14583)) := by
  have c122004 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122005 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122006 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122007 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122008 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122009 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_4 := arithEq_of_rows h (row := 2000) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_4
  simp only [← c122004, ← c122005, ← c122006] at e_2000_4
  have e_2000_5 := arithEq_of_rows h (row := 2000) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_5
  simp only [← c122007, ← c122008, k2, ← c122009] at e_2000_5
  have hr := e_2000_5
  simp only [e_2000_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14584)) := by
  have c122010 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122011 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122012 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122013 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122014 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122015 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_6 := arithEq_of_rows h (row := 2000) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_6
  simp only [← c122010, ← c122011, ← c122012] at e_2000_6
  have e_2000_7 := arithEq_of_rows h (row := 2000) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_7
  simp only [← c122013, ← c122014, k2, ← c122015] at e_2000_7
  have hr := e_2000_7
  simp only [e_2000_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14585)) := by
  have c122016 := (publicBatchWrapper89_copies3813 a h).1
  have c122017 := (publicBatchWrapper89_copies3813 a h).2.1
  have c122018 := (publicBatchWrapper89_copies3813 a h).2.2.1
  have c122019 := (publicBatchWrapper89_copies3813 a h).2.2.2.1
  have c122020 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.1
  have c122021 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_8 := arithEq_of_rows h (row := 2000) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_8
  simp only [← c122016, ← c122017, ← c122018] at e_2000_8
  have e_2000_9 := arithEq_of_rows h (row := 2000) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_9
  simp only [← c122019, ← c122020, k2, ← c122021] at e_2000_9
  have hr := e_2000_9
  simp only [e_2000_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14586)) := by
  have c122022 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.1
  have c122023 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.1
  have c122024 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.1
  have c122025 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.1
  have c122026 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122027 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_10 := arithEq_of_rows h (row := 2000) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_10
  simp only [← c122022, ← c122023, ← c122024] at e_2000_10
  have e_2000_11 := arithEq_of_rows h (row := 2000) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_11
  simp only [← c122025, ← c122026, k2, ← c122027] at e_2000_11
  have hr := e_2000_11
  simp only [e_2000_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14587)) := by
  have c122028 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122029 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122030 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122031 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122032 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122033 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_12 := arithEq_of_rows h (row := 2000) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_12
  simp only [← c122028, ← c122029, ← c122030] at e_2000_12
  have e_2000_13 := arithEq_of_rows h (row := 2000) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_13
  simp only [← c122031, ← c122032, k2, ← c122033] at e_2000_13
  have hr := e_2000_13
  simp only [e_2000_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14588)) := by
  have c122034 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122035 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122036 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122037 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122038 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122039 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_14 := arithEq_of_rows h (row := 2000) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_14
  simp only [← c122034, ← c122035, ← c122036] at e_2000_14
  have e_2000_15 := arithEq_of_rows h (row := 2000) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_15
  simp only [← c122037, ← c122038, k2, ← c122039] at e_2000_15
  have hr := e_2000_15
  simp only [e_2000_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14589)) := by
  have c122040 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122041 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122042 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122043 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122044 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122045 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_16 := arithEq_of_rows h (row := 2000) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_16
  simp only [← c122040, ← c122041, ← c122042] at e_2000_16
  have e_2000_17 := arithEq_of_rows h (row := 2000) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_17
  simp only [← c122043, ← c122044, k2, ← c122045] at e_2000_17
  have hr := e_2000_17
  simp only [e_2000_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14590)) := by
  have c122046 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122047 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122048 := (publicBatchWrapper89_copies3814 a h).1
  have c122049 := (publicBatchWrapper89_copies3814 a h).2.1
  have c122050 := (publicBatchWrapper89_copies3814 a h).2.2.1
  have c122051 := (publicBatchWrapper89_copies3814 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_18 := arithEq_of_rows h (row := 2000) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_18
  simp only [← c122046, ← c122047, ← c122048] at e_2000_18
  have e_2000_19 := arithEq_of_rows h (row := 2000) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_19
  simp only [← c122049, ← c122050, k2, ← c122051] at e_2000_19
  have hr := e_2000_19
  simp only [e_2000_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14591)) := by
  have c122052 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.1
  have c122053 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.1
  have c122054 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.1
  have c122055 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.1
  have c122056 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.1
  have c122057 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_0 := arithEq_of_rows h (row := 2001) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_0
  simp only [← c122052, ← c122053, ← c122054] at e_2001_0
  have e_2001_1 := arithEq_of_rows h (row := 2001) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_1
  simp only [← c122055, ← c122056, k2, ← c122057] at e_2001_1
  have hr := e_2001_1
  simp only [e_2001_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14592)) := by
  have c122058 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122059 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122060 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122061 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122062 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122063 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_2 := arithEq_of_rows h (row := 2001) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_2
  simp only [← c122058, ← c122059, ← c122060] at e_2001_2
  have e_2001_3 := arithEq_of_rows h (row := 2001) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_3
  simp only [← c122061, ← c122062, k2, ← c122063] at e_2001_3
  have hr := e_2001_3
  simp only [e_2001_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14593)) := by
  have c122064 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122065 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122066 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122067 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122068 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122069 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_4 := arithEq_of_rows h (row := 2001) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_4
  simp only [← c122064, ← c122065, ← c122066] at e_2001_4
  have e_2001_5 := arithEq_of_rows h (row := 2001) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_5
  simp only [← c122067, ← c122068, k2, ← c122069] at e_2001_5
  have hr := e_2001_5
  simp only [e_2001_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14594)) := by
  have c122070 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122071 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122072 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122073 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122074 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122075 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_6 := arithEq_of_rows h (row := 2001) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_6
  simp only [← c122070, ← c122071, ← c122072] at e_2001_6
  have e_2001_7 := arithEq_of_rows h (row := 2001) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_7
  simp only [← c122073, ← c122074, k2, ← c122075] at e_2001_7
  have hr := e_2001_7
  simp only [e_2001_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14595)) := by
  have c122076 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122077 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122078 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122079 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122080 := (publicBatchWrapper89_copies3815 a h).1
  have c122081 := (publicBatchWrapper89_copies3815 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_8 := arithEq_of_rows h (row := 2001) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_8
  simp only [← c122076, ← c122077, ← c122078] at e_2001_8
  have e_2001_9 := arithEq_of_rows h (row := 2001) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_9
  simp only [← c122079, ← c122080, k2, ← c122081] at e_2001_9
  have hr := e_2001_9
  simp only [e_2001_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14596)) := by
  have c122082 := (publicBatchWrapper89_copies3815 a h).2.2.1
  have c122083 := (publicBatchWrapper89_copies3815 a h).2.2.2.1
  have c122084 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.1
  have c122085 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.1
  have c122086 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.1
  have c122087 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_10 := arithEq_of_rows h (row := 2001) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_10
  simp only [← c122082, ← c122083, ← c122084] at e_2001_10
  have e_2001_11 := arithEq_of_rows h (row := 2001) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_11
  simp only [← c122085, ← c122086, k2, ← c122087] at e_2001_11
  have hr := e_2001_11
  simp only [e_2001_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14597)) := by
  have c122088 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.1
  have c122089 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.1
  have c122090 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122091 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122092 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122093 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_12 := arithEq_of_rows h (row := 2001) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_12
  simp only [← c122088, ← c122089, ← c122090] at e_2001_12
  have e_2001_13 := arithEq_of_rows h (row := 2001) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_13
  simp only [← c122091, ← c122092, k2, ← c122093] at e_2001_13
  have hr := e_2001_13
  simp only [e_2001_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14598)) := by
  have c122094 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122095 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122096 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122097 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122098 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122099 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_14 := arithEq_of_rows h (row := 2001) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_14
  simp only [← c122094, ← c122095, ← c122096] at e_2001_14
  have e_2001_15 := arithEq_of_rows h (row := 2001) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_15
  simp only [← c122097, ← c122098, k2, ← c122099] at e_2001_15
  have hr := e_2001_15
  simp only [e_2001_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14599)) := by
  have c122100 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122101 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122102 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122103 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122104 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122105 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_16 := arithEq_of_rows h (row := 2001) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_16
  simp only [← c122100, ← c122101, ← c122102] at e_2001_16
  have e_2001_17 := arithEq_of_rows h (row := 2001) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_17
  simp only [← c122103, ← c122104, k2, ← c122105] at e_2001_17
  have hr := e_2001_17
  simp only [e_2001_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14600)) := by
  have c122106 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122107 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122108 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122109 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122110 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122111 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_18 := arithEq_of_rows h (row := 2001) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_18
  simp only [← c122106, ← c122107, ← c122108] at e_2001_18
  have e_2001_19 := arithEq_of_rows h (row := 2001) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_19
  simp only [← c122109, ← c122110, k2, ← c122111] at e_2001_19
  have hr := e_2001_19
  simp only [e_2001_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14601)) := by
  have c122112 := (publicBatchWrapper89_copies3816 a h).1
  have c122113 := (publicBatchWrapper89_copies3816 a h).2.1
  have c122114 := (publicBatchWrapper89_copies3816 a h).2.2.1
  have c122115 := (publicBatchWrapper89_copies3816 a h).2.2.2.1
  have c122116 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.1
  have c122117 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_0 := arithEq_of_rows h (row := 2002) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_0
  simp only [← c122112, ← c122113, ← c122114] at e_2002_0
  have e_2002_1 := arithEq_of_rows h (row := 2002) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_1
  simp only [← c122115, ← c122116, k2, ← c122117] at e_2002_1
  have hr := e_2002_1
  simp only [e_2002_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14770)) := by
  have c122118 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.1
  have c122119 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.1
  have c122120 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.1
  have c122121 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.1
  have c122122 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122123 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_2 := arithEq_of_rows h (row := 2002) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_2
  simp only [← c122118, ← c122119, ← c122120] at e_2002_2
  have e_2002_3 := arithEq_of_rows h (row := 2002) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_3
  simp only [← c122121, ← c122122, k2, ← c122123] at e_2002_3
  have hr := e_2002_3
  simp only [e_2002_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14771)) := by
  have c122124 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122125 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122126 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122127 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122128 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122129 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_4 := arithEq_of_rows h (row := 2002) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_4
  simp only [← c122124, ← c122125, ← c122126] at e_2002_4
  have e_2002_5 := arithEq_of_rows h (row := 2002) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_5
  simp only [← c122127, ← c122128, k2, ← c122129] at e_2002_5
  have hr := e_2002_5
  simp only [e_2002_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14772)) := by
  have c122130 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122131 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122132 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122133 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122134 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122135 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_6 := arithEq_of_rows h (row := 2002) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_6
  simp only [← c122130, ← c122131, ← c122132] at e_2002_6
  have e_2002_7 := arithEq_of_rows h (row := 2002) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_7
  simp only [← c122133, ← c122134, k2, ← c122135] at e_2002_7
  have hr := e_2002_7
  simp only [e_2002_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14773)) := by
  have c122136 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122137 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122138 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122139 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122140 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122141 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_8 := arithEq_of_rows h (row := 2002) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_8
  simp only [← c122136, ← c122137, ← c122138] at e_2002_8
  have e_2002_9 := arithEq_of_rows h (row := 2002) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_9
  simp only [← c122139, ← c122140, k2, ← c122141] at e_2002_9
  have hr := e_2002_9
  simp only [e_2002_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14774)) := by
  have c122142 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122143 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122144 := (publicBatchWrapper89_copies3817 a h).1
  have c122145 := (publicBatchWrapper89_copies3817 a h).2.1
  have c122146 := (publicBatchWrapper89_copies3817 a h).2.2.1
  have c122147 := (publicBatchWrapper89_copies3817 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_10 := arithEq_of_rows h (row := 2002) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_10
  simp only [← c122142, ← c122143, ← c122144] at e_2002_10
  have e_2002_11 := arithEq_of_rows h (row := 2002) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_11
  simp only [← c122145, ← c122146, k2, ← c122147] at e_2002_11
  have hr := e_2002_11
  simp only [e_2002_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14775)) := by
  have c122148 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.1
  have c122149 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.1
  have c122150 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.1
  have c122151 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.1
  have c122152 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.1
  have c122153 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_12 := arithEq_of_rows h (row := 2002) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_12
  simp only [← c122148, ← c122149, ← c122150] at e_2002_12
  have e_2002_13 := arithEq_of_rows h (row := 2002) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_13
  simp only [← c122151, ← c122152, k2, ← c122153] at e_2002_13
  have hr := e_2002_13
  simp only [e_2002_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14776)) := by
  have c122154 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122155 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122156 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122157 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122158 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122159 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_14 := arithEq_of_rows h (row := 2002) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_14
  simp only [← c122154, ← c122155, ← c122156] at e_2002_14
  have e_2002_15 := arithEq_of_rows h (row := 2002) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_15
  simp only [← c122157, ← c122158, k2, ← c122159] at e_2002_15
  have hr := e_2002_15
  simp only [e_2002_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14777)) := by
  have c122160 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122161 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122162 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122163 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122164 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122165 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_16 := arithEq_of_rows h (row := 2002) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_16
  simp only [← c122160, ← c122161, ← c122162] at e_2002_16
  have e_2002_17 := arithEq_of_rows h (row := 2002) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_17
  simp only [← c122163, ← c122164, k2, ← c122165] at e_2002_17
  have hr := e_2002_17
  simp only [e_2002_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14778)) := by
  have c122166 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122167 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122168 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122169 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122170 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122171 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_18 := arithEq_of_rows h (row := 2002) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_18
  simp only [← c122166, ← c122167, ← c122168] at e_2002_18
  have e_2002_19 := arithEq_of_rows h (row := 2002) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_19
  simp only [← c122169, ← c122170, k2, ← c122171] at e_2002_19
  have hr := e_2002_19
  simp only [e_2002_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14779)) := by
  have c122172 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122173 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122174 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122175 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122176 := (publicBatchWrapper89_copies3818 a h).1
  have c122177 := (publicBatchWrapper89_copies3818 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_0 := arithEq_of_rows h (row := 2003) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_0
  simp only [← c122172, ← c122173, ← c122174] at e_2003_0
  have e_2003_1 := arithEq_of_rows h (row := 2003) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_1
  simp only [← c122175, ← c122176, k2, ← c122177] at e_2003_1
  have hr := e_2003_1
  simp only [e_2003_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14780)) := by
  have c122178 := (publicBatchWrapper89_copies3818 a h).2.2.1
  have c122179 := (publicBatchWrapper89_copies3818 a h).2.2.2.1
  have c122180 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.1
  have c122181 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.1
  have c122182 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.1
  have c122183 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_2 := arithEq_of_rows h (row := 2003) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_2
  simp only [← c122178, ← c122179, ← c122180] at e_2003_2
  have e_2003_3 := arithEq_of_rows h (row := 2003) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_3
  simp only [← c122181, ← c122182, k2, ← c122183] at e_2003_3
  have hr := e_2003_3
  simp only [e_2003_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14781)) := by
  have c122184 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.1
  have c122185 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.1
  have c122186 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122187 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122188 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122189 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_4 := arithEq_of_rows h (row := 2003) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_4
  simp only [← c122184, ← c122185, ← c122186] at e_2003_4
  have e_2003_5 := arithEq_of_rows h (row := 2003) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_5
  simp only [← c122187, ← c122188, k2, ← c122189] at e_2003_5
  have hr := e_2003_5
  simp only [e_2003_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14782)) := by
  have c122190 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122191 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122192 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122193 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122194 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122195 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_6 := arithEq_of_rows h (row := 2003) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_6
  simp only [← c122190, ← c122191, ← c122192] at e_2003_6
  have e_2003_7 := arithEq_of_rows h (row := 2003) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_7
  simp only [← c122193, ← c122194, k2, ← c122195] at e_2003_7
  have hr := e_2003_7
  simp only [e_2003_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14783)) := by
  have c122196 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122197 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122198 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122199 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122200 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122201 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_8 := arithEq_of_rows h (row := 2003) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_8
  simp only [← c122196, ← c122197, ← c122198] at e_2003_8
  have e_2003_9 := arithEq_of_rows h (row := 2003) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_9
  simp only [← c122199, ← c122200, k2, ← c122201] at e_2003_9
  have hr := e_2003_9
  simp only [e_2003_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14784)) := by
  have c122202 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122203 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122204 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122205 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122206 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122207 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_10 := arithEq_of_rows h (row := 2003) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_10
  simp only [← c122202, ← c122203, ← c122204] at e_2003_10
  have e_2003_11 := arithEq_of_rows h (row := 2003) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_11
  simp only [← c122205, ← c122206, k2, ← c122207] at e_2003_11
  have hr := e_2003_11
  simp only [e_2003_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14785)) := by
  have c122208 := (publicBatchWrapper89_copies3819 a h).1
  have c122209 := (publicBatchWrapper89_copies3819 a h).2.1
  have c122210 := (publicBatchWrapper89_copies3819 a h).2.2.1
  have c122211 := (publicBatchWrapper89_copies3819 a h).2.2.2.1
  have c122212 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.1
  have c122213 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_12 := arithEq_of_rows h (row := 2003) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_12
  simp only [← c122208, ← c122209, ← c122210] at e_2003_12
  have e_2003_13 := arithEq_of_rows h (row := 2003) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_13
  simp only [← c122211, ← c122212, k2, ← c122213] at e_2003_13
  have hr := e_2003_13
  simp only [e_2003_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14786)) := by
  have c122214 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.1
  have c122215 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.1
  have c122216 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.1
  have c122217 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.1
  have c122218 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122219 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_14 := arithEq_of_rows h (row := 2003) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_14
  simp only [← c122214, ← c122215, ← c122216] at e_2003_14
  have e_2003_15 := arithEq_of_rows h (row := 2003) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_15
  simp only [← c122217, ← c122218, k2, ← c122219] at e_2003_15
  have hr := e_2003_15
  simp only [e_2003_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14787)) := by
  have c122220 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122221 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122222 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122223 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122224 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122225 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_16 := arithEq_of_rows h (row := 2003) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_16
  simp only [← c122220, ← c122221, ← c122222] at e_2003_16
  have e_2003_17 := arithEq_of_rows h (row := 2003) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_17
  simp only [← c122223, ← c122224, k2, ← c122225] at e_2003_17
  have hr := e_2003_17
  simp only [e_2003_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14788)) := by
  have c122226 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122227 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122228 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122229 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122230 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122231 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_18 := arithEq_of_rows h (row := 2003) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_18
  simp only [← c122226, ← c122227, ← c122228] at e_2003_18
  have e_2003_19 := arithEq_of_rows h (row := 2003) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_19
  simp only [← c122229, ← c122230, k2, ← c122231] at e_2003_19
  have hr := e_2003_19
  simp only [e_2003_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14789)) := by
  have c122232 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122233 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122234 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122235 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122236 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122237 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_0 := arithEq_of_rows h (row := 2004) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_0
  simp only [← c122232, ← c122233, ← c122234] at e_2004_0
  have e_2004_1 := arithEq_of_rows h (row := 2004) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_1
  simp only [← c122235, ← c122236, k2, ← c122237] at e_2004_1
  have hr := e_2004_1
  simp only [e_2004_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14790)) := by
  have c122238 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122239 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122240 := (publicBatchWrapper89_copies3820 a h).1
  have c122241 := (publicBatchWrapper89_copies3820 a h).2.1
  have c122242 := (publicBatchWrapper89_copies3820 a h).2.2.1
  have c122243 := (publicBatchWrapper89_copies3820 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_2 := arithEq_of_rows h (row := 2004) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_2
  simp only [← c122238, ← c122239, ← c122240] at e_2004_2
  have e_2004_3 := arithEq_of_rows h (row := 2004) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_3
  simp only [← c122241, ← c122242, k2, ← c122243] at e_2004_3
  have hr := e_2004_3
  simp only [e_2004_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14791)) := by
  have c122244 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.1
  have c122245 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.1
  have c122246 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.1
  have c122247 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.1
  have c122248 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.1
  have c122249 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_4 := arithEq_of_rows h (row := 2004) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_4
  simp only [← c122244, ← c122245, ← c122246] at e_2004_4
  have e_2004_5 := arithEq_of_rows h (row := 2004) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_5
  simp only [← c122247, ← c122248, k2, ← c122249] at e_2004_5
  have hr := e_2004_5
  simp only [e_2004_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14792)) := by
  have c122250 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122251 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122252 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122253 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122254 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122255 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_6 := arithEq_of_rows h (row := 2004) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_6
  simp only [← c122250, ← c122251, ← c122252] at e_2004_6
  have e_2004_7 := arithEq_of_rows h (row := 2004) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_7
  simp only [← c122253, ← c122254, k2, ← c122255] at e_2004_7
  have hr := e_2004_7
  simp only [e_2004_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14793)) := by
  have c122256 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122257 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122258 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122259 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122260 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122261 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_8 := arithEq_of_rows h (row := 2004) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_8
  simp only [← c122256, ← c122257, ← c122258] at e_2004_8
  have e_2004_9 := arithEq_of_rows h (row := 2004) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_9
  simp only [← c122259, ← c122260, k2, ← c122261] at e_2004_9
  have hr := e_2004_9
  simp only [e_2004_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14794)) := by
  have c122262 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122263 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122264 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122265 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122266 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122267 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_10 := arithEq_of_rows h (row := 2004) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_10
  simp only [← c122262, ← c122263, ← c122264] at e_2004_10
  have e_2004_11 := arithEq_of_rows h (row := 2004) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_11
  simp only [← c122265, ← c122266, k2, ← c122267] at e_2004_11
  have hr := e_2004_11
  simp only [e_2004_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14795)) := by
  have c122268 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122269 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122270 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122271 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122272 := (publicBatchWrapper89_copies3821 a h).1
  have c122273 := (publicBatchWrapper89_copies3821 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_12 := arithEq_of_rows h (row := 2004) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_12
  simp only [← c122268, ← c122269, ← c122270] at e_2004_12
  have e_2004_13 := arithEq_of_rows h (row := 2004) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_13
  simp only [← c122271, ← c122272, k2, ← c122273] at e_2004_13
  have hr := e_2004_13
  simp only [e_2004_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14796)) := by
  have c122274 := (publicBatchWrapper89_copies3821 a h).2.2.1
  have c122275 := (publicBatchWrapper89_copies3821 a h).2.2.2.1
  have c122276 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.1
  have c122277 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.1
  have c122278 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.1
  have c122279 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_14 := arithEq_of_rows h (row := 2004) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_14
  simp only [← c122274, ← c122275, ← c122276] at e_2004_14
  have e_2004_15 := arithEq_of_rows h (row := 2004) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_15
  simp only [← c122277, ← c122278, k2, ← c122279] at e_2004_15
  have hr := e_2004_15
  simp only [e_2004_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14797)) := by
  have c122280 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.1
  have c122281 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.1
  have c122282 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122283 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122284 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122285 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_16 := arithEq_of_rows h (row := 2004) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_16
  simp only [← c122280, ← c122281, ← c122282] at e_2004_16
  have e_2004_17 := arithEq_of_rows h (row := 2004) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_17
  simp only [← c122283, ← c122284, k2, ← c122285] at e_2004_17
  have hr := e_2004_17
  simp only [e_2004_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14798)) := by
  have c122286 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122287 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122288 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122289 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122290 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122291 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_18 := arithEq_of_rows h (row := 2004) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_18
  simp only [← c122286, ← c122287, ← c122288] at e_2004_18
  have e_2004_19 := arithEq_of_rows h (row := 2004) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_19
  simp only [← c122289, ← c122290, k2, ← c122291] at e_2004_19
  have hr := e_2004_19
  simp only [e_2004_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14799)) := by
  have c122292 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122293 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122294 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122295 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122296 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122297 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_0 := arithEq_of_rows h (row := 2005) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_0
  simp only [← c122292, ← c122293, ← c122294] at e_2005_0
  have e_2005_1 := arithEq_of_rows h (row := 2005) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_1
  simp only [← c122295, ← c122296, k2, ← c122297] at e_2005_1
  have hr := e_2005_1
  simp only [e_2005_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14800)) := by
  have c122298 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122299 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122300 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122301 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122302 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122303 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_2 := arithEq_of_rows h (row := 2005) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_2
  simp only [← c122298, ← c122299, ← c122300] at e_2005_2
  have e_2005_3 := arithEq_of_rows h (row := 2005) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_3
  simp only [← c122301, ← c122302, k2, ← c122303] at e_2005_3
  have hr := e_2005_3
  simp only [e_2005_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14801)) := by
  have c122304 := (publicBatchWrapper89_copies3822 a h).1
  have c122305 := (publicBatchWrapper89_copies3822 a h).2.1
  have c122306 := (publicBatchWrapper89_copies3822 a h).2.2.1
  have c122307 := (publicBatchWrapper89_copies3822 a h).2.2.2.1
  have c122308 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.1
  have c122309 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_4 := arithEq_of_rows h (row := 2005) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_4
  simp only [← c122304, ← c122305, ← c122306] at e_2005_4
  have e_2005_5 := arithEq_of_rows h (row := 2005) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_5
  simp only [← c122307, ← c122308, k2, ← c122309] at e_2005_5
  have hr := e_2005_5
  simp only [e_2005_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14970)) := by
  have c122310 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.1
  have c122311 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.1
  have c122312 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.1
  have c122313 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.1
  have c122314 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122315 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_6 := arithEq_of_rows h (row := 2005) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_6
  simp only [← c122310, ← c122311, ← c122312] at e_2005_6
  have e_2005_7 := arithEq_of_rows h (row := 2005) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_7
  simp only [← c122313, ← c122314, k2, ← c122315] at e_2005_7
  have hr := e_2005_7
  simp only [e_2005_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14971)) := by
  have c122316 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122317 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122318 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122319 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122320 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122321 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_8 := arithEq_of_rows h (row := 2005) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_8
  simp only [← c122316, ← c122317, ← c122318] at e_2005_8
  have e_2005_9 := arithEq_of_rows h (row := 2005) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_9
  simp only [← c122319, ← c122320, k2, ← c122321] at e_2005_9
  have hr := e_2005_9
  simp only [e_2005_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14972)) := by
  have c122322 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122323 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122324 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122325 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122326 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122327 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_10 := arithEq_of_rows h (row := 2005) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_10
  simp only [← c122322, ← c122323, ← c122324] at e_2005_10
  have e_2005_11 := arithEq_of_rows h (row := 2005) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_11
  simp only [← c122325, ← c122326, k2, ← c122327] at e_2005_11
  have hr := e_2005_11
  simp only [e_2005_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14973)) := by
  have c122328 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122329 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122330 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122331 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122332 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122333 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_12 := arithEq_of_rows h (row := 2005) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_12
  simp only [← c122328, ← c122329, ← c122330] at e_2005_12
  have e_2005_13 := arithEq_of_rows h (row := 2005) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_13
  simp only [← c122331, ← c122332, k2, ← c122333] at e_2005_13
  have hr := e_2005_13
  simp only [e_2005_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14974)) := by
  have c122334 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122335 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122336 := (publicBatchWrapper89_copies3823 a h).1
  have c122337 := (publicBatchWrapper89_copies3823 a h).2.1
  have c122338 := (publicBatchWrapper89_copies3823 a h).2.2.1
  have c122339 := (publicBatchWrapper89_copies3823 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_14 := arithEq_of_rows h (row := 2005) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_14
  simp only [← c122334, ← c122335, ← c122336] at e_2005_14
  have e_2005_15 := arithEq_of_rows h (row := 2005) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_15
  simp only [← c122337, ← c122338, k2, ← c122339] at e_2005_15
  have hr := e_2005_15
  simp only [e_2005_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14975)) := by
  have c122340 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.1
  have c122341 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.1
  have c122342 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.1
  have c122343 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.1
  have c122344 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.1
  have c122345 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_16 := arithEq_of_rows h (row := 2005) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_16
  simp only [← c122340, ← c122341, ← c122342] at e_2005_16
  have e_2005_17 := arithEq_of_rows h (row := 2005) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_17
  simp only [← c122343, ← c122344, k2, ← c122345] at e_2005_17
  have hr := e_2005_17
  simp only [e_2005_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14976)) := by
  have c122346 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122347 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122348 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122349 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122350 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122351 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_18 := arithEq_of_rows h (row := 2005) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_18
  simp only [← c122346, ← c122347, ← c122348] at e_2005_18
  have e_2005_19 := arithEq_of_rows h (row := 2005) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_19
  simp only [← c122349, ← c122350, k2, ← c122351] at e_2005_19
  have hr := e_2005_19
  simp only [e_2005_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14977)) := by
  have c122352 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122353 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122354 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122355 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122356 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122357 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_0 := arithEq_of_rows h (row := 2006) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_0
  simp only [← c122352, ← c122353, ← c122354] at e_2006_0
  have e_2006_1 := arithEq_of_rows h (row := 2006) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_1
  simp only [← c122355, ← c122356, k2, ← c122357] at e_2006_1
  have hr := e_2006_1
  simp only [e_2006_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14978)) := by
  have c122358 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122359 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122360 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122361 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122362 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122363 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_2 := arithEq_of_rows h (row := 2006) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_2
  simp only [← c122358, ← c122359, ← c122360] at e_2006_2
  have e_2006_3 := arithEq_of_rows h (row := 2006) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_3
  simp only [← c122361, ← c122362, k2, ← c122363] at e_2006_3
  have hr := e_2006_3
  simp only [e_2006_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14979)) := by
  have c122364 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122365 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122366 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122367 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122368 := (publicBatchWrapper89_copies3824 a h).1
  have c122369 := (publicBatchWrapper89_copies3824 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_4 := arithEq_of_rows h (row := 2006) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_4
  simp only [← c122364, ← c122365, ← c122366] at e_2006_4
  have e_2006_5 := arithEq_of_rows h (row := 2006) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_5
  simp only [← c122367, ← c122368, k2, ← c122369] at e_2006_5
  have hr := e_2006_5
  simp only [e_2006_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14980)) := by
  have c122370 := (publicBatchWrapper89_copies3824 a h).2.2.1
  have c122371 := (publicBatchWrapper89_copies3824 a h).2.2.2.1
  have c122372 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.1
  have c122373 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.1
  have c122374 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.1
  have c122375 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_6 := arithEq_of_rows h (row := 2006) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_6
  simp only [← c122370, ← c122371, ← c122372] at e_2006_6
  have e_2006_7 := arithEq_of_rows h (row := 2006) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_7
  simp only [← c122373, ← c122374, k2, ← c122375] at e_2006_7
  have hr := e_2006_7
  simp only [e_2006_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14981)) := by
  have c122376 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.1
  have c122377 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.1
  have c122378 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122379 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122380 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122381 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_8 := arithEq_of_rows h (row := 2006) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_8
  simp only [← c122376, ← c122377, ← c122378] at e_2006_8
  have e_2006_9 := arithEq_of_rows h (row := 2006) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_9
  simp only [← c122379, ← c122380, k2, ← c122381] at e_2006_9
  have hr := e_2006_9
  simp only [e_2006_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14982)) := by
  have c122382 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122383 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122384 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122385 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122386 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122387 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_10 := arithEq_of_rows h (row := 2006) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_10
  simp only [← c122382, ← c122383, ← c122384] at e_2006_10
  have e_2006_11 := arithEq_of_rows h (row := 2006) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_11
  simp only [← c122385, ← c122386, k2, ← c122387] at e_2006_11
  have hr := e_2006_11
  simp only [e_2006_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14983)) := by
  have c122388 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122389 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122390 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122391 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122392 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122393 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_12 := arithEq_of_rows h (row := 2006) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_12
  simp only [← c122388, ← c122389, ← c122390] at e_2006_12
  have e_2006_13 := arithEq_of_rows h (row := 2006) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_13
  simp only [← c122391, ← c122392, k2, ← c122393] at e_2006_13
  have hr := e_2006_13
  simp only [e_2006_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14984)) := by
  have c122394 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122395 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122396 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122397 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122398 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122399 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_14 := arithEq_of_rows h (row := 2006) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_14
  simp only [← c122394, ← c122395, ← c122396] at e_2006_14
  have e_2006_15 := arithEq_of_rows h (row := 2006) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_15
  simp only [← c122397, ← c122398, k2, ← c122399] at e_2006_15
  have hr := e_2006_15
  simp only [e_2006_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14985)) := by
  have c122400 := (publicBatchWrapper89_copies3825 a h).1
  have c122401 := (publicBatchWrapper89_copies3825 a h).2.1
  have c122402 := (publicBatchWrapper89_copies3825 a h).2.2.1
  have c122403 := (publicBatchWrapper89_copies3825 a h).2.2.2.1
  have c122404 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.1
  have c122405 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_16 := arithEq_of_rows h (row := 2006) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_16
  simp only [← c122400, ← c122401, ← c122402] at e_2006_16
  have e_2006_17 := arithEq_of_rows h (row := 2006) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_17
  simp only [← c122403, ← c122404, k2, ← c122405] at e_2006_17
  have hr := e_2006_17
  simp only [e_2006_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14986)) := by
  have c122406 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.1
  have c122407 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.1
  have c122408 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.1
  have c122409 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.1
  have c122410 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122411 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_18 := arithEq_of_rows h (row := 2006) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_18
  simp only [← c122406, ← c122407, ← c122408] at e_2006_18
  have e_2006_19 := arithEq_of_rows h (row := 2006) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_19
  simp only [← c122409, ← c122410, k2, ← c122411] at e_2006_19
  have hr := e_2006_19
  simp only [e_2006_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14987)) := by
  have c122412 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122413 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122414 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122415 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122416 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122417 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_0 := arithEq_of_rows h (row := 2007) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_0
  simp only [← c122412, ← c122413, ← c122414] at e_2007_0
  have e_2007_1 := arithEq_of_rows h (row := 2007) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_1
  simp only [← c122415, ← c122416, k2, ← c122417] at e_2007_1
  have hr := e_2007_1
  simp only [e_2007_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14988)) := by
  have c122418 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122419 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122420 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122421 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122422 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122423 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_2 := arithEq_of_rows h (row := 2007) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_2
  simp only [← c122418, ← c122419, ← c122420] at e_2007_2
  have e_2007_3 := arithEq_of_rows h (row := 2007) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_3
  simp only [← c122421, ← c122422, k2, ← c122423] at e_2007_3
  have hr := e_2007_3
  simp only [e_2007_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14989)) := by
  have c122424 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122425 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122426 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122427 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122428 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122429 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_4 := arithEq_of_rows h (row := 2007) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_4
  simp only [← c122424, ← c122425, ← c122426] at e_2007_4
  have e_2007_5 := arithEq_of_rows h (row := 2007) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_5
  simp only [← c122427, ← c122428, k2, ← c122429] at e_2007_5
  have hr := e_2007_5
  simp only [e_2007_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14990)) := by
  have c122430 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122431 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122432 := (publicBatchWrapper89_copies3826 a h).1
  have c122433 := (publicBatchWrapper89_copies3826 a h).2.1
  have c122434 := (publicBatchWrapper89_copies3826 a h).2.2.1
  have c122435 := (publicBatchWrapper89_copies3826 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_6 := arithEq_of_rows h (row := 2007) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_6
  simp only [← c122430, ← c122431, ← c122432] at e_2007_6
  have e_2007_7 := arithEq_of_rows h (row := 2007) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_7
  simp only [← c122433, ← c122434, k2, ← c122435] at e_2007_7
  have hr := e_2007_7
  simp only [e_2007_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14991)) := by
  have c122436 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.1
  have c122437 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.1
  have c122438 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.1
  have c122439 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.1
  have c122440 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.1
  have c122441 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_8 := arithEq_of_rows h (row := 2007) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_8
  simp only [← c122436, ← c122437, ← c122438] at e_2007_8
  have e_2007_9 := arithEq_of_rows h (row := 2007) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_9
  simp only [← c122439, ← c122440, k2, ← c122441] at e_2007_9
  have hr := e_2007_9
  simp only [e_2007_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14992)) := by
  have c122442 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122443 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122444 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122445 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122446 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122447 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_10 := arithEq_of_rows h (row := 2007) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_10
  simp only [← c122442, ← c122443, ← c122444] at e_2007_10
  have e_2007_11 := arithEq_of_rows h (row := 2007) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_11
  simp only [← c122445, ← c122446, k2, ← c122447] at e_2007_11
  have hr := e_2007_11
  simp only [e_2007_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14993)) := by
  have c122448 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122449 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122450 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122451 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122452 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122453 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_12 := arithEq_of_rows h (row := 2007) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_12
  simp only [← c122448, ← c122449, ← c122450] at e_2007_12
  have e_2007_13 := arithEq_of_rows h (row := 2007) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_13
  simp only [← c122451, ← c122452, k2, ← c122453] at e_2007_13
  have hr := e_2007_13
  simp only [e_2007_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14994)) := by
  have c122454 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122455 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122456 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122457 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122458 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122459 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_14 := arithEq_of_rows h (row := 2007) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_14
  simp only [← c122454, ← c122455, ← c122456] at e_2007_14
  have e_2007_15 := arithEq_of_rows h (row := 2007) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_15
  simp only [← c122457, ← c122458, k2, ← c122459] at e_2007_15
  have hr := e_2007_15
  simp only [e_2007_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14995)) := by
  have c122460 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122461 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122462 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122463 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122464 := (publicBatchWrapper89_copies3827 a h).1
  have c122465 := (publicBatchWrapper89_copies3827 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_16 := arithEq_of_rows h (row := 2007) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_16
  simp only [← c122460, ← c122461, ← c122462] at e_2007_16
  have e_2007_17 := arithEq_of_rows h (row := 2007) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_17
  simp only [← c122463, ← c122464, k2, ← c122465] at e_2007_17
  have hr := e_2007_17
  simp only [e_2007_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14996)) := by
  have c122466 := (publicBatchWrapper89_copies3827 a h).2.2.1
  have c122467 := (publicBatchWrapper89_copies3827 a h).2.2.2.1
  have c122468 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.1
  have c122469 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.1
  have c122470 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.1
  have c122471 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_18 := arithEq_of_rows h (row := 2007) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_18
  simp only [← c122466, ← c122467, ← c122468] at e_2007_18
  have e_2007_19 := arithEq_of_rows h (row := 2007) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_19
  simp only [← c122469, ← c122470, k2, ← c122471] at e_2007_19
  have hr := e_2007_19
  simp only [e_2007_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14997)) := by
  have c122472 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.1
  have c122473 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.1
  have c122474 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122475 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122476 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122477 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_0 := arithEq_of_rows h (row := 2008) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_0
  simp only [← c122472, ← c122473, ← c122474] at e_2008_0
  have e_2008_1 := arithEq_of_rows h (row := 2008) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_1
  simp only [← c122475, ← c122476, k2, ← c122477] at e_2008_1
  have hr := e_2008_1
  simp only [e_2008_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14998)) := by
  have c122478 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122479 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122480 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122481 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122482 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122483 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_2 := arithEq_of_rows h (row := 2008) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_2
  simp only [← c122478, ← c122479, ← c122480] at e_2008_2
  have e_2008_3 := arithEq_of_rows h (row := 2008) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_3
  simp only [← c122481, ← c122482, k2, ← c122483] at e_2008_3
  have hr := e_2008_3
  simp only [e_2008_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14999)) := by
  have c122484 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122485 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122486 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122487 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122488 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122489 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_4 := arithEq_of_rows h (row := 2008) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_4
  simp only [← c122484, ← c122485, ← c122486] at e_2008_4
  have e_2008_5 := arithEq_of_rows h (row := 2008) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_5
  simp only [← c122487, ← c122488, k2, ← c122489] at e_2008_5
  have hr := e_2008_5
  simp only [e_2008_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 15000)) := by
  have c122490 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122491 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122492 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122493 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122494 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122495 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_6 := arithEq_of_rows h (row := 2008) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_6
  simp only [← c122490, ← c122491, ← c122492] at e_2008_6
  have e_2008_7 := arithEq_of_rows h (row := 2008) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_7
  simp only [← c122493, ← c122494, k2, ← c122495] at e_2008_7
  have hr := e_2008_7
  simp only [e_2008_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 15001)) := by
  have c122496 := (publicBatchWrapper89_copies3828 a h).1
  have c122497 := (publicBatchWrapper89_copies3828 a h).2.1
  have c122498 := (publicBatchWrapper89_copies3828 a h).2.2.1
  have c122499 := (publicBatchWrapper89_copies3828 a h).2.2.2.1
  have c122500 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.1
  have c122501 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_8 := arithEq_of_rows h (row := 2008) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_8
  simp only [← c122496, ← c122497, ← c122498] at e_2008_8
  have e_2008_9 := arithEq_of_rows h (row := 2008) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_9
  simp only [← c122499, ← c122500, k2, ← c122501] at e_2008_9
  have hr := e_2008_9
  simp only [e_2008_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15170)) := by
  have c122502 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.1
  have c122503 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.1
  have c122504 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.1
  have c122505 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.1
  have c122506 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122507 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_10 := arithEq_of_rows h (row := 2008) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_10
  simp only [← c122502, ← c122503, ← c122504] at e_2008_10
  have e_2008_11 := arithEq_of_rows h (row := 2008) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_11
  simp only [← c122505, ← c122506, k2, ← c122507] at e_2008_11
  have hr := e_2008_11
  simp only [e_2008_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15171)) := by
  have c122508 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122509 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122510 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122511 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122512 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122513 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_12 := arithEq_of_rows h (row := 2008) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_12
  simp only [← c122508, ← c122509, ← c122510] at e_2008_12
  have e_2008_13 := arithEq_of_rows h (row := 2008) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_13
  simp only [← c122511, ← c122512, k2, ← c122513] at e_2008_13
  have hr := e_2008_13
  simp only [e_2008_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15172)) := by
  have c122514 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122515 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122516 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122517 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122518 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122519 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_14 := arithEq_of_rows h (row := 2008) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_14
  simp only [← c122514, ← c122515, ← c122516] at e_2008_14
  have e_2008_15 := arithEq_of_rows h (row := 2008) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_15
  simp only [← c122517, ← c122518, k2, ← c122519] at e_2008_15
  have hr := e_2008_15
  simp only [e_2008_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15173)) := by
  have c122520 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122521 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122522 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122523 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122524 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122525 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_16 := arithEq_of_rows h (row := 2008) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_16
  simp only [← c122520, ← c122521, ← c122522] at e_2008_16
  have e_2008_17 := arithEq_of_rows h (row := 2008) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_17
  simp only [← c122523, ← c122524, k2, ← c122525] at e_2008_17
  have hr := e_2008_17
  simp only [e_2008_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15174)) := by
  have c122526 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122527 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122528 := (publicBatchWrapper89_copies3829 a h).1
  have c122529 := (publicBatchWrapper89_copies3829 a h).2.1
  have c122530 := (publicBatchWrapper89_copies3829 a h).2.2.1
  have c122531 := (publicBatchWrapper89_copies3829 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_18 := arithEq_of_rows h (row := 2008) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_18
  simp only [← c122526, ← c122527, ← c122528] at e_2008_18
  have e_2008_19 := arithEq_of_rows h (row := 2008) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_19
  simp only [← c122529, ← c122530, k2, ← c122531] at e_2008_19
  have hr := e_2008_19
  simp only [e_2008_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15175)) := by
  have c122532 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.1
  have c122533 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.1
  have c122534 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.1
  have c122535 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.1
  have c122536 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.1
  have c122537 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_0 := arithEq_of_rows h (row := 2009) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_0
  simp only [← c122532, ← c122533, ← c122534] at e_2009_0
  have e_2009_1 := arithEq_of_rows h (row := 2009) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_1
  simp only [← c122535, ← c122536, k2, ← c122537] at e_2009_1
  have hr := e_2009_1
  simp only [e_2009_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15176)) := by
  have c122538 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122539 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122540 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122541 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122542 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122543 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_2 := arithEq_of_rows h (row := 2009) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_2
  simp only [← c122538, ← c122539, ← c122540] at e_2009_2
  have e_2009_3 := arithEq_of_rows h (row := 2009) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_3
  simp only [← c122541, ← c122542, k2, ← c122543] at e_2009_3
  have hr := e_2009_3
  simp only [e_2009_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15177)) := by
  have c122544 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122545 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122546 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122547 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122548 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122549 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_4 := arithEq_of_rows h (row := 2009) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_4
  simp only [← c122544, ← c122545, ← c122546] at e_2009_4
  have e_2009_5 := arithEq_of_rows h (row := 2009) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_5
  simp only [← c122547, ← c122548, k2, ← c122549] at e_2009_5
  have hr := e_2009_5
  simp only [e_2009_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15178)) := by
  have c122550 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122551 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122552 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122553 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122554 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122555 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_6 := arithEq_of_rows h (row := 2009) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_6
  simp only [← c122550, ← c122551, ← c122552] at e_2009_6
  have e_2009_7 := arithEq_of_rows h (row := 2009) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_7
  simp only [← c122553, ← c122554, k2, ← c122555] at e_2009_7
  have hr := e_2009_7
  simp only [e_2009_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15179)) := by
  have c122556 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122557 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122558 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122559 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122560 := (publicBatchWrapper89_copies3830 a h).1
  have c122561 := (publicBatchWrapper89_copies3830 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_8 := arithEq_of_rows h (row := 2009) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_8
  simp only [← c122556, ← c122557, ← c122558] at e_2009_8
  have e_2009_9 := arithEq_of_rows h (row := 2009) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_9
  simp only [← c122559, ← c122560, k2, ← c122561] at e_2009_9
  have hr := e_2009_9
  simp only [e_2009_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15180)) := by
  have c122562 := (publicBatchWrapper89_copies3830 a h).2.2.1
  have c122563 := (publicBatchWrapper89_copies3830 a h).2.2.2.1
  have c122564 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.1
  have c122565 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.1
  have c122566 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.1
  have c122567 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_10 := arithEq_of_rows h (row := 2009) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_10
  simp only [← c122562, ← c122563, ← c122564] at e_2009_10
  have e_2009_11 := arithEq_of_rows h (row := 2009) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_11
  simp only [← c122565, ← c122566, k2, ← c122567] at e_2009_11
  have hr := e_2009_11
  simp only [e_2009_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15181)) := by
  have c122568 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.1
  have c122569 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.1
  have c122570 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122571 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122572 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122573 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_12 := arithEq_of_rows h (row := 2009) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_12
  simp only [← c122568, ← c122569, ← c122570] at e_2009_12
  have e_2009_13 := arithEq_of_rows h (row := 2009) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_13
  simp only [← c122571, ← c122572, k2, ← c122573] at e_2009_13
  have hr := e_2009_13
  simp only [e_2009_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15182)) := by
  have c122574 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122575 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122576 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122577 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122578 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122579 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_14 := arithEq_of_rows h (row := 2009) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_14
  simp only [← c122574, ← c122575, ← c122576] at e_2009_14
  have e_2009_15 := arithEq_of_rows h (row := 2009) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_15
  simp only [← c122577, ← c122578, k2, ← c122579] at e_2009_15
  have hr := e_2009_15
  simp only [e_2009_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15183)) := by
  have c122580 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122581 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122582 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122583 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122584 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122585 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_16 := arithEq_of_rows h (row := 2009) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_16
  simp only [← c122580, ← c122581, ← c122582] at e_2009_16
  have e_2009_17 := arithEq_of_rows h (row := 2009) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_17
  simp only [← c122583, ← c122584, k2, ← c122585] at e_2009_17
  have hr := e_2009_17
  simp only [e_2009_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15184)) := by
  have c122586 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122587 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122588 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122589 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122590 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122591 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_18 := arithEq_of_rows h (row := 2009) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_18
  simp only [← c122586, ← c122587, ← c122588] at e_2009_18
  have e_2009_19 := arithEq_of_rows h (row := 2009) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_19
  simp only [← c122589, ← c122590, k2, ← c122591] at e_2009_19
  have hr := e_2009_19
  simp only [e_2009_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15185)) := by
  have c122592 := (publicBatchWrapper89_copies3831 a h).1
  have c122593 := (publicBatchWrapper89_copies3831 a h).2.1
  have c122594 := (publicBatchWrapper89_copies3831 a h).2.2.1
  have c122595 := (publicBatchWrapper89_copies3831 a h).2.2.2.1
  have c122596 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.1
  have c122597 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_0 := arithEq_of_rows h (row := 2010) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_0
  simp only [← c122592, ← c122593, ← c122594] at e_2010_0
  have e_2010_1 := arithEq_of_rows h (row := 2010) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_1
  simp only [← c122595, ← c122596, k2, ← c122597] at e_2010_1
  have hr := e_2010_1
  simp only [e_2010_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15186)) := by
  have c122598 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.1
  have c122599 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.1
  have c122600 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.1
  have c122601 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.1
  have c122602 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122603 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_2 := arithEq_of_rows h (row := 2010) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_2
  simp only [← c122598, ← c122599, ← c122600] at e_2010_2
  have e_2010_3 := arithEq_of_rows h (row := 2010) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_3
  simp only [← c122601, ← c122602, k2, ← c122603] at e_2010_3
  have hr := e_2010_3
  simp only [e_2010_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15187)) := by
  have c122604 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122605 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122606 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122607 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122608 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122609 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_4 := arithEq_of_rows h (row := 2010) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_4
  simp only [← c122604, ← c122605, ← c122606] at e_2010_4
  have e_2010_5 := arithEq_of_rows h (row := 2010) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_5
  simp only [← c122607, ← c122608, k2, ← c122609] at e_2010_5
  have hr := e_2010_5
  simp only [e_2010_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15188)) := by
  have c122610 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122611 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122612 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122613 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122614 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122615 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_6 := arithEq_of_rows h (row := 2010) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_6
  simp only [← c122610, ← c122611, ← c122612] at e_2010_6
  have e_2010_7 := arithEq_of_rows h (row := 2010) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_7
  simp only [← c122613, ← c122614, k2, ← c122615] at e_2010_7
  have hr := e_2010_7
  simp only [e_2010_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15189)) := by
  have c122616 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122617 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122618 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122619 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122620 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122621 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_8 := arithEq_of_rows h (row := 2010) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_8
  simp only [← c122616, ← c122617, ← c122618] at e_2010_8
  have e_2010_9 := arithEq_of_rows h (row := 2010) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_9
  simp only [← c122619, ← c122620, k2, ← c122621] at e_2010_9
  have hr := e_2010_9
  simp only [e_2010_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15190)) := by
  have c122622 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122623 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122624 := (publicBatchWrapper89_copies3832 a h).1
  have c122625 := (publicBatchWrapper89_copies3832 a h).2.1
  have c122626 := (publicBatchWrapper89_copies3832 a h).2.2.1
  have c122627 := (publicBatchWrapper89_copies3832 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_10 := arithEq_of_rows h (row := 2010) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_10
  simp only [← c122622, ← c122623, ← c122624] at e_2010_10
  have e_2010_11 := arithEq_of_rows h (row := 2010) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_11
  simp only [← c122625, ← c122626, k2, ← c122627] at e_2010_11
  have hr := e_2010_11
  simp only [e_2010_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15191)) := by
  have c122628 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.1
  have c122629 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.1
  have c122630 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.1
  have c122631 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.1
  have c122632 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.1
  have c122633 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_12 := arithEq_of_rows h (row := 2010) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_12
  simp only [← c122628, ← c122629, ← c122630] at e_2010_12
  have e_2010_13 := arithEq_of_rows h (row := 2010) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_13
  simp only [← c122631, ← c122632, k2, ← c122633] at e_2010_13
  have hr := e_2010_13
  simp only [e_2010_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15192)) := by
  have c122634 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122635 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122636 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122637 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122638 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122639 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_14 := arithEq_of_rows h (row := 2010) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_14
  simp only [← c122634, ← c122635, ← c122636] at e_2010_14
  have e_2010_15 := arithEq_of_rows h (row := 2010) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_15
  simp only [← c122637, ← c122638, k2, ← c122639] at e_2010_15
  have hr := e_2010_15
  simp only [e_2010_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15193)) := by
  have c122640 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122641 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122642 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122643 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122644 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122645 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_16 := arithEq_of_rows h (row := 2010) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_16
  simp only [← c122640, ← c122641, ← c122642] at e_2010_16
  have e_2010_17 := arithEq_of_rows h (row := 2010) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_17
  simp only [← c122643, ← c122644, k2, ← c122645] at e_2010_17
  have hr := e_2010_17
  simp only [e_2010_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15194)) := by
  have c122646 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122647 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122648 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122649 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122650 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122651 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_18 := arithEq_of_rows h (row := 2010) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_18
  simp only [← c122646, ← c122647, ← c122648] at e_2010_18
  have e_2010_19 := arithEq_of_rows h (row := 2010) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_19
  simp only [← c122649, ← c122650, k2, ← c122651] at e_2010_19
  have hr := e_2010_19
  simp only [e_2010_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15195)) := by
  have c122652 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122653 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122654 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122655 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122656 := (publicBatchWrapper89_copies3833 a h).1
  have c122657 := (publicBatchWrapper89_copies3833 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_0 := arithEq_of_rows h (row := 2011) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_0
  simp only [← c122652, ← c122653, ← c122654] at e_2011_0
  have e_2011_1 := arithEq_of_rows h (row := 2011) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_1
  simp only [← c122655, ← c122656, k2, ← c122657] at e_2011_1
  have hr := e_2011_1
  simp only [e_2011_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15196)) := by
  have c122658 := (publicBatchWrapper89_copies3833 a h).2.2.1
  have c122659 := (publicBatchWrapper89_copies3833 a h).2.2.2.1
  have c122660 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.1
  have c122661 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.1
  have c122662 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.1
  have c122663 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_2 := arithEq_of_rows h (row := 2011) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_2
  simp only [← c122658, ← c122659, ← c122660] at e_2011_2
  have e_2011_3 := arithEq_of_rows h (row := 2011) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_3
  simp only [← c122661, ← c122662, k2, ← c122663] at e_2011_3
  have hr := e_2011_3
  simp only [e_2011_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15197)) := by
  have c122664 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.1
  have c122665 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.1
  have c122666 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122667 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122668 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122669 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_4 := arithEq_of_rows h (row := 2011) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_4
  simp only [← c122664, ← c122665, ← c122666] at e_2011_4
  have e_2011_5 := arithEq_of_rows h (row := 2011) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_5
  simp only [← c122667, ← c122668, k2, ← c122669] at e_2011_5
  have hr := e_2011_5
  simp only [e_2011_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15198)) := by
  have c122670 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122671 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122672 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122673 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122674 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122675 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_6 := arithEq_of_rows h (row := 2011) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_6
  simp only [← c122670, ← c122671, ← c122672] at e_2011_6
  have e_2011_7 := arithEq_of_rows h (row := 2011) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_7
  simp only [← c122673, ← c122674, k2, ← c122675] at e_2011_7
  have hr := e_2011_7
  simp only [e_2011_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15199)) := by
  have c122676 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122677 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122678 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122679 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122680 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122681 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_8 := arithEq_of_rows h (row := 2011) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_8
  simp only [← c122676, ← c122677, ← c122678] at e_2011_8
  have e_2011_9 := arithEq_of_rows h (row := 2011) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_9
  simp only [← c122679, ← c122680, k2, ← c122681] at e_2011_9
  have hr := e_2011_9
  simp only [e_2011_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15200)) := by
  have c122682 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122683 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122684 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122685 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122686 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122687 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_10 := arithEq_of_rows h (row := 2011) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_10
  simp only [← c122682, ← c122683, ← c122684] at e_2011_10
  have e_2011_11 := arithEq_of_rows h (row := 2011) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_11
  simp only [← c122685, ← c122686, k2, ← c122687] at e_2011_11
  have hr := e_2011_11
  simp only [e_2011_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15201)) := by
  have c122688 := (publicBatchWrapper89_copies3834 a h).1
  have c122689 := (publicBatchWrapper89_copies3834 a h).2.1
  have c122690 := (publicBatchWrapper89_copies3834 a h).2.2.1
  have c122691 := (publicBatchWrapper89_copies3834 a h).2.2.2.1
  have c122692 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.1
  have c122693 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_12 := arithEq_of_rows h (row := 2011) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_12
  simp only [← c122688, ← c122689, ← c122690] at e_2011_12
  have e_2011_13 := arithEq_of_rows h (row := 2011) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_13
  simp only [← c122691, ← c122692, k2, ← c122693] at e_2011_13
  have hr := e_2011_13
  simp only [e_2011_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15370)) := by
  have c122694 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.1
  have c122695 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.1
  have c122696 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.1
  have c122697 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.1
  have c122698 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122699 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_14 := arithEq_of_rows h (row := 2011) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_14
  simp only [← c122694, ← c122695, ← c122696] at e_2011_14
  have e_2011_15 := arithEq_of_rows h (row := 2011) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_15
  simp only [← c122697, ← c122698, k2, ← c122699] at e_2011_15
  have hr := e_2011_15
  simp only [e_2011_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15371)) := by
  have c122700 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122701 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122702 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122703 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122704 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122705 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_16 := arithEq_of_rows h (row := 2011) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_16
  simp only [← c122700, ← c122701, ← c122702] at e_2011_16
  have e_2011_17 := arithEq_of_rows h (row := 2011) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_17
  simp only [← c122703, ← c122704, k2, ← c122705] at e_2011_17
  have hr := e_2011_17
  simp only [e_2011_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15372)) := by
  have c122706 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122707 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122708 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122709 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122710 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122711 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_18 := arithEq_of_rows h (row := 2011) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_18
  simp only [← c122706, ← c122707, ← c122708] at e_2011_18
  have e_2011_19 := arithEq_of_rows h (row := 2011) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_19
  simp only [← c122709, ← c122710, k2, ← c122711] at e_2011_19
  have hr := e_2011_19
  simp only [e_2011_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15373)) := by
  have c122712 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122713 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122714 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122715 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122716 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122717 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_0 := arithEq_of_rows h (row := 2012) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_0
  simp only [← c122712, ← c122713, ← c122714] at e_2012_0
  have e_2012_1 := arithEq_of_rows h (row := 2012) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_1
  simp only [← c122715, ← c122716, k2, ← c122717] at e_2012_1
  have hr := e_2012_1
  simp only [e_2012_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15374)) := by
  have c122718 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122719 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122720 := (publicBatchWrapper89_copies3835 a h).1
  have c122721 := (publicBatchWrapper89_copies3835 a h).2.1
  have c122722 := (publicBatchWrapper89_copies3835 a h).2.2.1
  have c122723 := (publicBatchWrapper89_copies3835 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_2 := arithEq_of_rows h (row := 2012) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_2
  simp only [← c122718, ← c122719, ← c122720] at e_2012_2
  have e_2012_3 := arithEq_of_rows h (row := 2012) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_3
  simp only [← c122721, ← c122722, k2, ← c122723] at e_2012_3
  have hr := e_2012_3
  simp only [e_2012_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15375)) := by
  have c122724 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.1
  have c122725 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.1
  have c122726 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.1
  have c122727 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.1
  have c122728 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.1
  have c122729 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_4 := arithEq_of_rows h (row := 2012) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_4
  simp only [← c122724, ← c122725, ← c122726] at e_2012_4
  have e_2012_5 := arithEq_of_rows h (row := 2012) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_5
  simp only [← c122727, ← c122728, k2, ← c122729] at e_2012_5
  have hr := e_2012_5
  simp only [e_2012_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15376)) := by
  have c122730 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122731 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122732 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122733 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122734 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122735 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_6 := arithEq_of_rows h (row := 2012) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_6
  simp only [← c122730, ← c122731, ← c122732] at e_2012_6
  have e_2012_7 := arithEq_of_rows h (row := 2012) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_7
  simp only [← c122733, ← c122734, k2, ← c122735] at e_2012_7
  have hr := e_2012_7
  simp only [e_2012_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15377)) := by
  have c122736 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122737 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122738 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122739 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122740 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122741 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_8 := arithEq_of_rows h (row := 2012) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_8
  simp only [← c122736, ← c122737, ← c122738] at e_2012_8
  have e_2012_9 := arithEq_of_rows h (row := 2012) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_9
  simp only [← c122739, ← c122740, k2, ← c122741] at e_2012_9
  have hr := e_2012_9
  simp only [e_2012_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15378)) := by
  have c122742 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122743 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122744 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122745 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122746 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122747 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_10 := arithEq_of_rows h (row := 2012) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_10
  simp only [← c122742, ← c122743, ← c122744] at e_2012_10
  have e_2012_11 := arithEq_of_rows h (row := 2012) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_11
  simp only [← c122745, ← c122746, k2, ← c122747] at e_2012_11
  have hr := e_2012_11
  simp only [e_2012_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15379)) := by
  have c122748 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122749 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122750 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122751 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122752 := (publicBatchWrapper89_copies3836 a h).1
  have c122753 := (publicBatchWrapper89_copies3836 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_12 := arithEq_of_rows h (row := 2012) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_12
  simp only [← c122748, ← c122749, ← c122750] at e_2012_12
  have e_2012_13 := arithEq_of_rows h (row := 2012) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_13
  simp only [← c122751, ← c122752, k2, ← c122753] at e_2012_13
  have hr := e_2012_13
  simp only [e_2012_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15380)) := by
  have c122754 := (publicBatchWrapper89_copies3836 a h).2.2.1
  have c122755 := (publicBatchWrapper89_copies3836 a h).2.2.2.1
  have c122756 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.1
  have c122757 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.1
  have c122758 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.1
  have c122759 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_14 := arithEq_of_rows h (row := 2012) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_14
  simp only [← c122754, ← c122755, ← c122756] at e_2012_14
  have e_2012_15 := arithEq_of_rows h (row := 2012) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_15
  simp only [← c122757, ← c122758, k2, ← c122759] at e_2012_15
  have hr := e_2012_15
  simp only [e_2012_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15381)) := by
  have c122760 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.1
  have c122761 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.1
  have c122762 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122763 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122764 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122765 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_16 := arithEq_of_rows h (row := 2012) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_16
  simp only [← c122760, ← c122761, ← c122762] at e_2012_16
  have e_2012_17 := arithEq_of_rows h (row := 2012) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_17
  simp only [← c122763, ← c122764, k2, ← c122765] at e_2012_17
  have hr := e_2012_17
  simp only [e_2012_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15382)) := by
  have c122766 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122767 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122768 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122769 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122770 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122771 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_18 := arithEq_of_rows h (row := 2012) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_18
  simp only [← c122766, ← c122767, ← c122768] at e_2012_18
  have e_2012_19 := arithEq_of_rows h (row := 2012) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_19
  simp only [← c122769, ← c122770, k2, ← c122771] at e_2012_19
  have hr := e_2012_19
  simp only [e_2012_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15383)) := by
  have c122772 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122773 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122774 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122775 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122776 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122777 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_0 := arithEq_of_rows h (row := 2013) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_0
  simp only [← c122772, ← c122773, ← c122774] at e_2013_0
  have e_2013_1 := arithEq_of_rows h (row := 2013) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_1
  simp only [← c122775, ← c122776, k2, ← c122777] at e_2013_1
  have hr := e_2013_1
  simp only [e_2013_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15384)) := by
  have c122778 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122779 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122780 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122781 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122782 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122783 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_2 := arithEq_of_rows h (row := 2013) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_2
  simp only [← c122778, ← c122779, ← c122780] at e_2013_2
  have e_2013_3 := arithEq_of_rows h (row := 2013) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_3
  simp only [← c122781, ← c122782, k2, ← c122783] at e_2013_3
  have hr := e_2013_3
  simp only [e_2013_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15385)) := by
  have c122784 := (publicBatchWrapper89_copies3837 a h).1
  have c122785 := (publicBatchWrapper89_copies3837 a h).2.1
  have c122786 := (publicBatchWrapper89_copies3837 a h).2.2.1
  have c122787 := (publicBatchWrapper89_copies3837 a h).2.2.2.1
  have c122788 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.1
  have c122789 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_4 := arithEq_of_rows h (row := 2013) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_4
  simp only [← c122784, ← c122785, ← c122786] at e_2013_4
  have e_2013_5 := arithEq_of_rows h (row := 2013) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_5
  simp only [← c122787, ← c122788, k2, ← c122789] at e_2013_5
  have hr := e_2013_5
  simp only [e_2013_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15386)) := by
  have c122790 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.1
  have c122791 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.1
  have c122792 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.1
  have c122793 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.1
  have c122794 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122795 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_6 := arithEq_of_rows h (row := 2013) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_6
  simp only [← c122790, ← c122791, ← c122792] at e_2013_6
  have e_2013_7 := arithEq_of_rows h (row := 2013) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_7
  simp only [← c122793, ← c122794, k2, ← c122795] at e_2013_7
  have hr := e_2013_7
  simp only [e_2013_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15387)) := by
  have c122796 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122797 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122798 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122799 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122800 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122801 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_8 := arithEq_of_rows h (row := 2013) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_8
  simp only [← c122796, ← c122797, ← c122798] at e_2013_8
  have e_2013_9 := arithEq_of_rows h (row := 2013) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_9
  simp only [← c122799, ← c122800, k2, ← c122801] at e_2013_9
  have hr := e_2013_9
  simp only [e_2013_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15388)) := by
  have c122802 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122803 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122804 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122805 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122806 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122807 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_10 := arithEq_of_rows h (row := 2013) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_10
  simp only [← c122802, ← c122803, ← c122804] at e_2013_10
  have e_2013_11 := arithEq_of_rows h (row := 2013) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_11
  simp only [← c122805, ← c122806, k2, ← c122807] at e_2013_11
  have hr := e_2013_11
  simp only [e_2013_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15389)) := by
  have c122808 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122809 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122810 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122811 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122812 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122813 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_12 := arithEq_of_rows h (row := 2013) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_12
  simp only [← c122808, ← c122809, ← c122810] at e_2013_12
  have e_2013_13 := arithEq_of_rows h (row := 2013) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_13
  simp only [← c122811, ← c122812, k2, ← c122813] at e_2013_13
  have hr := e_2013_13
  simp only [e_2013_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15390)) := by
  have c122814 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122815 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122816 := (publicBatchWrapper89_copies3838 a h).1
  have c122817 := (publicBatchWrapper89_copies3838 a h).2.1
  have c122818 := (publicBatchWrapper89_copies3838 a h).2.2.1
  have c122819 := (publicBatchWrapper89_copies3838 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_14 := arithEq_of_rows h (row := 2013) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_14
  simp only [← c122814, ← c122815, ← c122816] at e_2013_14
  have e_2013_15 := arithEq_of_rows h (row := 2013) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_15
  simp only [← c122817, ← c122818, k2, ← c122819] at e_2013_15
  have hr := e_2013_15
  simp only [e_2013_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15391)) := by
  have c122820 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.1
  have c122821 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.1
  have c122822 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.1
  have c122823 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.1
  have c122824 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.1
  have c122825 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_16 := arithEq_of_rows h (row := 2013) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_16
  simp only [← c122820, ← c122821, ← c122822] at e_2013_16
  have e_2013_17 := arithEq_of_rows h (row := 2013) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_17
  simp only [← c122823, ← c122824, k2, ← c122825] at e_2013_17
  have hr := e_2013_17
  simp only [e_2013_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15392)) := by
  have c122826 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122827 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122828 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122829 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122830 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122831 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_18 := arithEq_of_rows h (row := 2013) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_18
  simp only [← c122826, ← c122827, ← c122828] at e_2013_18
  have e_2013_19 := arithEq_of_rows h (row := 2013) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_19
  simp only [← c122829, ← c122830, k2, ← c122831] at e_2013_19
  have hr := e_2013_19
  simp only [e_2013_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15393)) := by
  have c122832 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122833 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122834 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122835 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122836 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122837 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_0 := arithEq_of_rows h (row := 2014) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_0
  simp only [← c122832, ← c122833, ← c122834] at e_2014_0
  have e_2014_1 := arithEq_of_rows h (row := 2014) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_1
  simp only [← c122835, ← c122836, k2, ← c122837] at e_2014_1
  have hr := e_2014_1
  simp only [e_2014_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15394)) := by
  have c122838 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122839 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122840 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122841 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122842 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122843 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_2 := arithEq_of_rows h (row := 2014) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_2
  simp only [← c122838, ← c122839, ← c122840] at e_2014_2
  have e_2014_3 := arithEq_of_rows h (row := 2014) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_3
  simp only [← c122841, ← c122842, k2, ← c122843] at e_2014_3
  have hr := e_2014_3
  simp only [e_2014_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15395)) := by
  have c122844 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122845 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122846 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122847 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122848 := (publicBatchWrapper89_copies3839 a h).1
  have c122849 := (publicBatchWrapper89_copies3839 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_4 := arithEq_of_rows h (row := 2014) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_4
  simp only [← c122844, ← c122845, ← c122846] at e_2014_4
  have e_2014_5 := arithEq_of_rows h (row := 2014) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_5
  simp only [← c122847, ← c122848, k2, ← c122849] at e_2014_5
  have hr := e_2014_5
  simp only [e_2014_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15396)) := by
  have c122850 := (publicBatchWrapper89_copies3839 a h).2.2.1
  have c122851 := (publicBatchWrapper89_copies3839 a h).2.2.2.1
  have c122852 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.1
  have c122853 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.1
  have c122854 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.1
  have c122855 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_6 := arithEq_of_rows h (row := 2014) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_6
  simp only [← c122850, ← c122851, ← c122852] at e_2014_6
  have e_2014_7 := arithEq_of_rows h (row := 2014) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_7
  simp only [← c122853, ← c122854, k2, ← c122855] at e_2014_7
  have hr := e_2014_7
  simp only [e_2014_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15397)) := by
  have c122856 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.1
  have c122857 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.1
  have c122858 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122859 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122860 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122861 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_8 := arithEq_of_rows h (row := 2014) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_8
  simp only [← c122856, ← c122857, ← c122858] at e_2014_8
  have e_2014_9 := arithEq_of_rows h (row := 2014) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_9
  simp only [← c122859, ← c122860, k2, ← c122861] at e_2014_9
  have hr := e_2014_9
  simp only [e_2014_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15398)) := by
  have c122862 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122863 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122864 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122865 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122866 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122867 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_10 := arithEq_of_rows h (row := 2014) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_10
  simp only [← c122862, ← c122863, ← c122864] at e_2014_10
  have e_2014_11 := arithEq_of_rows h (row := 2014) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_11
  simp only [← c122865, ← c122866, k2, ← c122867] at e_2014_11
  have hr := e_2014_11
  simp only [e_2014_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15399)) := by
  have c122868 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122869 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122870 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122871 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122872 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122873 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_12 := arithEq_of_rows h (row := 2014) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_12
  simp only [← c122868, ← c122869, ← c122870] at e_2014_12
  have e_2014_13 := arithEq_of_rows h (row := 2014) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_13
  simp only [← c122871, ← c122872, k2, ← c122873] at e_2014_13
  have hr := e_2014_13
  simp only [e_2014_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15400)) := by
  have c122874 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122875 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122876 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122877 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122878 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122879 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_14 := arithEq_of_rows h (row := 2014) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_14
  simp only [← c122874, ← c122875, ← c122876] at e_2014_14
  have e_2014_15 := arithEq_of_rows h (row := 2014) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_15
  simp only [← c122877, ← c122878, k2, ← c122879] at e_2014_15
  have hr := e_2014_15
  simp only [e_2014_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15401)) := by
  have c122880 := (publicBatchWrapper89_copies3840 a h).1
  have c122881 := (publicBatchWrapper89_copies3840 a h).2.1
  have c122882 := (publicBatchWrapper89_copies3840 a h).2.2.1
  have c122883 := (publicBatchWrapper89_copies3840 a h).2.2.2.1
  have c122884 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.1
  have c122885 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_16 := arithEq_of_rows h (row := 2014) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_16
  simp only [← c122880, ← c122881, ← c122882] at e_2014_16
  have e_2014_17 := arithEq_of_rows h (row := 2014) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_17
  simp only [← c122883, ← c122884, k2, ← c122885] at e_2014_17
  have hr := e_2014_17
  simp only [e_2014_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15570)) := by
  have c122886 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.1
  have c122887 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.1
  have c122888 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.1
  have c122889 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.1
  have c122890 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122891 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_18 := arithEq_of_rows h (row := 2014) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_18
  simp only [← c122886, ← c122887, ← c122888] at e_2014_18
  have e_2014_19 := arithEq_of_rows h (row := 2014) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_19
  simp only [← c122889, ← c122890, k2, ← c122891] at e_2014_19
  have hr := e_2014_19
  simp only [e_2014_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15571)) := by
  have c122892 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122893 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122894 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122895 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122896 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122897 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_0 := arithEq_of_rows h (row := 2015) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_0
  simp only [← c122892, ← c122893, ← c122894] at e_2015_0
  have e_2015_1 := arithEq_of_rows h (row := 2015) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_1
  simp only [← c122895, ← c122896, k2, ← c122897] at e_2015_1
  have hr := e_2015_1
  simp only [e_2015_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15572)) := by
  have c122898 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122899 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122900 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122901 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122902 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122903 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_2 := arithEq_of_rows h (row := 2015) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_2
  simp only [← c122898, ← c122899, ← c122900] at e_2015_2
  have e_2015_3 := arithEq_of_rows h (row := 2015) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_3
  simp only [← c122901, ← c122902, k2, ← c122903] at e_2015_3
  have hr := e_2015_3
  simp only [e_2015_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15573)) := by
  have c122904 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122905 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122906 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122907 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122908 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122909 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_4 := arithEq_of_rows h (row := 2015) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_4
  simp only [← c122904, ← c122905, ← c122906] at e_2015_4
  have e_2015_5 := arithEq_of_rows h (row := 2015) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_5
  simp only [← c122907, ← c122908, k2, ← c122909] at e_2015_5
  have hr := e_2015_5
  simp only [e_2015_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15574)) := by
  have c122910 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122911 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122912 := (publicBatchWrapper89_copies3841 a h).1
  have c122913 := (publicBatchWrapper89_copies3841 a h).2.1
  have c122914 := (publicBatchWrapper89_copies3841 a h).2.2.1
  have c122915 := (publicBatchWrapper89_copies3841 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_6 := arithEq_of_rows h (row := 2015) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_6
  simp only [← c122910, ← c122911, ← c122912] at e_2015_6
  have e_2015_7 := arithEq_of_rows h (row := 2015) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_7
  simp only [← c122913, ← c122914, k2, ← c122915] at e_2015_7
  have hr := e_2015_7
  simp only [e_2015_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15575)) := by
  have c122916 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.1
  have c122917 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.1
  have c122918 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.1
  have c122919 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.1
  have c122920 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.1
  have c122921 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_8 := arithEq_of_rows h (row := 2015) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_8
  simp only [← c122916, ← c122917, ← c122918] at e_2015_8
  have e_2015_9 := arithEq_of_rows h (row := 2015) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_9
  simp only [← c122919, ← c122920, k2, ← c122921] at e_2015_9
  have hr := e_2015_9
  simp only [e_2015_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15576)) := by
  have c122922 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122923 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122924 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122925 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122926 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122927 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_10 := arithEq_of_rows h (row := 2015) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_10
  simp only [← c122922, ← c122923, ← c122924] at e_2015_10
  have e_2015_11 := arithEq_of_rows h (row := 2015) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_11
  simp only [← c122925, ← c122926, k2, ← c122927] at e_2015_11
  have hr := e_2015_11
  simp only [e_2015_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15577)) := by
  have c122928 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122929 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122930 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122931 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122932 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122933 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_12 := arithEq_of_rows h (row := 2015) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_12
  simp only [← c122928, ← c122929, ← c122930] at e_2015_12
  have e_2015_13 := arithEq_of_rows h (row := 2015) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_13
  simp only [← c122931, ← c122932, k2, ← c122933] at e_2015_13
  have hr := e_2015_13
  simp only [e_2015_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15578)) := by
  have c122934 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122935 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122936 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122937 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122938 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122939 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_14 := arithEq_of_rows h (row := 2015) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_14
  simp only [← c122934, ← c122935, ← c122936] at e_2015_14
  have e_2015_15 := arithEq_of_rows h (row := 2015) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_15
  simp only [← c122937, ← c122938, k2, ← c122939] at e_2015_15
  have hr := e_2015_15
  simp only [e_2015_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15579)) := by
  have c122940 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122941 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122942 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122943 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122944 := (publicBatchWrapper89_copies3842 a h).1
  have c122945 := (publicBatchWrapper89_copies3842 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_16 := arithEq_of_rows h (row := 2015) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_16
  simp only [← c122940, ← c122941, ← c122942] at e_2015_16
  have e_2015_17 := arithEq_of_rows h (row := 2015) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_17
  simp only [← c122943, ← c122944, k2, ← c122945] at e_2015_17
  have hr := e_2015_17
  simp only [e_2015_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15580)) := by
  have c122946 := (publicBatchWrapper89_copies3842 a h).2.2.1
  have c122947 := (publicBatchWrapper89_copies3842 a h).2.2.2.1
  have c122948 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.1
  have c122949 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.1
  have c122950 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.1
  have c122951 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_18 := arithEq_of_rows h (row := 2015) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_18
  simp only [← c122946, ← c122947, ← c122948] at e_2015_18
  have e_2015_19 := arithEq_of_rows h (row := 2015) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_19
  simp only [← c122949, ← c122950, k2, ← c122951] at e_2015_19
  have hr := e_2015_19
  simp only [e_2015_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15581)) := by
  have c122952 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.1
  have c122953 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.1
  have c122954 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122955 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122956 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122957 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_0 := arithEq_of_rows h (row := 2016) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_0
  simp only [← c122952, ← c122953, ← c122954] at e_2016_0
  have e_2016_1 := arithEq_of_rows h (row := 2016) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_1
  simp only [← c122955, ← c122956, k2, ← c122957] at e_2016_1
  have hr := e_2016_1
  simp only [e_2016_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15582)) := by
  have c122958 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122959 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122960 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122961 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122962 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122963 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_2 := arithEq_of_rows h (row := 2016) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_2
  simp only [← c122958, ← c122959, ← c122960] at e_2016_2
  have e_2016_3 := arithEq_of_rows h (row := 2016) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_3
  simp only [← c122961, ← c122962, k2, ← c122963] at e_2016_3
  have hr := e_2016_3
  simp only [e_2016_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15583)) := by
  have c122964 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122965 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122966 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122967 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122968 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122969 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_4 := arithEq_of_rows h (row := 2016) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_4
  simp only [← c122964, ← c122965, ← c122966] at e_2016_4
  have e_2016_5 := arithEq_of_rows h (row := 2016) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_5
  simp only [← c122967, ← c122968, k2, ← c122969] at e_2016_5
  have hr := e_2016_5
  simp only [e_2016_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15584)) := by
  have c122970 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122971 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122972 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122973 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122974 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122975 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_6 := arithEq_of_rows h (row := 2016) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_6
  simp only [← c122970, ← c122971, ← c122972] at e_2016_6
  have e_2016_7 := arithEq_of_rows h (row := 2016) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_7
  simp only [← c122973, ← c122974, k2, ← c122975] at e_2016_7
  have hr := e_2016_7
  simp only [e_2016_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15585)) := by
  have c122976 := (publicBatchWrapper89_copies3843 a h).1
  have c122977 := (publicBatchWrapper89_copies3843 a h).2.1
  have c122978 := (publicBatchWrapper89_copies3843 a h).2.2.1
  have c122979 := (publicBatchWrapper89_copies3843 a h).2.2.2.1
  have c122980 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.1
  have c122981 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_8 := arithEq_of_rows h (row := 2016) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_8
  simp only [← c122976, ← c122977, ← c122978] at e_2016_8
  have e_2016_9 := arithEq_of_rows h (row := 2016) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_9
  simp only [← c122979, ← c122980, k2, ← c122981] at e_2016_9
  have hr := e_2016_9
  simp only [e_2016_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15586)) := by
  have c122982 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.1
  have c122983 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.1
  have c122984 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.1
  have c122985 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.1
  have c122986 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122987 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_10 := arithEq_of_rows h (row := 2016) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_10
  simp only [← c122982, ← c122983, ← c122984] at e_2016_10
  have e_2016_11 := arithEq_of_rows h (row := 2016) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_11
  simp only [← c122985, ← c122986, k2, ← c122987] at e_2016_11
  have hr := e_2016_11
  simp only [e_2016_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15587)) := by
  have c122988 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122989 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122990 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122991 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122992 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122993 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_12 := arithEq_of_rows h (row := 2016) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_12
  simp only [← c122988, ← c122989, ← c122990] at e_2016_12
  have e_2016_13 := arithEq_of_rows h (row := 2016) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_13
  simp only [← c122991, ← c122992, k2, ← c122993] at e_2016_13
  have hr := e_2016_13
  simp only [e_2016_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15588)) := by
  have c122994 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122995 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122996 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122997 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122998 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122999 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_14 := arithEq_of_rows h (row := 2016) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_14
  simp only [← c122994, ← c122995, ← c122996] at e_2016_14
  have e_2016_15 := arithEq_of_rows h (row := 2016) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_15
  simp only [← c122997, ← c122998, k2, ← c122999] at e_2016_15
  have hr := e_2016_15
  simp only [e_2016_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15589)) := by
  have c123000 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123001 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123002 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123003 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123004 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123005 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_16 := arithEq_of_rows h (row := 2016) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_16
  simp only [← c123000, ← c123001, ← c123002] at e_2016_16
  have e_2016_17 := arithEq_of_rows h (row := 2016) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_17
  simp only [← c123003, ← c123004, k2, ← c123005] at e_2016_17
  have hr := e_2016_17
  simp only [e_2016_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15590)) := by
  have c123006 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123007 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123008 := (publicBatchWrapper89_copies3844 a h).1
  have c123009 := (publicBatchWrapper89_copies3844 a h).2.1
  have c123010 := (publicBatchWrapper89_copies3844 a h).2.2.1
  have c123011 := (publicBatchWrapper89_copies3844 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_18 := arithEq_of_rows h (row := 2016) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_18
  simp only [← c123006, ← c123007, ← c123008] at e_2016_18
  have e_2016_19 := arithEq_of_rows h (row := 2016) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_19
  simp only [← c123009, ← c123010, k2, ← c123011] at e_2016_19
  have hr := e_2016_19
  simp only [e_2016_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15591)) := by
  have c123012 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.1
  have c123013 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.1
  have c123014 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.1
  have c123015 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.1
  have c123016 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.1
  have c123017 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_0 := arithEq_of_rows h (row := 2017) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_0
  simp only [← c123012, ← c123013, ← c123014] at e_2017_0
  have e_2017_1 := arithEq_of_rows h (row := 2017) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_1
  simp only [← c123015, ← c123016, k2, ← c123017] at e_2017_1
  have hr := e_2017_1
  simp only [e_2017_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15592)) := by
  have c123018 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123019 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123020 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123021 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123022 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123023 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_2 := arithEq_of_rows h (row := 2017) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_2
  simp only [← c123018, ← c123019, ← c123020] at e_2017_2
  have e_2017_3 := arithEq_of_rows h (row := 2017) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_3
  simp only [← c123021, ← c123022, k2, ← c123023] at e_2017_3
  have hr := e_2017_3
  simp only [e_2017_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15593)) := by
  have c123024 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123025 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123026 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123027 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123028 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123029 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_4 := arithEq_of_rows h (row := 2017) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_4
  simp only [← c123024, ← c123025, ← c123026] at e_2017_4
  have e_2017_5 := arithEq_of_rows h (row := 2017) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_5
  simp only [← c123027, ← c123028, k2, ← c123029] at e_2017_5
  have hr := e_2017_5
  simp only [e_2017_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15594)) := by
  have c123030 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123031 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123032 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123033 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123034 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123035 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_6 := arithEq_of_rows h (row := 2017) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_6
  simp only [← c123030, ← c123031, ← c123032] at e_2017_6
  have e_2017_7 := arithEq_of_rows h (row := 2017) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_7
  simp only [← c123033, ← c123034, k2, ← c123035] at e_2017_7
  have hr := e_2017_7
  simp only [e_2017_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15595)) := by
  have c123036 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123037 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123038 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123039 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123040 := (publicBatchWrapper89_copies3845 a h).1
  have c123041 := (publicBatchWrapper89_copies3845 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_8 := arithEq_of_rows h (row := 2017) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_8
  simp only [← c123036, ← c123037, ← c123038] at e_2017_8
  have e_2017_9 := arithEq_of_rows h (row := 2017) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_9
  simp only [← c123039, ← c123040, k2, ← c123041] at e_2017_9
  have hr := e_2017_9
  simp only [e_2017_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15596)) := by
  have c123042 := (publicBatchWrapper89_copies3845 a h).2.2.1
  have c123043 := (publicBatchWrapper89_copies3845 a h).2.2.2.1
  have c123044 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.1
  have c123045 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.1
  have c123046 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.1
  have c123047 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_10 := arithEq_of_rows h (row := 2017) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_10
  simp only [← c123042, ← c123043, ← c123044] at e_2017_10
  have e_2017_11 := arithEq_of_rows h (row := 2017) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_11
  simp only [← c123045, ← c123046, k2, ← c123047] at e_2017_11
  have hr := e_2017_11
  simp only [e_2017_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15597)) := by
  have c123048 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.1
  have c123049 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.1
  have c123050 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123051 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123052 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123053 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_12 := arithEq_of_rows h (row := 2017) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_12
  simp only [← c123048, ← c123049, ← c123050] at e_2017_12
  have e_2017_13 := arithEq_of_rows h (row := 2017) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_13
  simp only [← c123051, ← c123052, k2, ← c123053] at e_2017_13
  have hr := e_2017_13
  simp only [e_2017_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15598)) := by
  have c123054 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123055 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123056 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123057 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123058 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123059 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_14 := arithEq_of_rows h (row := 2017) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_14
  simp only [← c123054, ← c123055, ← c123056] at e_2017_14
  have e_2017_15 := arithEq_of_rows h (row := 2017) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_15
  simp only [← c123057, ← c123058, k2, ← c123059] at e_2017_15
  have hr := e_2017_15
  simp only [e_2017_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15599)) := by
  have c123060 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123061 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123062 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123063 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123064 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123065 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_16 := arithEq_of_rows h (row := 2017) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_16
  simp only [← c123060, ← c123061, ← c123062] at e_2017_16
  have e_2017_17 := arithEq_of_rows h (row := 2017) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_17
  simp only [← c123063, ← c123064, k2, ← c123065] at e_2017_17
  have hr := e_2017_17
  simp only [e_2017_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15600)) := by
  have c123066 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123067 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123068 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123069 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123070 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123071 := (publicBatchWrapper89_copies3845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_18 := arithEq_of_rows h (row := 2017) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_18
  simp only [← c123066, ← c123067, ← c123068] at e_2017_18
  have e_2017_19 := arithEq_of_rows h (row := 2017) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_19
  simp only [← c123069, ← c123070, k2, ← c123071] at e_2017_19
  have hr := e_2017_19
  simp only [e_2017_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15601)) := by
  have c123072 := (publicBatchWrapper89_copies3846 a h).1
  have c123073 := (publicBatchWrapper89_copies3846 a h).2.1
  have c123074 := (publicBatchWrapper89_copies3846 a h).2.2.1
  have c123075 := (publicBatchWrapper89_copies3846 a h).2.2.2.1
  have c123076 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.1
  have c123077 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_0 := arithEq_of_rows h (row := 2018) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_0
  simp only [← c123072, ← c123073, ← c123074] at e_2018_0
  have e_2018_1 := arithEq_of_rows h (row := 2018) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_1
  simp only [← c123075, ← c123076, k2, ← c123077] at e_2018_1
  have hr := e_2018_1
  simp only [e_2018_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15770)) := by
  have c123078 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.1
  have c123079 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.1
  have c123080 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.1
  have c123081 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.1
  have c123082 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123083 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_2 := arithEq_of_rows h (row := 2018) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_2
  simp only [← c123078, ← c123079, ← c123080] at e_2018_2
  have e_2018_3 := arithEq_of_rows h (row := 2018) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_3
  simp only [← c123081, ← c123082, k2, ← c123083] at e_2018_3
  have hr := e_2018_3
  simp only [e_2018_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15771)) := by
  have c123084 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123085 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123086 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123087 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123088 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123089 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_4 := arithEq_of_rows h (row := 2018) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_4
  simp only [← c123084, ← c123085, ← c123086] at e_2018_4
  have e_2018_5 := arithEq_of_rows h (row := 2018) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_5
  simp only [← c123087, ← c123088, k2, ← c123089] at e_2018_5
  have hr := e_2018_5
  simp only [e_2018_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15772)) := by
  have c123090 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123091 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123092 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123093 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123094 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123095 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_6 := arithEq_of_rows h (row := 2018) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_6
  simp only [← c123090, ← c123091, ← c123092] at e_2018_6
  have e_2018_7 := arithEq_of_rows h (row := 2018) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_7
  simp only [← c123093, ← c123094, k2, ← c123095] at e_2018_7
  have hr := e_2018_7
  simp only [e_2018_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15773)) := by
  have c123096 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123097 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123098 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123099 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123100 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123101 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_8 := arithEq_of_rows h (row := 2018) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_8
  simp only [← c123096, ← c123097, ← c123098] at e_2018_8
  have e_2018_9 := arithEq_of_rows h (row := 2018) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_9
  simp only [← c123099, ← c123100, k2, ← c123101] at e_2018_9
  have hr := e_2018_9
  simp only [e_2018_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15774)) := by
  have c123102 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123103 := (publicBatchWrapper89_copies3846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123104 := (publicBatchWrapper89_copies3847 a h).1
  have c123105 := (publicBatchWrapper89_copies3847 a h).2.1
  have c123106 := (publicBatchWrapper89_copies3847 a h).2.2.1
  have c123107 := (publicBatchWrapper89_copies3847 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_10 := arithEq_of_rows h (row := 2018) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_10
  simp only [← c123102, ← c123103, ← c123104] at e_2018_10
  have e_2018_11 := arithEq_of_rows h (row := 2018) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_11
  simp only [← c123105, ← c123106, k2, ← c123107] at e_2018_11
  have hr := e_2018_11
  simp only [e_2018_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15775)) := by
  have c123108 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.1
  have c123109 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.1
  have c123110 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.1
  have c123111 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.1
  have c123112 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.1
  have c123113 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_12 := arithEq_of_rows h (row := 2018) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_12
  simp only [← c123108, ← c123109, ← c123110] at e_2018_12
  have e_2018_13 := arithEq_of_rows h (row := 2018) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_13
  simp only [← c123111, ← c123112, k2, ← c123113] at e_2018_13
  have hr := e_2018_13
  simp only [e_2018_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15776)) := by
  have c123114 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123115 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123116 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123117 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123118 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123119 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_14 := arithEq_of_rows h (row := 2018) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_14
  simp only [← c123114, ← c123115, ← c123116] at e_2018_14
  have e_2018_15 := arithEq_of_rows h (row := 2018) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_15
  simp only [← c123117, ← c123118, k2, ← c123119] at e_2018_15
  have hr := e_2018_15
  simp only [e_2018_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15777)) := by
  have c123120 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123121 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123122 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123123 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123124 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123125 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_16 := arithEq_of_rows h (row := 2018) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_16
  simp only [← c123120, ← c123121, ← c123122] at e_2018_16
  have e_2018_17 := arithEq_of_rows h (row := 2018) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_17
  simp only [← c123123, ← c123124, k2, ← c123125] at e_2018_17
  have hr := e_2018_17
  simp only [e_2018_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2018 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15778)) := by
  have c123126 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123127 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123128 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123129 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123130 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123131 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2018_18 := arithEq_of_rows h (row := 2018) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_18
  simp only [← c123126, ← c123127, ← c123128] at e_2018_18
  have e_2018_19 := arithEq_of_rows h (row := 2018) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2018_19
  simp only [← c123129, ← c123130, k2, ← c123131] at e_2018_19
  have hr := e_2018_19
  simp only [e_2018_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15779)) := by
  have c123132 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123133 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123134 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123135 := (publicBatchWrapper89_copies3847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123136 := (publicBatchWrapper89_copies3848 a h).1
  have c123137 := (publicBatchWrapper89_copies3848 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_0 := arithEq_of_rows h (row := 2019) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_0
  simp only [← c123132, ← c123133, ← c123134] at e_2019_0
  have e_2019_1 := arithEq_of_rows h (row := 2019) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_1
  simp only [← c123135, ← c123136, k2, ← c123137] at e_2019_1
  have hr := e_2019_1
  simp only [e_2019_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15780)) := by
  have c123138 := (publicBatchWrapper89_copies3848 a h).2.2.1
  have c123139 := (publicBatchWrapper89_copies3848 a h).2.2.2.1
  have c123140 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.1
  have c123141 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.1
  have c123142 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.1
  have c123143 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_2 := arithEq_of_rows h (row := 2019) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_2
  simp only [← c123138, ← c123139, ← c123140] at e_2019_2
  have e_2019_3 := arithEq_of_rows h (row := 2019) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_3
  simp only [← c123141, ← c123142, k2, ← c123143] at e_2019_3
  have hr := e_2019_3
  simp only [e_2019_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15781)) := by
  have c123144 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.1
  have c123145 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.1
  have c123146 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123147 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123148 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123149 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_4 := arithEq_of_rows h (row := 2019) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_4
  simp only [← c123144, ← c123145, ← c123146] at e_2019_4
  have e_2019_5 := arithEq_of_rows h (row := 2019) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_5
  simp only [← c123147, ← c123148, k2, ← c123149] at e_2019_5
  have hr := e_2019_5
  simp only [e_2019_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15782)) := by
  have c123150 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123151 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123152 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123153 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123154 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123155 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_6 := arithEq_of_rows h (row := 2019) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_6
  simp only [← c123150, ← c123151, ← c123152] at e_2019_6
  have e_2019_7 := arithEq_of_rows h (row := 2019) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_7
  simp only [← c123153, ← c123154, k2, ← c123155] at e_2019_7
  have hr := e_2019_7
  simp only [e_2019_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15783)) := by
  have c123156 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123157 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123158 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123159 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123160 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123161 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_8 := arithEq_of_rows h (row := 2019) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_8
  simp only [← c123156, ← c123157, ← c123158] at e_2019_8
  have e_2019_9 := arithEq_of_rows h (row := 2019) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_9
  simp only [← c123159, ← c123160, k2, ← c123161] at e_2019_9
  have hr := e_2019_9
  simp only [e_2019_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15784)) := by
  have c123162 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123163 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123164 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123165 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123166 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123167 := (publicBatchWrapper89_copies3848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_10 := arithEq_of_rows h (row := 2019) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_10
  simp only [← c123162, ← c123163, ← c123164] at e_2019_10
  have e_2019_11 := arithEq_of_rows h (row := 2019) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_11
  simp only [← c123165, ← c123166, k2, ← c123167] at e_2019_11
  have hr := e_2019_11
  simp only [e_2019_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15785)) := by
  have c123168 := (publicBatchWrapper89_copies3849 a h).1
  have c123169 := (publicBatchWrapper89_copies3849 a h).2.1
  have c123170 := (publicBatchWrapper89_copies3849 a h).2.2.1
  have c123171 := (publicBatchWrapper89_copies3849 a h).2.2.2.1
  have c123172 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.1
  have c123173 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_12 := arithEq_of_rows h (row := 2019) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_12
  simp only [← c123168, ← c123169, ← c123170] at e_2019_12
  have e_2019_13 := arithEq_of_rows h (row := 2019) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_13
  simp only [← c123171, ← c123172, k2, ← c123173] at e_2019_13
  have hr := e_2019_13
  simp only [e_2019_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15786)) := by
  have c123174 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.1
  have c123175 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.1
  have c123176 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.1
  have c123177 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.1
  have c123178 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123179 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_14 := arithEq_of_rows h (row := 2019) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_14
  simp only [← c123174, ← c123175, ← c123176] at e_2019_14
  have e_2019_15 := arithEq_of_rows h (row := 2019) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_15
  simp only [← c123177, ← c123178, k2, ← c123179] at e_2019_15
  have hr := e_2019_15
  simp only [e_2019_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15787)) := by
  have c123180 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123181 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123182 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123183 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123184 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123185 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_16 := arithEq_of_rows h (row := 2019) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_16
  simp only [← c123180, ← c123181, ← c123182] at e_2019_16
  have e_2019_17 := arithEq_of_rows h (row := 2019) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_17
  simp only [← c123183, ← c123184, k2, ← c123185] at e_2019_17
  have hr := e_2019_17
  simp only [e_2019_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2019 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15788)) := by
  have c123186 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123187 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123188 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123189 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123190 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123191 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2019_18 := arithEq_of_rows h (row := 2019) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_18
  simp only [← c123186, ← c123187, ← c123188] at e_2019_18
  have e_2019_19 := arithEq_of_rows h (row := 2019) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2019_19
  simp only [← c123189, ← c123190, k2, ← c123191] at e_2019_19
  have hr := e_2019_19
  simp only [e_2019_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15789)) := by
  have c123192 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123193 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123194 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123195 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123196 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123197 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_0 := arithEq_of_rows h (row := 2020) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_0
  simp only [← c123192, ← c123193, ← c123194] at e_2020_0
  have e_2020_1 := arithEq_of_rows h (row := 2020) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_1
  simp only [← c123195, ← c123196, k2, ← c123197] at e_2020_1
  have hr := e_2020_1
  simp only [e_2020_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15790)) := by
  have c123198 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123199 := (publicBatchWrapper89_copies3849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123200 := (publicBatchWrapper89_copies3850 a h).1
  have c123201 := (publicBatchWrapper89_copies3850 a h).2.1
  have c123202 := (publicBatchWrapper89_copies3850 a h).2.2.1
  have c123203 := (publicBatchWrapper89_copies3850 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_2 := arithEq_of_rows h (row := 2020) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_2
  simp only [← c123198, ← c123199, ← c123200] at e_2020_2
  have e_2020_3 := arithEq_of_rows h (row := 2020) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_3
  simp only [← c123201, ← c123202, k2, ← c123203] at e_2020_3
  have hr := e_2020_3
  simp only [e_2020_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15791)) := by
  have c123204 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.1
  have c123205 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.1
  have c123206 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.1
  have c123207 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.1
  have c123208 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.1
  have c123209 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_4 := arithEq_of_rows h (row := 2020) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_4
  simp only [← c123204, ← c123205, ← c123206] at e_2020_4
  have e_2020_5 := arithEq_of_rows h (row := 2020) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_5
  simp only [← c123207, ← c123208, k2, ← c123209] at e_2020_5
  have hr := e_2020_5
  simp only [e_2020_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15792)) := by
  have c123210 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123211 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123212 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123213 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123214 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123215 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_6 := arithEq_of_rows h (row := 2020) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_6
  simp only [← c123210, ← c123211, ← c123212] at e_2020_6
  have e_2020_7 := arithEq_of_rows h (row := 2020) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_7
  simp only [← c123213, ← c123214, k2, ← c123215] at e_2020_7
  have hr := e_2020_7
  simp only [e_2020_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15793)) := by
  have c123216 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123217 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123218 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123219 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123220 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123221 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_8 := arithEq_of_rows h (row := 2020) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_8
  simp only [← c123216, ← c123217, ← c123218] at e_2020_8
  have e_2020_9 := arithEq_of_rows h (row := 2020) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_9
  simp only [← c123219, ← c123220, k2, ← c123221] at e_2020_9
  have hr := e_2020_9
  simp only [e_2020_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15794)) := by
  have c123222 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123223 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123224 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123225 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123226 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123227 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_10 := arithEq_of_rows h (row := 2020) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_10
  simp only [← c123222, ← c123223, ← c123224] at e_2020_10
  have e_2020_11 := arithEq_of_rows h (row := 2020) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_11
  simp only [← c123225, ← c123226, k2, ← c123227] at e_2020_11
  have hr := e_2020_11
  simp only [e_2020_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15795)) := by
  have c123228 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123229 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123230 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123231 := (publicBatchWrapper89_copies3850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123232 := (publicBatchWrapper89_copies3851 a h).1
  have c123233 := (publicBatchWrapper89_copies3851 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_12 := arithEq_of_rows h (row := 2020) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_12
  simp only [← c123228, ← c123229, ← c123230] at e_2020_12
  have e_2020_13 := arithEq_of_rows h (row := 2020) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_13
  simp only [← c123231, ← c123232, k2, ← c123233] at e_2020_13
  have hr := e_2020_13
  simp only [e_2020_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15796)) := by
  have c123234 := (publicBatchWrapper89_copies3851 a h).2.2.1
  have c123235 := (publicBatchWrapper89_copies3851 a h).2.2.2.1
  have c123236 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.1
  have c123237 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.1
  have c123238 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.1
  have c123239 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_14 := arithEq_of_rows h (row := 2020) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_14
  simp only [← c123234, ← c123235, ← c123236] at e_2020_14
  have e_2020_15 := arithEq_of_rows h (row := 2020) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_15
  simp only [← c123237, ← c123238, k2, ← c123239] at e_2020_15
  have hr := e_2020_15
  simp only [e_2020_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15797)) := by
  have c123240 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.1
  have c123241 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.1
  have c123242 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123243 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123244 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123245 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_16 := arithEq_of_rows h (row := 2020) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_16
  simp only [← c123240, ← c123241, ← c123242] at e_2020_16
  have e_2020_17 := arithEq_of_rows h (row := 2020) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_17
  simp only [← c123243, ← c123244, k2, ← c123245] at e_2020_17
  have hr := e_2020_17
  simp only [e_2020_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2020 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15798)) := by
  have c123246 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123247 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123248 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123249 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123250 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123251 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2020_18 := arithEq_of_rows h (row := 2020) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_18
  simp only [← c123246, ← c123247, ← c123248] at e_2020_18
  have e_2020_19 := arithEq_of_rows h (row := 2020) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2020_19
  simp only [← c123249, ← c123250, k2, ← c123251] at e_2020_19
  have hr := e_2020_19
  simp only [e_2020_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15799)) := by
  have c123252 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123253 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123254 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123255 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123256 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123257 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_0 := arithEq_of_rows h (row := 2021) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_0
  simp only [← c123252, ← c123253, ← c123254] at e_2021_0
  have e_2021_1 := arithEq_of_rows h (row := 2021) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_1
  simp only [← c123255, ← c123256, k2, ← c123257] at e_2021_1
  have hr := e_2021_1
  simp only [e_2021_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15800)) := by
  have c123258 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123259 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123260 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123261 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123262 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123263 := (publicBatchWrapper89_copies3851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_2 := arithEq_of_rows h (row := 2021) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_2
  simp only [← c123258, ← c123259, ← c123260] at e_2021_2
  have e_2021_3 := arithEq_of_rows h (row := 2021) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_3
  simp only [← c123261, ← c123262, k2, ← c123263] at e_2021_3
  have hr := e_2021_3
  simp only [e_2021_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15801)) := by
  have c123264 := (publicBatchWrapper89_copies3852 a h).1
  have c123265 := (publicBatchWrapper89_copies3852 a h).2.1
  have c123266 := (publicBatchWrapper89_copies3852 a h).2.2.1
  have c123267 := (publicBatchWrapper89_copies3852 a h).2.2.2.1
  have c123268 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.1
  have c123269 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_4 := arithEq_of_rows h (row := 2021) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_4
  simp only [← c123264, ← c123265, ← c123266] at e_2021_4
  have e_2021_5 := arithEq_of_rows h (row := 2021) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_5
  simp only [← c123267, ← c123268, k2, ← c123269] at e_2021_5
  have hr := e_2021_5
  simp only [e_2021_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15970)) := by
  have c123270 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.1
  have c123271 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.1
  have c123272 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.1
  have c123273 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.1
  have c123274 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123275 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_6 := arithEq_of_rows h (row := 2021) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_6
  simp only [← c123270, ← c123271, ← c123272] at e_2021_6
  have e_2021_7 := arithEq_of_rows h (row := 2021) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_7
  simp only [← c123273, ← c123274, k2, ← c123275] at e_2021_7
  have hr := e_2021_7
  simp only [e_2021_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15971)) := by
  have c123276 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123277 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123278 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123279 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123280 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123281 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_8 := arithEq_of_rows h (row := 2021) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_8
  simp only [← c123276, ← c123277, ← c123278] at e_2021_8
  have e_2021_9 := arithEq_of_rows h (row := 2021) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_9
  simp only [← c123279, ← c123280, k2, ← c123281] at e_2021_9
  have hr := e_2021_9
  simp only [e_2021_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15972)) := by
  have c123282 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123283 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123284 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123285 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123286 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123287 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_10 := arithEq_of_rows h (row := 2021) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_10
  simp only [← c123282, ← c123283, ← c123284] at e_2021_10
  have e_2021_11 := arithEq_of_rows h (row := 2021) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_11
  simp only [← c123285, ← c123286, k2, ← c123287] at e_2021_11
  have hr := e_2021_11
  simp only [e_2021_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15973)) := by
  have c123288 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123289 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123290 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123291 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123292 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123293 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_12 := arithEq_of_rows h (row := 2021) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_12
  simp only [← c123288, ← c123289, ← c123290] at e_2021_12
  have e_2021_13 := arithEq_of_rows h (row := 2021) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_13
  simp only [← c123291, ← c123292, k2, ← c123293] at e_2021_13
  have hr := e_2021_13
  simp only [e_2021_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15974)) := by
  have c123294 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123295 := (publicBatchWrapper89_copies3852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123296 := (publicBatchWrapper89_copies3853 a h).1
  have c123297 := (publicBatchWrapper89_copies3853 a h).2.1
  have c123298 := (publicBatchWrapper89_copies3853 a h).2.2.1
  have c123299 := (publicBatchWrapper89_copies3853 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_14 := arithEq_of_rows h (row := 2021) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_14
  simp only [← c123294, ← c123295, ← c123296] at e_2021_14
  have e_2021_15 := arithEq_of_rows h (row := 2021) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_15
  simp only [← c123297, ← c123298, k2, ← c123299] at e_2021_15
  have hr := e_2021_15
  simp only [e_2021_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15975)) := by
  have c123300 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.1
  have c123301 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.1
  have c123302 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.1
  have c123303 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.1
  have c123304 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.1
  have c123305 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_16 := arithEq_of_rows h (row := 2021) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_16
  simp only [← c123300, ← c123301, ← c123302] at e_2021_16
  have e_2021_17 := arithEq_of_rows h (row := 2021) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_17
  simp only [← c123303, ← c123304, k2, ← c123305] at e_2021_17
  have hr := e_2021_17
  simp only [e_2021_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2021 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15976)) := by
  have c123306 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123307 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123308 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123309 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123310 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123311 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2021_18 := arithEq_of_rows h (row := 2021) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_18
  simp only [← c123306, ← c123307, ← c123308] at e_2021_18
  have e_2021_19 := arithEq_of_rows h (row := 2021) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2021_19
  simp only [← c123309, ← c123310, k2, ← c123311] at e_2021_19
  have hr := e_2021_19
  simp only [e_2021_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15977)) := by
  have c123312 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123313 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123314 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123315 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123316 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123317 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_0 := arithEq_of_rows h (row := 2022) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_0
  simp only [← c123312, ← c123313, ← c123314] at e_2022_0
  have e_2022_1 := arithEq_of_rows h (row := 2022) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_1
  simp only [← c123315, ← c123316, k2, ← c123317] at e_2022_1
  have hr := e_2022_1
  simp only [e_2022_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15978)) := by
  have c123318 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123319 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123320 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123321 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123322 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123323 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_2 := arithEq_of_rows h (row := 2022) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_2
  simp only [← c123318, ← c123319, ← c123320] at e_2022_2
  have e_2022_3 := arithEq_of_rows h (row := 2022) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_3
  simp only [← c123321, ← c123322, k2, ← c123323] at e_2022_3
  have hr := e_2022_3
  simp only [e_2022_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15979)) := by
  have c123324 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123325 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123326 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123327 := (publicBatchWrapper89_copies3853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123328 := (publicBatchWrapper89_copies3854 a h).1
  have c123329 := (publicBatchWrapper89_copies3854 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_4 := arithEq_of_rows h (row := 2022) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_4
  simp only [← c123324, ← c123325, ← c123326] at e_2022_4
  have e_2022_5 := arithEq_of_rows h (row := 2022) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_5
  simp only [← c123327, ← c123328, k2, ← c123329] at e_2022_5
  have hr := e_2022_5
  simp only [e_2022_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15980)) := by
  have c123330 := (publicBatchWrapper89_copies3854 a h).2.2.1
  have c123331 := (publicBatchWrapper89_copies3854 a h).2.2.2.1
  have c123332 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.1
  have c123333 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.1
  have c123334 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.1
  have c123335 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_6 := arithEq_of_rows h (row := 2022) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_6
  simp only [← c123330, ← c123331, ← c123332] at e_2022_6
  have e_2022_7 := arithEq_of_rows h (row := 2022) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_7
  simp only [← c123333, ← c123334, k2, ← c123335] at e_2022_7
  have hr := e_2022_7
  simp only [e_2022_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15981)) := by
  have c123336 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.1
  have c123337 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.1
  have c123338 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123339 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123340 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123341 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_8 := arithEq_of_rows h (row := 2022) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_8
  simp only [← c123336, ← c123337, ← c123338] at e_2022_8
  have e_2022_9 := arithEq_of_rows h (row := 2022) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_9
  simp only [← c123339, ← c123340, k2, ← c123341] at e_2022_9
  have hr := e_2022_9
  simp only [e_2022_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15982)) := by
  have c123342 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123343 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123344 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123345 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123346 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123347 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_10 := arithEq_of_rows h (row := 2022) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_10
  simp only [← c123342, ← c123343, ← c123344] at e_2022_10
  have e_2022_11 := arithEq_of_rows h (row := 2022) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_11
  simp only [← c123345, ← c123346, k2, ← c123347] at e_2022_11
  have hr := e_2022_11
  simp only [e_2022_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15983)) := by
  have c123348 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123349 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123350 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123351 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123352 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123353 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_12 := arithEq_of_rows h (row := 2022) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_12
  simp only [← c123348, ← c123349, ← c123350] at e_2022_12
  have e_2022_13 := arithEq_of_rows h (row := 2022) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_13
  simp only [← c123351, ← c123352, k2, ← c123353] at e_2022_13
  have hr := e_2022_13
  simp only [e_2022_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15984)) := by
  have c123354 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123355 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123356 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123357 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123358 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123359 := (publicBatchWrapper89_copies3854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_14 := arithEq_of_rows h (row := 2022) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_14
  simp only [← c123354, ← c123355, ← c123356] at e_2022_14
  have e_2022_15 := arithEq_of_rows h (row := 2022) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_15
  simp only [← c123357, ← c123358, k2, ← c123359] at e_2022_15
  have hr := e_2022_15
  simp only [e_2022_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15985)) := by
  have c123360 := (publicBatchWrapper89_copies3855 a h).1
  have c123361 := (publicBatchWrapper89_copies3855 a h).2.1
  have c123362 := (publicBatchWrapper89_copies3855 a h).2.2.1
  have c123363 := (publicBatchWrapper89_copies3855 a h).2.2.2.1
  have c123364 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.1
  have c123365 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_16 := arithEq_of_rows h (row := 2022) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_16
  simp only [← c123360, ← c123361, ← c123362] at e_2022_16
  have e_2022_17 := arithEq_of_rows h (row := 2022) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_17
  simp only [← c123363, ← c123364, k2, ← c123365] at e_2022_17
  have hr := e_2022_17
  simp only [e_2022_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2022 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15986)) := by
  have c123366 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.1
  have c123367 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.1
  have c123368 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.1
  have c123369 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.1
  have c123370 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123371 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2022_18 := arithEq_of_rows h (row := 2022) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_18
  simp only [← c123366, ← c123367, ← c123368] at e_2022_18
  have e_2022_19 := arithEq_of_rows h (row := 2022) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2022_19
  simp only [← c123369, ← c123370, k2, ← c123371] at e_2022_19
  have hr := e_2022_19
  simp only [e_2022_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15987)) := by
  have c123372 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123373 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123374 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123375 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123376 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123377 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_0 := arithEq_of_rows h (row := 2023) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_0
  simp only [← c123372, ← c123373, ← c123374] at e_2023_0
  have e_2023_1 := arithEq_of_rows h (row := 2023) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_1
  simp only [← c123375, ← c123376, k2, ← c123377] at e_2023_1
  have hr := e_2023_1
  simp only [e_2023_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15988)) := by
  have c123378 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123379 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123380 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123381 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123382 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123383 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_2 := arithEq_of_rows h (row := 2023) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_2
  simp only [← c123378, ← c123379, ← c123380] at e_2023_2
  have e_2023_3 := arithEq_of_rows h (row := 2023) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_3
  simp only [← c123381, ← c123382, k2, ← c123383] at e_2023_3
  have hr := e_2023_3
  simp only [e_2023_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15989)) := by
  have c123384 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123385 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123386 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123387 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123388 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123389 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_4 := arithEq_of_rows h (row := 2023) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_4
  simp only [← c123384, ← c123385, ← c123386] at e_2023_4
  have e_2023_5 := arithEq_of_rows h (row := 2023) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_5
  simp only [← c123387, ← c123388, k2, ← c123389] at e_2023_5
  have hr := e_2023_5
  simp only [e_2023_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15990)) := by
  have c123390 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123391 := (publicBatchWrapper89_copies3855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123392 := (publicBatchWrapper89_copies3856 a h).1
  have c123393 := (publicBatchWrapper89_copies3856 a h).2.1
  have c123394 := (publicBatchWrapper89_copies3856 a h).2.2.1
  have c123395 := (publicBatchWrapper89_copies3856 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_6 := arithEq_of_rows h (row := 2023) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_6
  simp only [← c123390, ← c123391, ← c123392] at e_2023_6
  have e_2023_7 := arithEq_of_rows h (row := 2023) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_7
  simp only [← c123393, ← c123394, k2, ← c123395] at e_2023_7
  have hr := e_2023_7
  simp only [e_2023_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15991)) := by
  have c123396 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.1
  have c123397 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.1
  have c123398 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.1
  have c123399 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.1
  have c123400 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.1
  have c123401 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_8 := arithEq_of_rows h (row := 2023) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_8
  simp only [← c123396, ← c123397, ← c123398] at e_2023_8
  have e_2023_9 := arithEq_of_rows h (row := 2023) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_9
  simp only [← c123399, ← c123400, k2, ← c123401] at e_2023_9
  have hr := e_2023_9
  simp only [e_2023_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15992)) := by
  have c123402 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123403 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123404 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123405 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123406 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123407 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_10 := arithEq_of_rows h (row := 2023) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_10
  simp only [← c123402, ← c123403, ← c123404] at e_2023_10
  have e_2023_11 := arithEq_of_rows h (row := 2023) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_11
  simp only [← c123405, ← c123406, k2, ← c123407] at e_2023_11
  have hr := e_2023_11
  simp only [e_2023_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15993)) := by
  have c123408 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123409 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123410 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123411 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123412 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123413 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_12 := arithEq_of_rows h (row := 2023) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_12
  simp only [← c123408, ← c123409, ← c123410] at e_2023_12
  have e_2023_13 := arithEq_of_rows h (row := 2023) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_13
  simp only [← c123411, ← c123412, k2, ← c123413] at e_2023_13
  have hr := e_2023_13
  simp only [e_2023_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15994)) := by
  have c123414 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123415 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123416 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123417 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123418 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123419 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_14 := arithEq_of_rows h (row := 2023) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_14
  simp only [← c123414, ← c123415, ← c123416] at e_2023_14
  have e_2023_15 := arithEq_of_rows h (row := 2023) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_15
  simp only [← c123417, ← c123418, k2, ← c123419] at e_2023_15
  have hr := e_2023_15
  simp only [e_2023_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15995)) := by
  have c123420 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123421 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123422 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123423 := (publicBatchWrapper89_copies3856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123424 := (publicBatchWrapper89_copies3857 a h).1
  have c123425 := (publicBatchWrapper89_copies3857 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_16 := arithEq_of_rows h (row := 2023) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_16
  simp only [← c123420, ← c123421, ← c123422] at e_2023_16
  have e_2023_17 := arithEq_of_rows h (row := 2023) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_17
  simp only [← c123423, ← c123424, k2, ← c123425] at e_2023_17
  have hr := e_2023_17
  simp only [e_2023_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2023 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15996)) := by
  have c123426 := (publicBatchWrapper89_copies3857 a h).2.2.1
  have c123427 := (publicBatchWrapper89_copies3857 a h).2.2.2.1
  have c123428 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.1
  have c123429 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.1
  have c123430 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.1
  have c123431 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2023_18 := arithEq_of_rows h (row := 2023) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_18
  simp only [← c123426, ← c123427, ← c123428] at e_2023_18
  have e_2023_19 := arithEq_of_rows h (row := 2023) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2023_19
  simp only [← c123429, ← c123430, k2, ← c123431] at e_2023_19
  have hr := e_2023_19
  simp only [e_2023_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15997)) := by
  have c123432 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.1
  have c123433 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.1
  have c123434 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123435 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123436 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123437 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_0 := arithEq_of_rows h (row := 2024) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_0
  simp only [← c123432, ← c123433, ← c123434] at e_2024_0
  have e_2024_1 := arithEq_of_rows h (row := 2024) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_1
  simp only [← c123435, ← c123436, k2, ← c123437] at e_2024_1
  have hr := e_2024_1
  simp only [e_2024_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15998)) := by
  have c123438 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123439 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123440 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123441 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123442 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123443 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_2 := arithEq_of_rows h (row := 2024) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_2
  simp only [← c123438, ← c123439, ← c123440] at e_2024_2
  have e_2024_3 := arithEq_of_rows h (row := 2024) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_3
  simp only [← c123441, ← c123442, k2, ← c123443] at e_2024_3
  have hr := e_2024_3
  simp only [e_2024_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15999)) := by
  have c123444 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123445 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123446 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123447 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123448 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123449 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_4 := arithEq_of_rows h (row := 2024) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_4
  simp only [← c123444, ← c123445, ← c123446] at e_2024_4
  have e_2024_5 := arithEq_of_rows h (row := 2024) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_5
  simp only [← c123447, ← c123448, k2, ← c123449] at e_2024_5
  have hr := e_2024_5
  simp only [e_2024_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 16000)) := by
  have c123450 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123451 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123452 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123453 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123454 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123455 := (publicBatchWrapper89_copies3857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_6 := arithEq_of_rows h (row := 2024) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_6
  simp only [← c123450, ← c123451, ← c123452] at e_2024_6
  have e_2024_7 := arithEq_of_rows h (row := 2024) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_7
  simp only [← c123453, ← c123454, k2, ← c123455] at e_2024_7
  have hr := e_2024_7
  simp only [e_2024_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 16001)) := by
  have c123456 := (publicBatchWrapper89_copies3858 a h).1
  have c123457 := (publicBatchWrapper89_copies3858 a h).2.1
  have c123458 := (publicBatchWrapper89_copies3858 a h).2.2.1
  have c123459 := (publicBatchWrapper89_copies3858 a h).2.2.2.1
  have c123460 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.1
  have c123461 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_8 := arithEq_of_rows h (row := 2024) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_8
  simp only [← c123456, ← c123457, ← c123458] at e_2024_8
  have e_2024_9 := arithEq_of_rows h (row := 2024) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_9
  simp only [← c123459, ← c123460, k2, ← c123461] at e_2024_9
  have hr := e_2024_9
  simp only [e_2024_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16170)) := by
  have c123462 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.1
  have c123463 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.1
  have c123464 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.1
  have c123465 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.1
  have c123466 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123467 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_10 := arithEq_of_rows h (row := 2024) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_10
  simp only [← c123462, ← c123463, ← c123464] at e_2024_10
  have e_2024_11 := arithEq_of_rows h (row := 2024) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_11
  simp only [← c123465, ← c123466, k2, ← c123467] at e_2024_11
  have hr := e_2024_11
  simp only [e_2024_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16171)) := by
  have c123468 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123469 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123470 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123471 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123472 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123473 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_12 := arithEq_of_rows h (row := 2024) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_12
  simp only [← c123468, ← c123469, ← c123470] at e_2024_12
  have e_2024_13 := arithEq_of_rows h (row := 2024) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_13
  simp only [← c123471, ← c123472, k2, ← c123473] at e_2024_13
  have hr := e_2024_13
  simp only [e_2024_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16172)) := by
  have c123474 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123475 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123476 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123477 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123478 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123479 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_14 := arithEq_of_rows h (row := 2024) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_14
  simp only [← c123474, ← c123475, ← c123476] at e_2024_14
  have e_2024_15 := arithEq_of_rows h (row := 2024) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_15
  simp only [← c123477, ← c123478, k2, ← c123479] at e_2024_15
  have hr := e_2024_15
  simp only [e_2024_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16173)) := by
  have c123480 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123481 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123482 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123483 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123484 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123485 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_16 := arithEq_of_rows h (row := 2024) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_16
  simp only [← c123480, ← c123481, ← c123482] at e_2024_16
  have e_2024_17 := arithEq_of_rows h (row := 2024) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_17
  simp only [← c123483, ← c123484, k2, ← c123485] at e_2024_17
  have hr := e_2024_17
  simp only [e_2024_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2024 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16174)) := by
  have c123486 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123487 := (publicBatchWrapper89_copies3858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123488 := (publicBatchWrapper89_copies3859 a h).1
  have c123489 := (publicBatchWrapper89_copies3859 a h).2.1
  have c123490 := (publicBatchWrapper89_copies3859 a h).2.2.1
  have c123491 := (publicBatchWrapper89_copies3859 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2024_18 := arithEq_of_rows h (row := 2024) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_18
  simp only [← c123486, ← c123487, ← c123488] at e_2024_18
  have e_2024_19 := arithEq_of_rows h (row := 2024) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2024_19
  simp only [← c123489, ← c123490, k2, ← c123491] at e_2024_19
  have hr := e_2024_19
  simp only [e_2024_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16175)) := by
  have c123492 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.1
  have c123493 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.1
  have c123494 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.1
  have c123495 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.1
  have c123496 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.1
  have c123497 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_0 := arithEq_of_rows h (row := 2025) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_0
  simp only [← c123492, ← c123493, ← c123494] at e_2025_0
  have e_2025_1 := arithEq_of_rows h (row := 2025) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_1
  simp only [← c123495, ← c123496, k2, ← c123497] at e_2025_1
  have hr := e_2025_1
  simp only [e_2025_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16176)) := by
  have c123498 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123499 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123500 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123501 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123502 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123503 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_2 := arithEq_of_rows h (row := 2025) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_2
  simp only [← c123498, ← c123499, ← c123500] at e_2025_2
  have e_2025_3 := arithEq_of_rows h (row := 2025) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_3
  simp only [← c123501, ← c123502, k2, ← c123503] at e_2025_3
  have hr := e_2025_3
  simp only [e_2025_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16177)) := by
  have c123504 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123505 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123506 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123507 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123508 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123509 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_4 := arithEq_of_rows h (row := 2025) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_4
  simp only [← c123504, ← c123505, ← c123506] at e_2025_4
  have e_2025_5 := arithEq_of_rows h (row := 2025) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_5
  simp only [← c123507, ← c123508, k2, ← c123509] at e_2025_5
  have hr := e_2025_5
  simp only [e_2025_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16178)) := by
  have c123510 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123511 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123512 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123513 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123514 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123515 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_6 := arithEq_of_rows h (row := 2025) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_6
  simp only [← c123510, ← c123511, ← c123512] at e_2025_6
  have e_2025_7 := arithEq_of_rows h (row := 2025) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_7
  simp only [← c123513, ← c123514, k2, ← c123515] at e_2025_7
  have hr := e_2025_7
  simp only [e_2025_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16179)) := by
  have c123516 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123517 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123518 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123519 := (publicBatchWrapper89_copies3859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123520 := (publicBatchWrapper89_copies3860 a h).1
  have c123521 := (publicBatchWrapper89_copies3860 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_8 := arithEq_of_rows h (row := 2025) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_8
  simp only [← c123516, ← c123517, ← c123518] at e_2025_8
  have e_2025_9 := arithEq_of_rows h (row := 2025) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_9
  simp only [← c123519, ← c123520, k2, ← c123521] at e_2025_9
  have hr := e_2025_9
  simp only [e_2025_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16180)) := by
  have c123522 := (publicBatchWrapper89_copies3860 a h).2.2.1
  have c123523 := (publicBatchWrapper89_copies3860 a h).2.2.2.1
  have c123524 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.1
  have c123525 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.1
  have c123526 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.1
  have c123527 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_10 := arithEq_of_rows h (row := 2025) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_10
  simp only [← c123522, ← c123523, ← c123524] at e_2025_10
  have e_2025_11 := arithEq_of_rows h (row := 2025) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_11
  simp only [← c123525, ← c123526, k2, ← c123527] at e_2025_11
  have hr := e_2025_11
  simp only [e_2025_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16181)) := by
  have c123528 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.1
  have c123529 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.1
  have c123530 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123531 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123532 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123533 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_12 := arithEq_of_rows h (row := 2025) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_12
  simp only [← c123528, ← c123529, ← c123530] at e_2025_12
  have e_2025_13 := arithEq_of_rows h (row := 2025) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_13
  simp only [← c123531, ← c123532, k2, ← c123533] at e_2025_13
  have hr := e_2025_13
  simp only [e_2025_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16182)) := by
  have c123534 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123535 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123536 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123537 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123538 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123539 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_14 := arithEq_of_rows h (row := 2025) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_14
  simp only [← c123534, ← c123535, ← c123536] at e_2025_14
  have e_2025_15 := arithEq_of_rows h (row := 2025) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_15
  simp only [← c123537, ← c123538, k2, ← c123539] at e_2025_15
  have hr := e_2025_15
  simp only [e_2025_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16183)) := by
  have c123540 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123541 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123542 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123543 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123544 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123545 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_16 := arithEq_of_rows h (row := 2025) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_16
  simp only [← c123540, ← c123541, ← c123542] at e_2025_16
  have e_2025_17 := arithEq_of_rows h (row := 2025) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_17
  simp only [← c123543, ← c123544, k2, ← c123545] at e_2025_17
  have hr := e_2025_17
  simp only [e_2025_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2025 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16184)) := by
  have c123546 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123547 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123548 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123549 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123550 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123551 := (publicBatchWrapper89_copies3860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2025_18 := arithEq_of_rows h (row := 2025) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_18
  simp only [← c123546, ← c123547, ← c123548] at e_2025_18
  have e_2025_19 := arithEq_of_rows h (row := 2025) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2025_19
  simp only [← c123549, ← c123550, k2, ← c123551] at e_2025_19
  have hr := e_2025_19
  simp only [e_2025_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16185)) := by
  have c123552 := (publicBatchWrapper89_copies3861 a h).1
  have c123553 := (publicBatchWrapper89_copies3861 a h).2.1
  have c123554 := (publicBatchWrapper89_copies3861 a h).2.2.1
  have c123555 := (publicBatchWrapper89_copies3861 a h).2.2.2.1
  have c123556 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.1
  have c123557 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_0 := arithEq_of_rows h (row := 2026) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_0
  simp only [← c123552, ← c123553, ← c123554] at e_2026_0
  have e_2026_1 := arithEq_of_rows h (row := 2026) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_1
  simp only [← c123555, ← c123556, k2, ← c123557] at e_2026_1
  have hr := e_2026_1
  simp only [e_2026_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16186)) := by
  have c123558 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.1
  have c123559 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.1
  have c123560 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.1
  have c123561 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.1
  have c123562 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123563 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_2 := arithEq_of_rows h (row := 2026) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_2
  simp only [← c123558, ← c123559, ← c123560] at e_2026_2
  have e_2026_3 := arithEq_of_rows h (row := 2026) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_3
  simp only [← c123561, ← c123562, k2, ← c123563] at e_2026_3
  have hr := e_2026_3
  simp only [e_2026_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16187)) := by
  have c123564 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123565 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123566 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123567 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123568 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123569 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_4 := arithEq_of_rows h (row := 2026) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_4
  simp only [← c123564, ← c123565, ← c123566] at e_2026_4
  have e_2026_5 := arithEq_of_rows h (row := 2026) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_5
  simp only [← c123567, ← c123568, k2, ← c123569] at e_2026_5
  have hr := e_2026_5
  simp only [e_2026_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16188)) := by
  have c123570 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123571 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123572 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123573 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123574 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123575 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_6 := arithEq_of_rows h (row := 2026) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_6
  simp only [← c123570, ← c123571, ← c123572] at e_2026_6
  have e_2026_7 := arithEq_of_rows h (row := 2026) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_7
  simp only [← c123573, ← c123574, k2, ← c123575] at e_2026_7
  have hr := e_2026_7
  simp only [e_2026_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16189)) := by
  have c123576 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123577 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123578 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123579 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123580 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123581 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_8 := arithEq_of_rows h (row := 2026) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_8
  simp only [← c123576, ← c123577, ← c123578] at e_2026_8
  have e_2026_9 := arithEq_of_rows h (row := 2026) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_9
  simp only [← c123579, ← c123580, k2, ← c123581] at e_2026_9
  have hr := e_2026_9
  simp only [e_2026_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16190)) := by
  have c123582 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123583 := (publicBatchWrapper89_copies3861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123584 := (publicBatchWrapper89_copies3862 a h).1
  have c123585 := (publicBatchWrapper89_copies3862 a h).2.1
  have c123586 := (publicBatchWrapper89_copies3862 a h).2.2.1
  have c123587 := (publicBatchWrapper89_copies3862 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_10 := arithEq_of_rows h (row := 2026) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_10
  simp only [← c123582, ← c123583, ← c123584] at e_2026_10
  have e_2026_11 := arithEq_of_rows h (row := 2026) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_11
  simp only [← c123585, ← c123586, k2, ← c123587] at e_2026_11
  have hr := e_2026_11
  simp only [e_2026_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16191)) := by
  have c123588 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.1
  have c123589 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.1
  have c123590 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.1
  have c123591 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.1
  have c123592 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.1
  have c123593 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_12 := arithEq_of_rows h (row := 2026) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_12
  simp only [← c123588, ← c123589, ← c123590] at e_2026_12
  have e_2026_13 := arithEq_of_rows h (row := 2026) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_13
  simp only [← c123591, ← c123592, k2, ← c123593] at e_2026_13
  have hr := e_2026_13
  simp only [e_2026_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16192)) := by
  have c123594 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123595 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123596 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123597 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123598 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123599 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_14 := arithEq_of_rows h (row := 2026) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_14
  simp only [← c123594, ← c123595, ← c123596] at e_2026_14
  have e_2026_15 := arithEq_of_rows h (row := 2026) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_15
  simp only [← c123597, ← c123598, k2, ← c123599] at e_2026_15
  have hr := e_2026_15
  simp only [e_2026_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16193)) := by
  have c123600 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123601 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123602 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123603 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123604 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123605 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_16 := arithEq_of_rows h (row := 2026) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_16
  simp only [← c123600, ← c123601, ← c123602] at e_2026_16
  have e_2026_17 := arithEq_of_rows h (row := 2026) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_17
  simp only [← c123603, ← c123604, k2, ← c123605] at e_2026_17
  have hr := e_2026_17
  simp only [e_2026_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2026 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16194)) := by
  have c123606 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123607 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123608 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123609 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123610 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123611 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2026_18 := arithEq_of_rows h (row := 2026) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_18
  simp only [← c123606, ← c123607, ← c123608] at e_2026_18
  have e_2026_19 := arithEq_of_rows h (row := 2026) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2026_19
  simp only [← c123609, ← c123610, k2, ← c123611] at e_2026_19
  have hr := e_2026_19
  simp only [e_2026_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16195)) := by
  have c123612 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123613 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123614 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123615 := (publicBatchWrapper89_copies3862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123616 := (publicBatchWrapper89_copies3863 a h).1
  have c123617 := (publicBatchWrapper89_copies3863 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_0 := arithEq_of_rows h (row := 2027) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_0
  simp only [← c123612, ← c123613, ← c123614] at e_2027_0
  have e_2027_1 := arithEq_of_rows h (row := 2027) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_1
  simp only [← c123615, ← c123616, k2, ← c123617] at e_2027_1
  have hr := e_2027_1
  simp only [e_2027_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16196)) := by
  have c123618 := (publicBatchWrapper89_copies3863 a h).2.2.1
  have c123619 := (publicBatchWrapper89_copies3863 a h).2.2.2.1
  have c123620 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.1
  have c123621 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.1
  have c123622 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.1
  have c123623 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_2 := arithEq_of_rows h (row := 2027) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_2
  simp only [← c123618, ← c123619, ← c123620] at e_2027_2
  have e_2027_3 := arithEq_of_rows h (row := 2027) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_3
  simp only [← c123621, ← c123622, k2, ← c123623] at e_2027_3
  have hr := e_2027_3
  simp only [e_2027_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16197)) := by
  have c123624 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.1
  have c123625 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.1
  have c123626 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123627 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123628 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123629 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_4 := arithEq_of_rows h (row := 2027) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_4
  simp only [← c123624, ← c123625, ← c123626] at e_2027_4
  have e_2027_5 := arithEq_of_rows h (row := 2027) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_5
  simp only [← c123627, ← c123628, k2, ← c123629] at e_2027_5
  have hr := e_2027_5
  simp only [e_2027_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16198)) := by
  have c123630 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123631 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123632 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123633 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123634 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123635 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_6 := arithEq_of_rows h (row := 2027) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_6
  simp only [← c123630, ← c123631, ← c123632] at e_2027_6
  have e_2027_7 := arithEq_of_rows h (row := 2027) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_7
  simp only [← c123633, ← c123634, k2, ← c123635] at e_2027_7
  have hr := e_2027_7
  simp only [e_2027_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16199)) := by
  have c123636 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123637 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123638 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123639 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123640 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123641 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_8 := arithEq_of_rows h (row := 2027) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_8
  simp only [← c123636, ← c123637, ← c123638] at e_2027_8
  have e_2027_9 := arithEq_of_rows h (row := 2027) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_9
  simp only [← c123639, ← c123640, k2, ← c123641] at e_2027_9
  have hr := e_2027_9
  simp only [e_2027_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16200)) := by
  have c123642 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123643 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123644 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123645 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123646 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123647 := (publicBatchWrapper89_copies3863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_10 := arithEq_of_rows h (row := 2027) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_10
  simp only [← c123642, ← c123643, ← c123644] at e_2027_10
  have e_2027_11 := arithEq_of_rows h (row := 2027) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_11
  simp only [← c123645, ← c123646, k2, ← c123647] at e_2027_11
  have hr := e_2027_11
  simp only [e_2027_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16201)) := by
  have c123648 := (publicBatchWrapper89_copies3864 a h).1
  have c123649 := (publicBatchWrapper89_copies3864 a h).2.1
  have c123650 := (publicBatchWrapper89_copies3864 a h).2.2.1
  have c123651 := (publicBatchWrapper89_copies3864 a h).2.2.2.1
  have c123652 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.1
  have c123653 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_12 := arithEq_of_rows h (row := 2027) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_12
  simp only [← c123648, ← c123649, ← c123650] at e_2027_12
  have e_2027_13 := arithEq_of_rows h (row := 2027) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_13
  simp only [← c123651, ← c123652, k2, ← c123653] at e_2027_13
  have hr := e_2027_13
  simp only [e_2027_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16370)) := by
  have c123654 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.1
  have c123655 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.1
  have c123656 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.1
  have c123657 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.1
  have c123658 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123659 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_14 := arithEq_of_rows h (row := 2027) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_14
  simp only [← c123654, ← c123655, ← c123656] at e_2027_14
  have e_2027_15 := arithEq_of_rows h (row := 2027) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_15
  simp only [← c123657, ← c123658, k2, ← c123659] at e_2027_15
  have hr := e_2027_15
  simp only [e_2027_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16371)) := by
  have c123660 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123661 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123662 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123663 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123664 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123665 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_16 := arithEq_of_rows h (row := 2027) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_16
  simp only [← c123660, ← c123661, ← c123662] at e_2027_16
  have e_2027_17 := arithEq_of_rows h (row := 2027) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_17
  simp only [← c123663, ← c123664, k2, ← c123665] at e_2027_17
  have hr := e_2027_17
  simp only [e_2027_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2027 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16372)) := by
  have c123666 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123667 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123668 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123669 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123670 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123671 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2027_18 := arithEq_of_rows h (row := 2027) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_18
  simp only [← c123666, ← c123667, ← c123668] at e_2027_18
  have e_2027_19 := arithEq_of_rows h (row := 2027) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2027_19
  simp only [← c123669, ← c123670, k2, ← c123671] at e_2027_19
  have hr := e_2027_19
  simp only [e_2027_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16373)) := by
  have c123672 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123673 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123674 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123675 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123676 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123677 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_0 := arithEq_of_rows h (row := 2028) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_0
  simp only [← c123672, ← c123673, ← c123674] at e_2028_0
  have e_2028_1 := arithEq_of_rows h (row := 2028) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_1
  simp only [← c123675, ← c123676, k2, ← c123677] at e_2028_1
  have hr := e_2028_1
  simp only [e_2028_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16374)) := by
  have c123678 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123679 := (publicBatchWrapper89_copies3864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123680 := (publicBatchWrapper89_copies3865 a h).1
  have c123681 := (publicBatchWrapper89_copies3865 a h).2.1
  have c123682 := (publicBatchWrapper89_copies3865 a h).2.2.1
  have c123683 := (publicBatchWrapper89_copies3865 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_2 := arithEq_of_rows h (row := 2028) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_2
  simp only [← c123678, ← c123679, ← c123680] at e_2028_2
  have e_2028_3 := arithEq_of_rows h (row := 2028) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_3
  simp only [← c123681, ← c123682, k2, ← c123683] at e_2028_3
  have hr := e_2028_3
  simp only [e_2028_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16375)) := by
  have c123684 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.1
  have c123685 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.1
  have c123686 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.1
  have c123687 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.1
  have c123688 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.1
  have c123689 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_4 := arithEq_of_rows h (row := 2028) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_4
  simp only [← c123684, ← c123685, ← c123686] at e_2028_4
  have e_2028_5 := arithEq_of_rows h (row := 2028) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_5
  simp only [← c123687, ← c123688, k2, ← c123689] at e_2028_5
  have hr := e_2028_5
  simp only [e_2028_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16376)) := by
  have c123690 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123691 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123692 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123693 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123694 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123695 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_6 := arithEq_of_rows h (row := 2028) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_6
  simp only [← c123690, ← c123691, ← c123692] at e_2028_6
  have e_2028_7 := arithEq_of_rows h (row := 2028) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_7
  simp only [← c123693, ← c123694, k2, ← c123695] at e_2028_7
  have hr := e_2028_7
  simp only [e_2028_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16377)) := by
  have c123696 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123697 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123698 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123699 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123700 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123701 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_8 := arithEq_of_rows h (row := 2028) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_8
  simp only [← c123696, ← c123697, ← c123698] at e_2028_8
  have e_2028_9 := arithEq_of_rows h (row := 2028) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_9
  simp only [← c123699, ← c123700, k2, ← c123701] at e_2028_9
  have hr := e_2028_9
  simp only [e_2028_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16378)) := by
  have c123702 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123703 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123704 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123705 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123706 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123707 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_10 := arithEq_of_rows h (row := 2028) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_10
  simp only [← c123702, ← c123703, ← c123704] at e_2028_10
  have e_2028_11 := arithEq_of_rows h (row := 2028) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_11
  simp only [← c123705, ← c123706, k2, ← c123707] at e_2028_11
  have hr := e_2028_11
  simp only [e_2028_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16379)) := by
  have c123708 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123709 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123710 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123711 := (publicBatchWrapper89_copies3865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123712 := (publicBatchWrapper89_copies3866 a h).1
  have c123713 := (publicBatchWrapper89_copies3866 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_12 := arithEq_of_rows h (row := 2028) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_12
  simp only [← c123708, ← c123709, ← c123710] at e_2028_12
  have e_2028_13 := arithEq_of_rows h (row := 2028) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_13
  simp only [← c123711, ← c123712, k2, ← c123713] at e_2028_13
  have hr := e_2028_13
  simp only [e_2028_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16380)) := by
  have c123714 := (publicBatchWrapper89_copies3866 a h).2.2.1
  have c123715 := (publicBatchWrapper89_copies3866 a h).2.2.2.1
  have c123716 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.1
  have c123717 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.1
  have c123718 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.1
  have c123719 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_14 := arithEq_of_rows h (row := 2028) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_14
  simp only [← c123714, ← c123715, ← c123716] at e_2028_14
  have e_2028_15 := arithEq_of_rows h (row := 2028) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_15
  simp only [← c123717, ← c123718, k2, ← c123719] at e_2028_15
  have hr := e_2028_15
  simp only [e_2028_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16381)) := by
  have c123720 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.1
  have c123721 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.1
  have c123722 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123723 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123724 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123725 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_16 := arithEq_of_rows h (row := 2028) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_16
  simp only [← c123720, ← c123721, ← c123722] at e_2028_16
  have e_2028_17 := arithEq_of_rows h (row := 2028) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_17
  simp only [← c123723, ← c123724, k2, ← c123725] at e_2028_17
  have hr := e_2028_17
  simp only [e_2028_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2028 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16382)) := by
  have c123726 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123727 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123728 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123729 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123730 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123731 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2028_18 := arithEq_of_rows h (row := 2028) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_18
  simp only [← c123726, ← c123727, ← c123728] at e_2028_18
  have e_2028_19 := arithEq_of_rows h (row := 2028) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2028_19
  simp only [← c123729, ← c123730, k2, ← c123731] at e_2028_19
  have hr := e_2028_19
  simp only [e_2028_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16383)) := by
  have c123732 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123733 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123734 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123735 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123736 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123737 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_0 := arithEq_of_rows h (row := 2029) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_0
  simp only [← c123732, ← c123733, ← c123734] at e_2029_0
  have e_2029_1 := arithEq_of_rows h (row := 2029) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_1
  simp only [← c123735, ← c123736, k2, ← c123737] at e_2029_1
  have hr := e_2029_1
  simp only [e_2029_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16384)) := by
  have c123738 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123739 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123740 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123741 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123742 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123743 := (publicBatchWrapper89_copies3866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_2 := arithEq_of_rows h (row := 2029) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_2
  simp only [← c123738, ← c123739, ← c123740] at e_2029_2
  have e_2029_3 := arithEq_of_rows h (row := 2029) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_3
  simp only [← c123741, ← c123742, k2, ← c123743] at e_2029_3
  have hr := e_2029_3
  simp only [e_2029_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16385)) := by
  have c123744 := (publicBatchWrapper89_copies3867 a h).1
  have c123745 := (publicBatchWrapper89_copies3867 a h).2.1
  have c123746 := (publicBatchWrapper89_copies3867 a h).2.2.1
  have c123747 := (publicBatchWrapper89_copies3867 a h).2.2.2.1
  have c123748 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.1
  have c123749 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_4 := arithEq_of_rows h (row := 2029) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_4
  simp only [← c123744, ← c123745, ← c123746] at e_2029_4
  have e_2029_5 := arithEq_of_rows h (row := 2029) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_5
  simp only [← c123747, ← c123748, k2, ← c123749] at e_2029_5
  have hr := e_2029_5
  simp only [e_2029_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16386)) := by
  have c123750 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.1
  have c123751 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.1
  have c123752 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.1
  have c123753 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.1
  have c123754 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123755 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_6 := arithEq_of_rows h (row := 2029) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_6
  simp only [← c123750, ← c123751, ← c123752] at e_2029_6
  have e_2029_7 := arithEq_of_rows h (row := 2029) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_7
  simp only [← c123753, ← c123754, k2, ← c123755] at e_2029_7
  have hr := e_2029_7
  simp only [e_2029_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16387)) := by
  have c123756 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123757 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123758 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123759 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123760 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123761 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_8 := arithEq_of_rows h (row := 2029) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_8
  simp only [← c123756, ← c123757, ← c123758] at e_2029_8
  have e_2029_9 := arithEq_of_rows h (row := 2029) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_9
  simp only [← c123759, ← c123760, k2, ← c123761] at e_2029_9
  have hr := e_2029_9
  simp only [e_2029_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16388)) := by
  have c123762 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123763 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123764 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123765 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123766 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123767 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_10 := arithEq_of_rows h (row := 2029) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_10
  simp only [← c123762, ← c123763, ← c123764] at e_2029_10
  have e_2029_11 := arithEq_of_rows h (row := 2029) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_11
  simp only [← c123765, ← c123766, k2, ← c123767] at e_2029_11
  have hr := e_2029_11
  simp only [e_2029_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16389)) := by
  have c123768 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123769 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123770 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123771 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123772 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123773 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_12 := arithEq_of_rows h (row := 2029) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_12
  simp only [← c123768, ← c123769, ← c123770] at e_2029_12
  have e_2029_13 := arithEq_of_rows h (row := 2029) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_13
  simp only [← c123771, ← c123772, k2, ← c123773] at e_2029_13
  have hr := e_2029_13
  simp only [e_2029_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16390)) := by
  have c123774 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123775 := (publicBatchWrapper89_copies3867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123776 := (publicBatchWrapper89_copies3868 a h).1
  have c123777 := (publicBatchWrapper89_copies3868 a h).2.1
  have c123778 := (publicBatchWrapper89_copies3868 a h).2.2.1
  have c123779 := (publicBatchWrapper89_copies3868 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_14 := arithEq_of_rows h (row := 2029) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_14
  simp only [← c123774, ← c123775, ← c123776] at e_2029_14
  have e_2029_15 := arithEq_of_rows h (row := 2029) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_15
  simp only [← c123777, ← c123778, k2, ← c123779] at e_2029_15
  have hr := e_2029_15
  simp only [e_2029_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16391)) := by
  have c123780 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.1
  have c123781 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.1
  have c123782 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.1
  have c123783 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.1
  have c123784 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.1
  have c123785 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_16 := arithEq_of_rows h (row := 2029) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_16
  simp only [← c123780, ← c123781, ← c123782] at e_2029_16
  have e_2029_17 := arithEq_of_rows h (row := 2029) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_17
  simp only [← c123783, ← c123784, k2, ← c123785] at e_2029_17
  have hr := e_2029_17
  simp only [e_2029_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2029 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16392)) := by
  have c123786 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123787 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123788 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123789 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123790 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123791 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2029_18 := arithEq_of_rows h (row := 2029) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_18
  simp only [← c123786, ← c123787, ← c123788] at e_2029_18
  have e_2029_19 := arithEq_of_rows h (row := 2029) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2029_19
  simp only [← c123789, ← c123790, k2, ← c123791] at e_2029_19
  have hr := e_2029_19
  simp only [e_2029_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16393)) := by
  have c123792 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123793 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123794 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123795 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123796 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123797 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_0 := arithEq_of_rows h (row := 2030) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_0
  simp only [← c123792, ← c123793, ← c123794] at e_2030_0
  have e_2030_1 := arithEq_of_rows h (row := 2030) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_1
  simp only [← c123795, ← c123796, k2, ← c123797] at e_2030_1
  have hr := e_2030_1
  simp only [e_2030_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16394)) := by
  have c123798 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123799 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123800 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123801 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123802 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123803 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_2 := arithEq_of_rows h (row := 2030) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_2
  simp only [← c123798, ← c123799, ← c123800] at e_2030_2
  have e_2030_3 := arithEq_of_rows h (row := 2030) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_3
  simp only [← c123801, ← c123802, k2, ← c123803] at e_2030_3
  have hr := e_2030_3
  simp only [e_2030_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16395)) := by
  have c123804 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123805 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123806 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123807 := (publicBatchWrapper89_copies3868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123808 := (publicBatchWrapper89_copies3869 a h).1
  have c123809 := (publicBatchWrapper89_copies3869 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_4 := arithEq_of_rows h (row := 2030) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_4
  simp only [← c123804, ← c123805, ← c123806] at e_2030_4
  have e_2030_5 := arithEq_of_rows h (row := 2030) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_5
  simp only [← c123807, ← c123808, k2, ← c123809] at e_2030_5
  have hr := e_2030_5
  simp only [e_2030_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16396)) := by
  have c123810 := (publicBatchWrapper89_copies3869 a h).2.2.1
  have c123811 := (publicBatchWrapper89_copies3869 a h).2.2.2.1
  have c123812 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.1
  have c123813 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.1
  have c123814 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.1
  have c123815 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_6 := arithEq_of_rows h (row := 2030) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_6
  simp only [← c123810, ← c123811, ← c123812] at e_2030_6
  have e_2030_7 := arithEq_of_rows h (row := 2030) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_7
  simp only [← c123813, ← c123814, k2, ← c123815] at e_2030_7
  have hr := e_2030_7
  simp only [e_2030_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16397)) := by
  have c123816 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.1
  have c123817 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.1
  have c123818 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123819 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123820 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123821 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_8 := arithEq_of_rows h (row := 2030) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_8
  simp only [← c123816, ← c123817, ← c123818] at e_2030_8
  have e_2030_9 := arithEq_of_rows h (row := 2030) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_9
  simp only [← c123819, ← c123820, k2, ← c123821] at e_2030_9
  have hr := e_2030_9
  simp only [e_2030_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16398)) := by
  have c123822 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123823 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123824 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123825 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123826 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123827 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_10 := arithEq_of_rows h (row := 2030) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_10
  simp only [← c123822, ← c123823, ← c123824] at e_2030_10
  have e_2030_11 := arithEq_of_rows h (row := 2030) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_11
  simp only [← c123825, ← c123826, k2, ← c123827] at e_2030_11
  have hr := e_2030_11
  simp only [e_2030_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16399)) := by
  have c123828 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123829 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123830 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123831 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123832 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123833 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_12 := arithEq_of_rows h (row := 2030) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_12
  simp only [← c123828, ← c123829, ← c123830] at e_2030_12
  have e_2030_13 := arithEq_of_rows h (row := 2030) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_13
  simp only [← c123831, ← c123832, k2, ← c123833] at e_2030_13
  have hr := e_2030_13
  simp only [e_2030_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16400)) := by
  have c123834 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123835 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123836 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123837 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123838 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123839 := (publicBatchWrapper89_copies3869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_14 := arithEq_of_rows h (row := 2030) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_14
  simp only [← c123834, ← c123835, ← c123836] at e_2030_14
  have e_2030_15 := arithEq_of_rows h (row := 2030) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_15
  simp only [← c123837, ← c123838, k2, ← c123839] at e_2030_15
  have hr := e_2030_15
  simp only [e_2030_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16401)) := by
  have c123840 := (publicBatchWrapper89_copies3870 a h).1
  have c123841 := (publicBatchWrapper89_copies3870 a h).2.1
  have c123842 := (publicBatchWrapper89_copies3870 a h).2.2.1
  have c123843 := (publicBatchWrapper89_copies3870 a h).2.2.2.1
  have c123844 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.1
  have c123845 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_16 := arithEq_of_rows h (row := 2030) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_16
  simp only [← c123840, ← c123841, ← c123842] at e_2030_16
  have e_2030_17 := arithEq_of_rows h (row := 2030) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_17
  simp only [← c123843, ← c123844, k2, ← c123845] at e_2030_17
  have hr := e_2030_17
  simp only [e_2030_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2030 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16570)) := by
  have c123846 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.1
  have c123847 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.1
  have c123848 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.1
  have c123849 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.1
  have c123850 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123851 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2030_18 := arithEq_of_rows h (row := 2030) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_18
  simp only [← c123846, ← c123847, ← c123848] at e_2030_18
  have e_2030_19 := arithEq_of_rows h (row := 2030) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2030_19
  simp only [← c123849, ← c123850, k2, ← c123851] at e_2030_19
  have hr := e_2030_19
  simp only [e_2030_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16571)) := by
  have c123852 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123853 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123854 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123855 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123856 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123857 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_0 := arithEq_of_rows h (row := 2031) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_0
  simp only [← c123852, ← c123853, ← c123854] at e_2031_0
  have e_2031_1 := arithEq_of_rows h (row := 2031) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_1
  simp only [← c123855, ← c123856, k2, ← c123857] at e_2031_1
  have hr := e_2031_1
  simp only [e_2031_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16572)) := by
  have c123858 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123859 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123860 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123861 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123862 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123863 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_2 := arithEq_of_rows h (row := 2031) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_2
  simp only [← c123858, ← c123859, ← c123860] at e_2031_2
  have e_2031_3 := arithEq_of_rows h (row := 2031) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_3
  simp only [← c123861, ← c123862, k2, ← c123863] at e_2031_3
  have hr := e_2031_3
  simp only [e_2031_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16573)) := by
  have c123864 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123865 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123866 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123867 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123868 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123869 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_4 := arithEq_of_rows h (row := 2031) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_4
  simp only [← c123864, ← c123865, ← c123866] at e_2031_4
  have e_2031_5 := arithEq_of_rows h (row := 2031) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_5
  simp only [← c123867, ← c123868, k2, ← c123869] at e_2031_5
  have hr := e_2031_5
  simp only [e_2031_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16574)) := by
  have c123870 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123871 := (publicBatchWrapper89_copies3870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123872 := (publicBatchWrapper89_copies3871 a h).1
  have c123873 := (publicBatchWrapper89_copies3871 a h).2.1
  have c123874 := (publicBatchWrapper89_copies3871 a h).2.2.1
  have c123875 := (publicBatchWrapper89_copies3871 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_6 := arithEq_of_rows h (row := 2031) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_6
  simp only [← c123870, ← c123871, ← c123872] at e_2031_6
  have e_2031_7 := arithEq_of_rows h (row := 2031) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_7
  simp only [← c123873, ← c123874, k2, ← c123875] at e_2031_7
  have hr := e_2031_7
  simp only [e_2031_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16575)) := by
  have c123876 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.1
  have c123877 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.1
  have c123878 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.1
  have c123879 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.1
  have c123880 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.1
  have c123881 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_8 := arithEq_of_rows h (row := 2031) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_8
  simp only [← c123876, ← c123877, ← c123878] at e_2031_8
  have e_2031_9 := arithEq_of_rows h (row := 2031) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_9
  simp only [← c123879, ← c123880, k2, ← c123881] at e_2031_9
  have hr := e_2031_9
  simp only [e_2031_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16576)) := by
  have c123882 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123883 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123884 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123885 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123886 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123887 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_10 := arithEq_of_rows h (row := 2031) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_10
  simp only [← c123882, ← c123883, ← c123884] at e_2031_10
  have e_2031_11 := arithEq_of_rows h (row := 2031) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_11
  simp only [← c123885, ← c123886, k2, ← c123887] at e_2031_11
  have hr := e_2031_11
  simp only [e_2031_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16577)) := by
  have c123888 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123889 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123890 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123891 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123892 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123893 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_12 := arithEq_of_rows h (row := 2031) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_12
  simp only [← c123888, ← c123889, ← c123890] at e_2031_12
  have e_2031_13 := arithEq_of_rows h (row := 2031) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_13
  simp only [← c123891, ← c123892, k2, ← c123893] at e_2031_13
  have hr := e_2031_13
  simp only [e_2031_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16578)) := by
  have c123894 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123895 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123896 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123897 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123898 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123899 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_14 := arithEq_of_rows h (row := 2031) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_14
  simp only [← c123894, ← c123895, ← c123896] at e_2031_14
  have e_2031_15 := arithEq_of_rows h (row := 2031) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_15
  simp only [← c123897, ← c123898, k2, ← c123899] at e_2031_15
  have hr := e_2031_15
  simp only [e_2031_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16579)) := by
  have c123900 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123901 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123902 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123903 := (publicBatchWrapper89_copies3871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123904 := (publicBatchWrapper89_copies3872 a h).1
  have c123905 := (publicBatchWrapper89_copies3872 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_16 := arithEq_of_rows h (row := 2031) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_16
  simp only [← c123900, ← c123901, ← c123902] at e_2031_16
  have e_2031_17 := arithEq_of_rows h (row := 2031) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_17
  simp only [← c123903, ← c123904, k2, ← c123905] at e_2031_17
  have hr := e_2031_17
  simp only [e_2031_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2031 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16580)) := by
  have c123906 := (publicBatchWrapper89_copies3872 a h).2.2.1
  have c123907 := (publicBatchWrapper89_copies3872 a h).2.2.2.1
  have c123908 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.1
  have c123909 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.1
  have c123910 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.1
  have c123911 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2031_18 := arithEq_of_rows h (row := 2031) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_18
  simp only [← c123906, ← c123907, ← c123908] at e_2031_18
  have e_2031_19 := arithEq_of_rows h (row := 2031) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2031_19
  simp only [← c123909, ← c123910, k2, ← c123911] at e_2031_19
  have hr := e_2031_19
  simp only [e_2031_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16581)) := by
  have c123912 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.1
  have c123913 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.1
  have c123914 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123915 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123916 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123917 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_0 := arithEq_of_rows h (row := 2032) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_0
  simp only [← c123912, ← c123913, ← c123914] at e_2032_0
  have e_2032_1 := arithEq_of_rows h (row := 2032) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_1
  simp only [← c123915, ← c123916, k2, ← c123917] at e_2032_1
  have hr := e_2032_1
  simp only [e_2032_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16582)) := by
  have c123918 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123919 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123920 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123921 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123922 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123923 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_2 := arithEq_of_rows h (row := 2032) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_2
  simp only [← c123918, ← c123919, ← c123920] at e_2032_2
  have e_2032_3 := arithEq_of_rows h (row := 2032) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_3
  simp only [← c123921, ← c123922, k2, ← c123923] at e_2032_3
  have hr := e_2032_3
  simp only [e_2032_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16583)) := by
  have c123924 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123925 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123926 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123927 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123928 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123929 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_4 := arithEq_of_rows h (row := 2032) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_4
  simp only [← c123924, ← c123925, ← c123926] at e_2032_4
  have e_2032_5 := arithEq_of_rows h (row := 2032) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_5
  simp only [← c123927, ← c123928, k2, ← c123929] at e_2032_5
  have hr := e_2032_5
  simp only [e_2032_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16584)) := by
  have c123930 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123931 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123932 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123933 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123934 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123935 := (publicBatchWrapper89_copies3872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_6 := arithEq_of_rows h (row := 2032) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_6
  simp only [← c123930, ← c123931, ← c123932] at e_2032_6
  have e_2032_7 := arithEq_of_rows h (row := 2032) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_7
  simp only [← c123933, ← c123934, k2, ← c123935] at e_2032_7
  have hr := e_2032_7
  simp only [e_2032_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16585)) := by
  have c123936 := (publicBatchWrapper89_copies3873 a h).1
  have c123937 := (publicBatchWrapper89_copies3873 a h).2.1
  have c123938 := (publicBatchWrapper89_copies3873 a h).2.2.1
  have c123939 := (publicBatchWrapper89_copies3873 a h).2.2.2.1
  have c123940 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.1
  have c123941 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_8 := arithEq_of_rows h (row := 2032) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_8
  simp only [← c123936, ← c123937, ← c123938] at e_2032_8
  have e_2032_9 := arithEq_of_rows h (row := 2032) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_9
  simp only [← c123939, ← c123940, k2, ← c123941] at e_2032_9
  have hr := e_2032_9
  simp only [e_2032_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16586)) := by
  have c123942 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.1
  have c123943 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.1
  have c123944 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.1
  have c123945 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.1
  have c123946 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123947 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_10 := arithEq_of_rows h (row := 2032) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_10
  simp only [← c123942, ← c123943, ← c123944] at e_2032_10
  have e_2032_11 := arithEq_of_rows h (row := 2032) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_11
  simp only [← c123945, ← c123946, k2, ← c123947] at e_2032_11
  have hr := e_2032_11
  simp only [e_2032_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16587)) := by
  have c123948 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123949 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123950 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123951 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123952 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123953 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_12 := arithEq_of_rows h (row := 2032) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_12
  simp only [← c123948, ← c123949, ← c123950] at e_2032_12
  have e_2032_13 := arithEq_of_rows h (row := 2032) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_13
  simp only [← c123951, ← c123952, k2, ← c123953] at e_2032_13
  have hr := e_2032_13
  simp only [e_2032_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16588)) := by
  have c123954 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123955 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123956 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123957 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123958 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123959 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_14 := arithEq_of_rows h (row := 2032) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_14
  simp only [← c123954, ← c123955, ← c123956] at e_2032_14
  have e_2032_15 := arithEq_of_rows h (row := 2032) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_15
  simp only [← c123957, ← c123958, k2, ← c123959] at e_2032_15
  have hr := e_2032_15
  simp only [e_2032_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16589)) := by
  have c123960 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123961 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123962 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123963 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123964 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123965 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_16 := arithEq_of_rows h (row := 2032) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_16
  simp only [← c123960, ← c123961, ← c123962] at e_2032_16
  have e_2032_17 := arithEq_of_rows h (row := 2032) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_17
  simp only [← c123963, ← c123964, k2, ← c123965] at e_2032_17
  have hr := e_2032_17
  simp only [e_2032_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2032 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16590)) := by
  have c123966 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123967 := (publicBatchWrapper89_copies3873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123968 := (publicBatchWrapper89_copies3874 a h).1
  have c123969 := (publicBatchWrapper89_copies3874 a h).2.1
  have c123970 := (publicBatchWrapper89_copies3874 a h).2.2.1
  have c123971 := (publicBatchWrapper89_copies3874 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2032_18 := arithEq_of_rows h (row := 2032) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_18
  simp only [← c123966, ← c123967, ← c123968] at e_2032_18
  have e_2032_19 := arithEq_of_rows h (row := 2032) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2032_19
  simp only [← c123969, ← c123970, k2, ← c123971] at e_2032_19
  have hr := e_2032_19
  simp only [e_2032_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16591)) := by
  have c123972 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.1
  have c123973 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.1
  have c123974 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.1
  have c123975 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.1
  have c123976 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.1
  have c123977 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_0 := arithEq_of_rows h (row := 2033) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_0
  simp only [← c123972, ← c123973, ← c123974] at e_2033_0
  have e_2033_1 := arithEq_of_rows h (row := 2033) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_1
  simp only [← c123975, ← c123976, k2, ← c123977] at e_2033_1
  have hr := e_2033_1
  simp only [e_2033_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16592)) := by
  have c123978 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123979 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123980 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123981 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123982 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123983 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_2 := arithEq_of_rows h (row := 2033) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_2
  simp only [← c123978, ← c123979, ← c123980] at e_2033_2
  have e_2033_3 := arithEq_of_rows h (row := 2033) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_3
  simp only [← c123981, ← c123982, k2, ← c123983] at e_2033_3
  have hr := e_2033_3
  simp only [e_2033_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16593)) := by
  have c123984 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123985 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123986 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123987 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123988 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123989 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_4 := arithEq_of_rows h (row := 2033) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_4
  simp only [← c123984, ← c123985, ← c123986] at e_2033_4
  have e_2033_5 := arithEq_of_rows h (row := 2033) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_5
  simp only [← c123987, ← c123988, k2, ← c123989] at e_2033_5
  have hr := e_2033_5
  simp only [e_2033_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16594)) := by
  have c123990 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123991 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123992 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123993 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123994 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123995 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_6 := arithEq_of_rows h (row := 2033) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_6
  simp only [← c123990, ← c123991, ← c123992] at e_2033_6
  have e_2033_7 := arithEq_of_rows h (row := 2033) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_7
  simp only [← c123993, ← c123994, k2, ← c123995] at e_2033_7
  have hr := e_2033_7
  simp only [e_2033_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16595)) := by
  have c123996 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123997 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123998 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123999 := (publicBatchWrapper89_copies3874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124000 := (publicBatchWrapper89_copies3875 a h).1
  have c124001 := (publicBatchWrapper89_copies3875 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_8 := arithEq_of_rows h (row := 2033) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_8
  simp only [← c123996, ← c123997, ← c123998] at e_2033_8
  have e_2033_9 := arithEq_of_rows h (row := 2033) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_9
  simp only [← c123999, ← c124000, k2, ← c124001] at e_2033_9
  have hr := e_2033_9
  simp only [e_2033_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16596)) := by
  have c124002 := (publicBatchWrapper89_copies3875 a h).2.2.1
  have c124003 := (publicBatchWrapper89_copies3875 a h).2.2.2.1
  have c124004 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.1
  have c124005 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.1
  have c124006 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.1
  have c124007 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_10 := arithEq_of_rows h (row := 2033) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_10
  simp only [← c124002, ← c124003, ← c124004] at e_2033_10
  have e_2033_11 := arithEq_of_rows h (row := 2033) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_11
  simp only [← c124005, ← c124006, k2, ← c124007] at e_2033_11
  have hr := e_2033_11
  simp only [e_2033_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16597)) := by
  have c124008 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.1
  have c124009 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.1
  have c124010 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124011 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124012 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124013 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_12 := arithEq_of_rows h (row := 2033) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_12
  simp only [← c124008, ← c124009, ← c124010] at e_2033_12
  have e_2033_13 := arithEq_of_rows h (row := 2033) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_13
  simp only [← c124011, ← c124012, k2, ← c124013] at e_2033_13
  have hr := e_2033_13
  simp only [e_2033_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16598)) := by
  have c124014 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124015 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124016 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124017 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124018 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124019 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_14 := arithEq_of_rows h (row := 2033) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_14
  simp only [← c124014, ← c124015, ← c124016] at e_2033_14
  have e_2033_15 := arithEq_of_rows h (row := 2033) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_15
  simp only [← c124017, ← c124018, k2, ← c124019] at e_2033_15
  have hr := e_2033_15
  simp only [e_2033_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16599)) := by
  have c124020 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124021 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124022 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124023 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124024 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124025 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_16 := arithEq_of_rows h (row := 2033) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_16
  simp only [← c124020, ← c124021, ← c124022] at e_2033_16
  have e_2033_17 := arithEq_of_rows h (row := 2033) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_17
  simp only [← c124023, ← c124024, k2, ← c124025] at e_2033_17
  have hr := e_2033_17
  simp only [e_2033_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2033 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16600)) := by
  have c124026 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124027 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124028 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124029 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124030 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124031 := (publicBatchWrapper89_copies3875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2033_18 := arithEq_of_rows h (row := 2033) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_18
  simp only [← c124026, ← c124027, ← c124028] at e_2033_18
  have e_2033_19 := arithEq_of_rows h (row := 2033) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2033_19
  simp only [← c124029, ← c124030, k2, ← c124031] at e_2033_19
  have hr := e_2033_19
  simp only [e_2033_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16601)) := by
  have c124032 := (publicBatchWrapper89_copies3876 a h).1
  have c124033 := (publicBatchWrapper89_copies3876 a h).2.1
  have c124034 := (publicBatchWrapper89_copies3876 a h).2.2.1
  have c124035 := (publicBatchWrapper89_copies3876 a h).2.2.2.1
  have c124036 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.1
  have c124037 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_0 := arithEq_of_rows h (row := 2034) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_0
  simp only [← c124032, ← c124033, ← c124034] at e_2034_0
  have e_2034_1 := arithEq_of_rows h (row := 2034) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_1
  simp only [← c124035, ← c124036, k2, ← c124037] at e_2034_1
  have hr := e_2034_1
  simp only [e_2034_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16770)) := by
  have c124038 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.1
  have c124039 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.1
  have c124040 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.1
  have c124041 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.1
  have c124042 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124043 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_2 := arithEq_of_rows h (row := 2034) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_2
  simp only [← c124038, ← c124039, ← c124040] at e_2034_2
  have e_2034_3 := arithEq_of_rows h (row := 2034) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_3
  simp only [← c124041, ← c124042, k2, ← c124043] at e_2034_3
  have hr := e_2034_3
  simp only [e_2034_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16771)) := by
  have c124044 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124045 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124046 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124047 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124048 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124049 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_4 := arithEq_of_rows h (row := 2034) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_4
  simp only [← c124044, ← c124045, ← c124046] at e_2034_4
  have e_2034_5 := arithEq_of_rows h (row := 2034) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_5
  simp only [← c124047, ← c124048, k2, ← c124049] at e_2034_5
  have hr := e_2034_5
  simp only [e_2034_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16772)) := by
  have c124050 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124051 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124052 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124053 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124054 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124055 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_6 := arithEq_of_rows h (row := 2034) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_6
  simp only [← c124050, ← c124051, ← c124052] at e_2034_6
  have e_2034_7 := arithEq_of_rows h (row := 2034) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_7
  simp only [← c124053, ← c124054, k2, ← c124055] at e_2034_7
  have hr := e_2034_7
  simp only [e_2034_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16773)) := by
  have c124056 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124057 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124058 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124059 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124060 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124061 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_8 := arithEq_of_rows h (row := 2034) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_8
  simp only [← c124056, ← c124057, ← c124058] at e_2034_8
  have e_2034_9 := arithEq_of_rows h (row := 2034) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_9
  simp only [← c124059, ← c124060, k2, ← c124061] at e_2034_9
  have hr := e_2034_9
  simp only [e_2034_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16774)) := by
  have c124062 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124063 := (publicBatchWrapper89_copies3876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124064 := (publicBatchWrapper89_copies3877 a h).1
  have c124065 := (publicBatchWrapper89_copies3877 a h).2.1
  have c124066 := (publicBatchWrapper89_copies3877 a h).2.2.1
  have c124067 := (publicBatchWrapper89_copies3877 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_10 := arithEq_of_rows h (row := 2034) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_10
  simp only [← c124062, ← c124063, ← c124064] at e_2034_10
  have e_2034_11 := arithEq_of_rows h (row := 2034) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_11
  simp only [← c124065, ← c124066, k2, ← c124067] at e_2034_11
  have hr := e_2034_11
  simp only [e_2034_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16775)) := by
  have c124068 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.1
  have c124069 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.1
  have c124070 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.1
  have c124071 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.1
  have c124072 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.1
  have c124073 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_12 := arithEq_of_rows h (row := 2034) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_12
  simp only [← c124068, ← c124069, ← c124070] at e_2034_12
  have e_2034_13 := arithEq_of_rows h (row := 2034) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_13
  simp only [← c124071, ← c124072, k2, ← c124073] at e_2034_13
  have hr := e_2034_13
  simp only [e_2034_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16776)) := by
  have c124074 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124075 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124076 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124077 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124078 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124079 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_14 := arithEq_of_rows h (row := 2034) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_14
  simp only [← c124074, ← c124075, ← c124076] at e_2034_14
  have e_2034_15 := arithEq_of_rows h (row := 2034) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_15
  simp only [← c124077, ← c124078, k2, ← c124079] at e_2034_15
  have hr := e_2034_15
  simp only [e_2034_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16777)) := by
  have c124080 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124081 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124082 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124083 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124084 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124085 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_16 := arithEq_of_rows h (row := 2034) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_16
  simp only [← c124080, ← c124081, ← c124082] at e_2034_16
  have e_2034_17 := arithEq_of_rows h (row := 2034) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_17
  simp only [← c124083, ← c124084, k2, ← c124085] at e_2034_17
  have hr := e_2034_17
  simp only [e_2034_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2034 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16778)) := by
  have c124086 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124087 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124088 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124089 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124090 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124091 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2034_18 := arithEq_of_rows h (row := 2034) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_18
  simp only [← c124086, ← c124087, ← c124088] at e_2034_18
  have e_2034_19 := arithEq_of_rows h (row := 2034) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2034_19
  simp only [← c124089, ← c124090, k2, ← c124091] at e_2034_19
  have hr := e_2034_19
  simp only [e_2034_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16779)) := by
  have c124092 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124093 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124094 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124095 := (publicBatchWrapper89_copies3877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124096 := (publicBatchWrapper89_copies3878 a h).1
  have c124097 := (publicBatchWrapper89_copies3878 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_0 := arithEq_of_rows h (row := 2035) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_0
  simp only [← c124092, ← c124093, ← c124094] at e_2035_0
  have e_2035_1 := arithEq_of_rows h (row := 2035) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_1
  simp only [← c124095, ← c124096, k2, ← c124097] at e_2035_1
  have hr := e_2035_1
  simp only [e_2035_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16780)) := by
  have c124098 := (publicBatchWrapper89_copies3878 a h).2.2.1
  have c124099 := (publicBatchWrapper89_copies3878 a h).2.2.2.1
  have c124100 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.1
  have c124101 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.1
  have c124102 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.1
  have c124103 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_2 := arithEq_of_rows h (row := 2035) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_2
  simp only [← c124098, ← c124099, ← c124100] at e_2035_2
  have e_2035_3 := arithEq_of_rows h (row := 2035) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_3
  simp only [← c124101, ← c124102, k2, ← c124103] at e_2035_3
  have hr := e_2035_3
  simp only [e_2035_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16781)) := by
  have c124104 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.1
  have c124105 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.1
  have c124106 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124107 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124108 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124109 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_4 := arithEq_of_rows h (row := 2035) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_4
  simp only [← c124104, ← c124105, ← c124106] at e_2035_4
  have e_2035_5 := arithEq_of_rows h (row := 2035) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_5
  simp only [← c124107, ← c124108, k2, ← c124109] at e_2035_5
  have hr := e_2035_5
  simp only [e_2035_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16782)) := by
  have c124110 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124111 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124112 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124113 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124114 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124115 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_6 := arithEq_of_rows h (row := 2035) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_6
  simp only [← c124110, ← c124111, ← c124112] at e_2035_6
  have e_2035_7 := arithEq_of_rows h (row := 2035) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_7
  simp only [← c124113, ← c124114, k2, ← c124115] at e_2035_7
  have hr := e_2035_7
  simp only [e_2035_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16783)) := by
  have c124116 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124117 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124118 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124119 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124120 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124121 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_8 := arithEq_of_rows h (row := 2035) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_8
  simp only [← c124116, ← c124117, ← c124118] at e_2035_8
  have e_2035_9 := arithEq_of_rows h (row := 2035) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_9
  simp only [← c124119, ← c124120, k2, ← c124121] at e_2035_9
  have hr := e_2035_9
  simp only [e_2035_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16784)) := by
  have c124122 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124123 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124124 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124125 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124126 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124127 := (publicBatchWrapper89_copies3878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_10 := arithEq_of_rows h (row := 2035) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_10
  simp only [← c124122, ← c124123, ← c124124] at e_2035_10
  have e_2035_11 := arithEq_of_rows h (row := 2035) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_11
  simp only [← c124125, ← c124126, k2, ← c124127] at e_2035_11
  have hr := e_2035_11
  simp only [e_2035_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16785)) := by
  have c124128 := (publicBatchWrapper89_copies3879 a h).1
  have c124129 := (publicBatchWrapper89_copies3879 a h).2.1
  have c124130 := (publicBatchWrapper89_copies3879 a h).2.2.1
  have c124131 := (publicBatchWrapper89_copies3879 a h).2.2.2.1
  have c124132 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.1
  have c124133 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_12 := arithEq_of_rows h (row := 2035) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_12
  simp only [← c124128, ← c124129, ← c124130] at e_2035_12
  have e_2035_13 := arithEq_of_rows h (row := 2035) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_13
  simp only [← c124131, ← c124132, k2, ← c124133] at e_2035_13
  have hr := e_2035_13
  simp only [e_2035_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16786)) := by
  have c124134 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.1
  have c124135 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.1
  have c124136 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.1
  have c124137 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.1
  have c124138 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124139 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_14 := arithEq_of_rows h (row := 2035) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_14
  simp only [← c124134, ← c124135, ← c124136] at e_2035_14
  have e_2035_15 := arithEq_of_rows h (row := 2035) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_15
  simp only [← c124137, ← c124138, k2, ← c124139] at e_2035_15
  have hr := e_2035_15
  simp only [e_2035_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16787)) := by
  have c124140 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124141 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124142 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124143 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124144 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124145 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_16 := arithEq_of_rows h (row := 2035) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_16
  simp only [← c124140, ← c124141, ← c124142] at e_2035_16
  have e_2035_17 := arithEq_of_rows h (row := 2035) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_17
  simp only [← c124143, ← c124144, k2, ← c124145] at e_2035_17
  have hr := e_2035_17
  simp only [e_2035_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2035 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16788)) := by
  have c124146 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124147 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124148 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124149 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124150 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124151 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2035_18 := arithEq_of_rows h (row := 2035) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_18
  simp only [← c124146, ← c124147, ← c124148] at e_2035_18
  have e_2035_19 := arithEq_of_rows h (row := 2035) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2035_19
  simp only [← c124149, ← c124150, k2, ← c124151] at e_2035_19
  have hr := e_2035_19
  simp only [e_2035_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16789)) := by
  have c124152 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124153 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124154 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124155 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124156 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124157 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_0 := arithEq_of_rows h (row := 2036) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_0
  simp only [← c124152, ← c124153, ← c124154] at e_2036_0
  have e_2036_1 := arithEq_of_rows h (row := 2036) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_1
  simp only [← c124155, ← c124156, k2, ← c124157] at e_2036_1
  have hr := e_2036_1
  simp only [e_2036_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16790)) := by
  have c124158 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124159 := (publicBatchWrapper89_copies3879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124160 := (publicBatchWrapper89_copies3880 a h).1
  have c124161 := (publicBatchWrapper89_copies3880 a h).2.1
  have c124162 := (publicBatchWrapper89_copies3880 a h).2.2.1
  have c124163 := (publicBatchWrapper89_copies3880 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_2 := arithEq_of_rows h (row := 2036) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_2
  simp only [← c124158, ← c124159, ← c124160] at e_2036_2
  have e_2036_3 := arithEq_of_rows h (row := 2036) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_3
  simp only [← c124161, ← c124162, k2, ← c124163] at e_2036_3
  have hr := e_2036_3
  simp only [e_2036_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16791)) := by
  have c124164 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.1
  have c124165 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.1
  have c124166 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.1
  have c124167 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.1
  have c124168 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.1
  have c124169 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_4 := arithEq_of_rows h (row := 2036) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_4
  simp only [← c124164, ← c124165, ← c124166] at e_2036_4
  have e_2036_5 := arithEq_of_rows h (row := 2036) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_5
  simp only [← c124167, ← c124168, k2, ← c124169] at e_2036_5
  have hr := e_2036_5
  simp only [e_2036_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16792)) := by
  have c124170 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124171 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124172 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124173 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124174 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124175 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_6 := arithEq_of_rows h (row := 2036) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_6
  simp only [← c124170, ← c124171, ← c124172] at e_2036_6
  have e_2036_7 := arithEq_of_rows h (row := 2036) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_7
  simp only [← c124173, ← c124174, k2, ← c124175] at e_2036_7
  have hr := e_2036_7
  simp only [e_2036_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16793)) := by
  have c124176 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124177 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124178 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124179 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124180 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124181 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_8 := arithEq_of_rows h (row := 2036) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_8
  simp only [← c124176, ← c124177, ← c124178] at e_2036_8
  have e_2036_9 := arithEq_of_rows h (row := 2036) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_9
  simp only [← c124179, ← c124180, k2, ← c124181] at e_2036_9
  have hr := e_2036_9
  simp only [e_2036_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16794)) := by
  have c124182 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124183 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124184 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124185 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124186 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124187 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_10 := arithEq_of_rows h (row := 2036) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_10
  simp only [← c124182, ← c124183, ← c124184] at e_2036_10
  have e_2036_11 := arithEq_of_rows h (row := 2036) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_11
  simp only [← c124185, ← c124186, k2, ← c124187] at e_2036_11
  have hr := e_2036_11
  simp only [e_2036_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16795)) := by
  have c124188 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124189 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124190 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124191 := (publicBatchWrapper89_copies3880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124192 := (publicBatchWrapper89_copies3881 a h).1
  have c124193 := (publicBatchWrapper89_copies3881 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_12 := arithEq_of_rows h (row := 2036) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_12
  simp only [← c124188, ← c124189, ← c124190] at e_2036_12
  have e_2036_13 := arithEq_of_rows h (row := 2036) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_13
  simp only [← c124191, ← c124192, k2, ← c124193] at e_2036_13
  have hr := e_2036_13
  simp only [e_2036_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16796)) := by
  have c124194 := (publicBatchWrapper89_copies3881 a h).2.2.1
  have c124195 := (publicBatchWrapper89_copies3881 a h).2.2.2.1
  have c124196 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.1
  have c124197 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.1
  have c124198 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.1
  have c124199 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_14 := arithEq_of_rows h (row := 2036) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_14
  simp only [← c124194, ← c124195, ← c124196] at e_2036_14
  have e_2036_15 := arithEq_of_rows h (row := 2036) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_15
  simp only [← c124197, ← c124198, k2, ← c124199] at e_2036_15
  have hr := e_2036_15
  simp only [e_2036_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16797)) := by
  have c124200 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.1
  have c124201 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.1
  have c124202 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124203 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124204 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124205 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_16 := arithEq_of_rows h (row := 2036) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_16
  simp only [← c124200, ← c124201, ← c124202] at e_2036_16
  have e_2036_17 := arithEq_of_rows h (row := 2036) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_17
  simp only [← c124203, ← c124204, k2, ← c124205] at e_2036_17
  have hr := e_2036_17
  simp only [e_2036_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2036 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16798)) := by
  have c124206 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124207 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124208 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124209 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124210 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124211 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2036_18 := arithEq_of_rows h (row := 2036) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_18
  simp only [← c124206, ← c124207, ← c124208] at e_2036_18
  have e_2036_19 := arithEq_of_rows h (row := 2036) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2036_19
  simp only [← c124209, ← c124210, k2, ← c124211] at e_2036_19
  have hr := e_2036_19
  simp only [e_2036_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16799)) := by
  have c124212 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124213 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124214 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124215 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124216 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124217 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_0 := arithEq_of_rows h (row := 2037) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_0
  simp only [← c124212, ← c124213, ← c124214] at e_2037_0
  have e_2037_1 := arithEq_of_rows h (row := 2037) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_1
  simp only [← c124215, ← c124216, k2, ← c124217] at e_2037_1
  have hr := e_2037_1
  simp only [e_2037_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16800)) := by
  have c124218 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124219 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124220 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124221 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124222 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124223 := (publicBatchWrapper89_copies3881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_2 := arithEq_of_rows h (row := 2037) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_2
  simp only [← c124218, ← c124219, ← c124220] at e_2037_2
  have e_2037_3 := arithEq_of_rows h (row := 2037) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_3
  simp only [← c124221, ← c124222, k2, ← c124223] at e_2037_3
  have hr := e_2037_3
  simp only [e_2037_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16801)) := by
  have c124224 := (publicBatchWrapper89_copies3882 a h).1
  have c124225 := (publicBatchWrapper89_copies3882 a h).2.1
  have c124226 := (publicBatchWrapper89_copies3882 a h).2.2.1
  have c124227 := (publicBatchWrapper89_copies3882 a h).2.2.2.1
  have c124228 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.1
  have c124229 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_4 := arithEq_of_rows h (row := 2037) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_4
  simp only [← c124224, ← c124225, ← c124226] at e_2037_4
  have e_2037_5 := arithEq_of_rows h (row := 2037) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_5
  simp only [← c124227, ← c124228, k2, ← c124229] at e_2037_5
  have hr := e_2037_5
  simp only [e_2037_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16970)) := by
  have c124230 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.1
  have c124231 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.1
  have c124232 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.1
  have c124233 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.1
  have c124234 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124235 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_6 := arithEq_of_rows h (row := 2037) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_6
  simp only [← c124230, ← c124231, ← c124232] at e_2037_6
  have e_2037_7 := arithEq_of_rows h (row := 2037) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_7
  simp only [← c124233, ← c124234, k2, ← c124235] at e_2037_7
  have hr := e_2037_7
  simp only [e_2037_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16971)) := by
  have c124236 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124237 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124238 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124239 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124240 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124241 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_8 := arithEq_of_rows h (row := 2037) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_8
  simp only [← c124236, ← c124237, ← c124238] at e_2037_8
  have e_2037_9 := arithEq_of_rows h (row := 2037) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_9
  simp only [← c124239, ← c124240, k2, ← c124241] at e_2037_9
  have hr := e_2037_9
  simp only [e_2037_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16972)) := by
  have c124242 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124243 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124244 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124245 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124246 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124247 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_10 := arithEq_of_rows h (row := 2037) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_10
  simp only [← c124242, ← c124243, ← c124244] at e_2037_10
  have e_2037_11 := arithEq_of_rows h (row := 2037) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_11
  simp only [← c124245, ← c124246, k2, ← c124247] at e_2037_11
  have hr := e_2037_11
  simp only [e_2037_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16973)) := by
  have c124248 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124249 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124250 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124251 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124252 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124253 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_12 := arithEq_of_rows h (row := 2037) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_12
  simp only [← c124248, ← c124249, ← c124250] at e_2037_12
  have e_2037_13 := arithEq_of_rows h (row := 2037) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_13
  simp only [← c124251, ← c124252, k2, ← c124253] at e_2037_13
  have hr := e_2037_13
  simp only [e_2037_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16974)) := by
  have c124254 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124255 := (publicBatchWrapper89_copies3882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124256 := (publicBatchWrapper89_copies3883 a h).1
  have c124257 := (publicBatchWrapper89_copies3883 a h).2.1
  have c124258 := (publicBatchWrapper89_copies3883 a h).2.2.1
  have c124259 := (publicBatchWrapper89_copies3883 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_14 := arithEq_of_rows h (row := 2037) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_14
  simp only [← c124254, ← c124255, ← c124256] at e_2037_14
  have e_2037_15 := arithEq_of_rows h (row := 2037) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_15
  simp only [← c124257, ← c124258, k2, ← c124259] at e_2037_15
  have hr := e_2037_15
  simp only [e_2037_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16975)) := by
  have c124260 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.1
  have c124261 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.1
  have c124262 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.1
  have c124263 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.1
  have c124264 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.1
  have c124265 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_16 := arithEq_of_rows h (row := 2037) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_16
  simp only [← c124260, ← c124261, ← c124262] at e_2037_16
  have e_2037_17 := arithEq_of_rows h (row := 2037) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_17
  simp only [← c124263, ← c124264, k2, ← c124265] at e_2037_17
  have hr := e_2037_17
  simp only [e_2037_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2037 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16976)) := by
  have c124266 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124267 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124268 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124269 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124270 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124271 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2037_18 := arithEq_of_rows h (row := 2037) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_18
  simp only [← c124266, ← c124267, ← c124268] at e_2037_18
  have e_2037_19 := arithEq_of_rows h (row := 2037) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2037_19
  simp only [← c124269, ← c124270, k2, ← c124271] at e_2037_19
  have hr := e_2037_19
  simp only [e_2037_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16977)) := by
  have c124272 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124273 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124274 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124275 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124276 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124277 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_0 := arithEq_of_rows h (row := 2038) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_0
  simp only [← c124272, ← c124273, ← c124274] at e_2038_0
  have e_2038_1 := arithEq_of_rows h (row := 2038) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_1
  simp only [← c124275, ← c124276, k2, ← c124277] at e_2038_1
  have hr := e_2038_1
  simp only [e_2038_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16978)) := by
  have c124278 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124279 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124280 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124281 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124282 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124283 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_2 := arithEq_of_rows h (row := 2038) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_2
  simp only [← c124278, ← c124279, ← c124280] at e_2038_2
  have e_2038_3 := arithEq_of_rows h (row := 2038) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_3
  simp only [← c124281, ← c124282, k2, ← c124283] at e_2038_3
  have hr := e_2038_3
  simp only [e_2038_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16979)) := by
  have c124284 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124285 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124286 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124287 := (publicBatchWrapper89_copies3883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124288 := (publicBatchWrapper89_copies3884 a h).1
  have c124289 := (publicBatchWrapper89_copies3884 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_4 := arithEq_of_rows h (row := 2038) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_4
  simp only [← c124284, ← c124285, ← c124286] at e_2038_4
  have e_2038_5 := arithEq_of_rows h (row := 2038) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_5
  simp only [← c124287, ← c124288, k2, ← c124289] at e_2038_5
  have hr := e_2038_5
  simp only [e_2038_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16980)) := by
  have c124290 := (publicBatchWrapper89_copies3884 a h).2.2.1
  have c124291 := (publicBatchWrapper89_copies3884 a h).2.2.2.1
  have c124292 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.1
  have c124293 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.1
  have c124294 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.1
  have c124295 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_6 := arithEq_of_rows h (row := 2038) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_6
  simp only [← c124290, ← c124291, ← c124292] at e_2038_6
  have e_2038_7 := arithEq_of_rows h (row := 2038) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_7
  simp only [← c124293, ← c124294, k2, ← c124295] at e_2038_7
  have hr := e_2038_7
  simp only [e_2038_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16981)) := by
  have c124296 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.1
  have c124297 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.1
  have c124298 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124299 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124300 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124301 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_8 := arithEq_of_rows h (row := 2038) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_8
  simp only [← c124296, ← c124297, ← c124298] at e_2038_8
  have e_2038_9 := arithEq_of_rows h (row := 2038) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_9
  simp only [← c124299, ← c124300, k2, ← c124301] at e_2038_9
  have hr := e_2038_9
  simp only [e_2038_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16982)) := by
  have c124302 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124303 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124304 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124305 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124306 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124307 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_10 := arithEq_of_rows h (row := 2038) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_10
  simp only [← c124302, ← c124303, ← c124304] at e_2038_10
  have e_2038_11 := arithEq_of_rows h (row := 2038) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_11
  simp only [← c124305, ← c124306, k2, ← c124307] at e_2038_11
  have hr := e_2038_11
  simp only [e_2038_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16983)) := by
  have c124308 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124309 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124310 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124311 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124312 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124313 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_12 := arithEq_of_rows h (row := 2038) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_12
  simp only [← c124308, ← c124309, ← c124310] at e_2038_12
  have e_2038_13 := arithEq_of_rows h (row := 2038) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_13
  simp only [← c124311, ← c124312, k2, ← c124313] at e_2038_13
  have hr := e_2038_13
  simp only [e_2038_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16984)) := by
  have c124314 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124315 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124316 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124317 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124318 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124319 := (publicBatchWrapper89_copies3884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_14 := arithEq_of_rows h (row := 2038) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_14
  simp only [← c124314, ← c124315, ← c124316] at e_2038_14
  have e_2038_15 := arithEq_of_rows h (row := 2038) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_15
  simp only [← c124317, ← c124318, k2, ← c124319] at e_2038_15
  have hr := e_2038_15
  simp only [e_2038_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16985)) := by
  have c124320 := (publicBatchWrapper89_copies3885 a h).1
  have c124321 := (publicBatchWrapper89_copies3885 a h).2.1
  have c124322 := (publicBatchWrapper89_copies3885 a h).2.2.1
  have c124323 := (publicBatchWrapper89_copies3885 a h).2.2.2.1
  have c124324 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.1
  have c124325 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_16 := arithEq_of_rows h (row := 2038) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_16
  simp only [← c124320, ← c124321, ← c124322] at e_2038_16
  have e_2038_17 := arithEq_of_rows h (row := 2038) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_17
  simp only [← c124323, ← c124324, k2, ← c124325] at e_2038_17
  have hr := e_2038_17
  simp only [e_2038_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2038 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16986)) := by
  have c124326 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.1
  have c124327 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.1
  have c124328 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.1
  have c124329 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.1
  have c124330 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124331 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2038_18 := arithEq_of_rows h (row := 2038) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_18
  simp only [← c124326, ← c124327, ← c124328] at e_2038_18
  have e_2038_19 := arithEq_of_rows h (row := 2038) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2038_19
  simp only [← c124329, ← c124330, k2, ← c124331] at e_2038_19
  have hr := e_2038_19
  simp only [e_2038_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16987)) := by
  have c124332 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124333 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124334 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124335 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124336 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124337 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_0 := arithEq_of_rows h (row := 2039) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_0
  simp only [← c124332, ← c124333, ← c124334] at e_2039_0
  have e_2039_1 := arithEq_of_rows h (row := 2039) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_1
  simp only [← c124335, ← c124336, k2, ← c124337] at e_2039_1
  have hr := e_2039_1
  simp only [e_2039_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16988)) := by
  have c124338 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124339 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124340 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124341 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124342 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124343 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_2 := arithEq_of_rows h (row := 2039) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_2
  simp only [← c124338, ← c124339, ← c124340] at e_2039_2
  have e_2039_3 := arithEq_of_rows h (row := 2039) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_3
  simp only [← c124341, ← c124342, k2, ← c124343] at e_2039_3
  have hr := e_2039_3
  simp only [e_2039_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16989)) := by
  have c124344 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124345 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124346 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124347 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124348 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124349 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_4 := arithEq_of_rows h (row := 2039) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_4
  simp only [← c124344, ← c124345, ← c124346] at e_2039_4
  have e_2039_5 := arithEq_of_rows h (row := 2039) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_5
  simp only [← c124347, ← c124348, k2, ← c124349] at e_2039_5
  have hr := e_2039_5
  simp only [e_2039_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16990)) := by
  have c124350 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124351 := (publicBatchWrapper89_copies3885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124352 := (publicBatchWrapper89_copies3886 a h).1
  have c124353 := (publicBatchWrapper89_copies3886 a h).2.1
  have c124354 := (publicBatchWrapper89_copies3886 a h).2.2.1
  have c124355 := (publicBatchWrapper89_copies3886 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_6 := arithEq_of_rows h (row := 2039) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_6
  simp only [← c124350, ← c124351, ← c124352] at e_2039_6
  have e_2039_7 := arithEq_of_rows h (row := 2039) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_7
  simp only [← c124353, ← c124354, k2, ← c124355] at e_2039_7
  have hr := e_2039_7
  simp only [e_2039_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16991)) := by
  have c124356 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.1
  have c124357 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.1
  have c124358 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.1
  have c124359 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.1
  have c124360 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.1
  have c124361 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_8 := arithEq_of_rows h (row := 2039) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_8
  simp only [← c124356, ← c124357, ← c124358] at e_2039_8
  have e_2039_9 := arithEq_of_rows h (row := 2039) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_9
  simp only [← c124359, ← c124360, k2, ← c124361] at e_2039_9
  have hr := e_2039_9
  simp only [e_2039_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16992)) := by
  have c124362 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124363 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124364 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124365 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124366 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124367 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_10 := arithEq_of_rows h (row := 2039) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_10
  simp only [← c124362, ← c124363, ← c124364] at e_2039_10
  have e_2039_11 := arithEq_of_rows h (row := 2039) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_11
  simp only [← c124365, ← c124366, k2, ← c124367] at e_2039_11
  have hr := e_2039_11
  simp only [e_2039_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16993)) := by
  have c124368 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124369 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124370 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124371 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124372 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124373 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_12 := arithEq_of_rows h (row := 2039) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_12
  simp only [← c124368, ← c124369, ← c124370] at e_2039_12
  have e_2039_13 := arithEq_of_rows h (row := 2039) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_13
  simp only [← c124371, ← c124372, k2, ← c124373] at e_2039_13
  have hr := e_2039_13
  simp only [e_2039_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16994)) := by
  have c124374 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124375 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124376 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124377 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124378 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124379 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_14 := arithEq_of_rows h (row := 2039) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_14
  simp only [← c124374, ← c124375, ← c124376] at e_2039_14
  have e_2039_15 := arithEq_of_rows h (row := 2039) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_15
  simp only [← c124377, ← c124378, k2, ← c124379] at e_2039_15
  have hr := e_2039_15
  simp only [e_2039_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16995)) := by
  have c124380 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124381 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124382 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124383 := (publicBatchWrapper89_copies3886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124384 := (publicBatchWrapper89_copies3887 a h).1
  have c124385 := (publicBatchWrapper89_copies3887 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_16 := arithEq_of_rows h (row := 2039) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_16
  simp only [← c124380, ← c124381, ← c124382] at e_2039_16
  have e_2039_17 := arithEq_of_rows h (row := 2039) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_17
  simp only [← c124383, ← c124384, k2, ← c124385] at e_2039_17
  have hr := e_2039_17
  simp only [e_2039_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2039 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16996)) := by
  have c124386 := (publicBatchWrapper89_copies3887 a h).2.2.1
  have c124387 := (publicBatchWrapper89_copies3887 a h).2.2.2.1
  have c124388 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.1
  have c124389 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.1
  have c124390 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.1
  have c124391 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2039_18 := arithEq_of_rows h (row := 2039) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_18
  simp only [← c124386, ← c124387, ← c124388] at e_2039_18
  have e_2039_19 := arithEq_of_rows h (row := 2039) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2039_19
  simp only [← c124389, ← c124390, k2, ← c124391] at e_2039_19
  have hr := e_2039_19
  simp only [e_2039_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16997)) := by
  have c124392 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.1
  have c124393 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.1
  have c124394 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124395 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124396 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124397 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_0 := arithEq_of_rows h (row := 2040) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_0
  simp only [← c124392, ← c124393, ← c124394] at e_2040_0
  have e_2040_1 := arithEq_of_rows h (row := 2040) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_1
  simp only [← c124395, ← c124396, k2, ← c124397] at e_2040_1
  have hr := e_2040_1
  simp only [e_2040_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16998)) := by
  have c124398 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124399 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124400 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124401 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124402 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124403 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_2 := arithEq_of_rows h (row := 2040) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_2
  simp only [← c124398, ← c124399, ← c124400] at e_2040_2
  have e_2040_3 := arithEq_of_rows h (row := 2040) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_3
  simp only [← c124401, ← c124402, k2, ← c124403] at e_2040_3
  have hr := e_2040_3
  simp only [e_2040_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16999)) := by
  have c124404 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124405 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124406 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124407 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124408 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124409 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_4 := arithEq_of_rows h (row := 2040) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_4
  simp only [← c124404, ← c124405, ← c124406] at e_2040_4
  have e_2040_5 := arithEq_of_rows h (row := 2040) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_5
  simp only [← c124407, ← c124408, k2, ← c124409] at e_2040_5
  have hr := e_2040_5
  simp only [e_2040_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 17000)) := by
  have c124410 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124411 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124412 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124413 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124414 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124415 := (publicBatchWrapper89_copies3887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_6 := arithEq_of_rows h (row := 2040) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_6
  simp only [← c124410, ← c124411, ← c124412] at e_2040_6
  have e_2040_7 := arithEq_of_rows h (row := 2040) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_7
  simp only [← c124413, ← c124414, k2, ← c124415] at e_2040_7
  have hr := e_2040_7
  simp only [e_2040_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 17001)) := by
  have c124416 := (publicBatchWrapper89_copies3888 a h).1
  have c124417 := (publicBatchWrapper89_copies3888 a h).2.1
  have c124418 := (publicBatchWrapper89_copies3888 a h).2.2.1
  have c124419 := (publicBatchWrapper89_copies3888 a h).2.2.2.1
  have c124420 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.1
  have c124421 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_8 := arithEq_of_rows h (row := 2040) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_8
  simp only [← c124416, ← c124417, ← c124418] at e_2040_8
  have e_2040_9 := arithEq_of_rows h (row := 2040) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_9
  simp only [← c124419, ← c124420, k2, ← c124421] at e_2040_9
  have hr := e_2040_9
  simp only [e_2040_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17170)) := by
  have c124422 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.1
  have c124423 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.1
  have c124424 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.1
  have c124425 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.1
  have c124426 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124427 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_10 := arithEq_of_rows h (row := 2040) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_10
  simp only [← c124422, ← c124423, ← c124424] at e_2040_10
  have e_2040_11 := arithEq_of_rows h (row := 2040) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_11
  simp only [← c124425, ← c124426, k2, ← c124427] at e_2040_11
  have hr := e_2040_11
  simp only [e_2040_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17171)) := by
  have c124428 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124429 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124430 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124431 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124432 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124433 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_12 := arithEq_of_rows h (row := 2040) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_12
  simp only [← c124428, ← c124429, ← c124430] at e_2040_12
  have e_2040_13 := arithEq_of_rows h (row := 2040) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_13
  simp only [← c124431, ← c124432, k2, ← c124433] at e_2040_13
  have hr := e_2040_13
  simp only [e_2040_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17172)) := by
  have c124434 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124435 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124436 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124437 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124438 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124439 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_14 := arithEq_of_rows h (row := 2040) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_14
  simp only [← c124434, ← c124435, ← c124436] at e_2040_14
  have e_2040_15 := arithEq_of_rows h (row := 2040) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_15
  simp only [← c124437, ← c124438, k2, ← c124439] at e_2040_15
  have hr := e_2040_15
  simp only [e_2040_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17173)) := by
  have c124440 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124441 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124442 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124443 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124444 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124445 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_16 := arithEq_of_rows h (row := 2040) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_16
  simp only [← c124440, ← c124441, ← c124442] at e_2040_16
  have e_2040_17 := arithEq_of_rows h (row := 2040) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_17
  simp only [← c124443, ← c124444, k2, ← c124445] at e_2040_17
  have hr := e_2040_17
  simp only [e_2040_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2040 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17174)) := by
  have c124446 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124447 := (publicBatchWrapper89_copies3888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124448 := (publicBatchWrapper89_copies3889 a h).1
  have c124449 := (publicBatchWrapper89_copies3889 a h).2.1
  have c124450 := (publicBatchWrapper89_copies3889 a h).2.2.1
  have c124451 := (publicBatchWrapper89_copies3889 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2040_18 := arithEq_of_rows h (row := 2040) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_18
  simp only [← c124446, ← c124447, ← c124448] at e_2040_18
  have e_2040_19 := arithEq_of_rows h (row := 2040) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2040_19
  simp only [← c124449, ← c124450, k2, ← c124451] at e_2040_19
  have hr := e_2040_19
  simp only [e_2040_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17175)) := by
  have c124452 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.1
  have c124453 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.1
  have c124454 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.1
  have c124455 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.1
  have c124456 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.1
  have c124457 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_0 := arithEq_of_rows h (row := 2041) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_0
  simp only [← c124452, ← c124453, ← c124454] at e_2041_0
  have e_2041_1 := arithEq_of_rows h (row := 2041) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_1
  simp only [← c124455, ← c124456, k2, ← c124457] at e_2041_1
  have hr := e_2041_1
  simp only [e_2041_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17176)) := by
  have c124458 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124459 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124460 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124461 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124462 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124463 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_2 := arithEq_of_rows h (row := 2041) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_2
  simp only [← c124458, ← c124459, ← c124460] at e_2041_2
  have e_2041_3 := arithEq_of_rows h (row := 2041) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_3
  simp only [← c124461, ← c124462, k2, ← c124463] at e_2041_3
  have hr := e_2041_3
  simp only [e_2041_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17177)) := by
  have c124464 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124465 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124466 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124467 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124468 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124469 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_4 := arithEq_of_rows h (row := 2041) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_4
  simp only [← c124464, ← c124465, ← c124466] at e_2041_4
  have e_2041_5 := arithEq_of_rows h (row := 2041) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_5
  simp only [← c124467, ← c124468, k2, ← c124469] at e_2041_5
  have hr := e_2041_5
  simp only [e_2041_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17178)) := by
  have c124470 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124471 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124472 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124473 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124474 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124475 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_6 := arithEq_of_rows h (row := 2041) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_6
  simp only [← c124470, ← c124471, ← c124472] at e_2041_6
  have e_2041_7 := arithEq_of_rows h (row := 2041) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_7
  simp only [← c124473, ← c124474, k2, ← c124475] at e_2041_7
  have hr := e_2041_7
  simp only [e_2041_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17179)) := by
  have c124476 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124477 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124478 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124479 := (publicBatchWrapper89_copies3889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124480 := (publicBatchWrapper89_copies3890 a h).1
  have c124481 := (publicBatchWrapper89_copies3890 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_8 := arithEq_of_rows h (row := 2041) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_8
  simp only [← c124476, ← c124477, ← c124478] at e_2041_8
  have e_2041_9 := arithEq_of_rows h (row := 2041) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_9
  simp only [← c124479, ← c124480, k2, ← c124481] at e_2041_9
  have hr := e_2041_9
  simp only [e_2041_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17180)) := by
  have c124482 := (publicBatchWrapper89_copies3890 a h).2.2.1
  have c124483 := (publicBatchWrapper89_copies3890 a h).2.2.2.1
  have c124484 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.1
  have c124485 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.1
  have c124486 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.1
  have c124487 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_10 := arithEq_of_rows h (row := 2041) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_10
  simp only [← c124482, ← c124483, ← c124484] at e_2041_10
  have e_2041_11 := arithEq_of_rows h (row := 2041) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_11
  simp only [← c124485, ← c124486, k2, ← c124487] at e_2041_11
  have hr := e_2041_11
  simp only [e_2041_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17181)) := by
  have c124488 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.1
  have c124489 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.1
  have c124490 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124491 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124492 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124493 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_12 := arithEq_of_rows h (row := 2041) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_12
  simp only [← c124488, ← c124489, ← c124490] at e_2041_12
  have e_2041_13 := arithEq_of_rows h (row := 2041) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_13
  simp only [← c124491, ← c124492, k2, ← c124493] at e_2041_13
  have hr := e_2041_13
  simp only [e_2041_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17182)) := by
  have c124494 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124495 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124496 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124497 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124498 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124499 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_14 := arithEq_of_rows h (row := 2041) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_14
  simp only [← c124494, ← c124495, ← c124496] at e_2041_14
  have e_2041_15 := arithEq_of_rows h (row := 2041) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_15
  simp only [← c124497, ← c124498, k2, ← c124499] at e_2041_15
  have hr := e_2041_15
  simp only [e_2041_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17183)) := by
  have c124500 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124501 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124502 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124503 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124504 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124505 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_16 := arithEq_of_rows h (row := 2041) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_16
  simp only [← c124500, ← c124501, ← c124502] at e_2041_16
  have e_2041_17 := arithEq_of_rows h (row := 2041) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_17
  simp only [← c124503, ← c124504, k2, ← c124505] at e_2041_17
  have hr := e_2041_17
  simp only [e_2041_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2041 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17184)) := by
  have c124506 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124507 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124508 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124509 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124510 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124511 := (publicBatchWrapper89_copies3890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2041_18 := arithEq_of_rows h (row := 2041) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_18
  simp only [← c124506, ← c124507, ← c124508] at e_2041_18
  have e_2041_19 := arithEq_of_rows h (row := 2041) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2041_19
  simp only [← c124509, ← c124510, k2, ← c124511] at e_2041_19
  have hr := e_2041_19
  simp only [e_2041_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17185)) := by
  have c124512 := (publicBatchWrapper89_copies3891 a h).1
  have c124513 := (publicBatchWrapper89_copies3891 a h).2.1
  have c124514 := (publicBatchWrapper89_copies3891 a h).2.2.1
  have c124515 := (publicBatchWrapper89_copies3891 a h).2.2.2.1
  have c124516 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.1
  have c124517 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_0 := arithEq_of_rows h (row := 2042) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_0
  simp only [← c124512, ← c124513, ← c124514] at e_2042_0
  have e_2042_1 := arithEq_of_rows h (row := 2042) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_1
  simp only [← c124515, ← c124516, k2, ← c124517] at e_2042_1
  have hr := e_2042_1
  simp only [e_2042_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17186)) := by
  have c124518 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.1
  have c124519 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.1
  have c124520 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.1
  have c124521 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.1
  have c124522 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124523 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_2 := arithEq_of_rows h (row := 2042) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_2
  simp only [← c124518, ← c124519, ← c124520] at e_2042_2
  have e_2042_3 := arithEq_of_rows h (row := 2042) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_3
  simp only [← c124521, ← c124522, k2, ← c124523] at e_2042_3
  have hr := e_2042_3
  simp only [e_2042_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17187)) := by
  have c124524 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124525 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124526 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124527 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124528 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124529 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_4 := arithEq_of_rows h (row := 2042) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_4
  simp only [← c124524, ← c124525, ← c124526] at e_2042_4
  have e_2042_5 := arithEq_of_rows h (row := 2042) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_5
  simp only [← c124527, ← c124528, k2, ← c124529] at e_2042_5
  have hr := e_2042_5
  simp only [e_2042_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17188)) := by
  have c124530 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124531 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124532 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124533 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124534 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124535 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_6 := arithEq_of_rows h (row := 2042) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_6
  simp only [← c124530, ← c124531, ← c124532] at e_2042_6
  have e_2042_7 := arithEq_of_rows h (row := 2042) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_7
  simp only [← c124533, ← c124534, k2, ← c124535] at e_2042_7
  have hr := e_2042_7
  simp only [e_2042_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17189)) := by
  have c124536 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124537 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124538 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124539 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124540 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124541 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_8 := arithEq_of_rows h (row := 2042) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_8
  simp only [← c124536, ← c124537, ← c124538] at e_2042_8
  have e_2042_9 := arithEq_of_rows h (row := 2042) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_9
  simp only [← c124539, ← c124540, k2, ← c124541] at e_2042_9
  have hr := e_2042_9
  simp only [e_2042_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17190)) := by
  have c124542 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124543 := (publicBatchWrapper89_copies3891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124544 := (publicBatchWrapper89_copies3892 a h).1
  have c124545 := (publicBatchWrapper89_copies3892 a h).2.1
  have c124546 := (publicBatchWrapper89_copies3892 a h).2.2.1
  have c124547 := (publicBatchWrapper89_copies3892 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_10 := arithEq_of_rows h (row := 2042) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_10
  simp only [← c124542, ← c124543, ← c124544] at e_2042_10
  have e_2042_11 := arithEq_of_rows h (row := 2042) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_11
  simp only [← c124545, ← c124546, k2, ← c124547] at e_2042_11
  have hr := e_2042_11
  simp only [e_2042_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17191)) := by
  have c124548 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.1
  have c124549 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.1
  have c124550 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.1
  have c124551 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.1
  have c124552 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.1
  have c124553 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_12 := arithEq_of_rows h (row := 2042) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_12
  simp only [← c124548, ← c124549, ← c124550] at e_2042_12
  have e_2042_13 := arithEq_of_rows h (row := 2042) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_13
  simp only [← c124551, ← c124552, k2, ← c124553] at e_2042_13
  have hr := e_2042_13
  simp only [e_2042_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17192)) := by
  have c124554 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124555 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124556 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124557 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124558 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124559 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_14 := arithEq_of_rows h (row := 2042) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_14
  simp only [← c124554, ← c124555, ← c124556] at e_2042_14
  have e_2042_15 := arithEq_of_rows h (row := 2042) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_15
  simp only [← c124557, ← c124558, k2, ← c124559] at e_2042_15
  have hr := e_2042_15
  simp only [e_2042_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17193)) := by
  have c124560 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124561 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124562 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124563 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124564 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124565 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_16 := arithEq_of_rows h (row := 2042) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_16
  simp only [← c124560, ← c124561, ← c124562] at e_2042_16
  have e_2042_17 := arithEq_of_rows h (row := 2042) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_17
  simp only [← c124563, ← c124564, k2, ← c124565] at e_2042_17
  have hr := e_2042_17
  simp only [e_2042_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2042 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17194)) := by
  have c124566 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124567 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124568 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124569 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124570 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124571 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2042_18 := arithEq_of_rows h (row := 2042) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_18
  simp only [← c124566, ← c124567, ← c124568] at e_2042_18
  have e_2042_19 := arithEq_of_rows h (row := 2042) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2042_19
  simp only [← c124569, ← c124570, k2, ← c124571] at e_2042_19
  have hr := e_2042_19
  simp only [e_2042_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17195)) := by
  have c124572 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124573 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124574 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124575 := (publicBatchWrapper89_copies3892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124576 := (publicBatchWrapper89_copies3893 a h).1
  have c124577 := (publicBatchWrapper89_copies3893 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_0 := arithEq_of_rows h (row := 2043) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_0
  simp only [← c124572, ← c124573, ← c124574] at e_2043_0
  have e_2043_1 := arithEq_of_rows h (row := 2043) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_1
  simp only [← c124575, ← c124576, k2, ← c124577] at e_2043_1
  have hr := e_2043_1
  simp only [e_2043_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17196)) := by
  have c124578 := (publicBatchWrapper89_copies3893 a h).2.2.1
  have c124579 := (publicBatchWrapper89_copies3893 a h).2.2.2.1
  have c124580 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.1
  have c124581 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.1
  have c124582 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.1
  have c124583 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_2 := arithEq_of_rows h (row := 2043) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_2
  simp only [← c124578, ← c124579, ← c124580] at e_2043_2
  have e_2043_3 := arithEq_of_rows h (row := 2043) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_3
  simp only [← c124581, ← c124582, k2, ← c124583] at e_2043_3
  have hr := e_2043_3
  simp only [e_2043_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17197)) := by
  have c124584 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.1
  have c124585 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.1
  have c124586 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124587 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124588 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124589 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_4 := arithEq_of_rows h (row := 2043) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_4
  simp only [← c124584, ← c124585, ← c124586] at e_2043_4
  have e_2043_5 := arithEq_of_rows h (row := 2043) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_5
  simp only [← c124587, ← c124588, k2, ← c124589] at e_2043_5
  have hr := e_2043_5
  simp only [e_2043_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17198)) := by
  have c124590 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124591 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124592 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124593 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124594 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124595 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_6 := arithEq_of_rows h (row := 2043) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_6
  simp only [← c124590, ← c124591, ← c124592] at e_2043_6
  have e_2043_7 := arithEq_of_rows h (row := 2043) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_7
  simp only [← c124593, ← c124594, k2, ← c124595] at e_2043_7
  have hr := e_2043_7
  simp only [e_2043_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17199)) := by
  have c124596 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124597 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124598 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124599 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124600 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124601 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_8 := arithEq_of_rows h (row := 2043) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_8
  simp only [← c124596, ← c124597, ← c124598] at e_2043_8
  have e_2043_9 := arithEq_of_rows h (row := 2043) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_9
  simp only [← c124599, ← c124600, k2, ← c124601] at e_2043_9
  have hr := e_2043_9
  simp only [e_2043_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17200)) := by
  have c124602 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124603 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124604 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124605 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124606 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124607 := (publicBatchWrapper89_copies3893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_10 := arithEq_of_rows h (row := 2043) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_10
  simp only [← c124602, ← c124603, ← c124604] at e_2043_10
  have e_2043_11 := arithEq_of_rows h (row := 2043) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_11
  simp only [← c124605, ← c124606, k2, ← c124607] at e_2043_11
  have hr := e_2043_11
  simp only [e_2043_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17201)) := by
  have c124608 := (publicBatchWrapper89_copies3894 a h).1
  have c124609 := (publicBatchWrapper89_copies3894 a h).2.1
  have c124610 := (publicBatchWrapper89_copies3894 a h).2.2.1
  have c124611 := (publicBatchWrapper89_copies3894 a h).2.2.2.1
  have c124612 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.1
  have c124613 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_12 := arithEq_of_rows h (row := 2043) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_12
  simp only [← c124608, ← c124609, ← c124610] at e_2043_12
  have e_2043_13 := arithEq_of_rows h (row := 2043) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_13
  simp only [← c124611, ← c124612, k2, ← c124613] at e_2043_13
  have hr := e_2043_13
  simp only [e_2043_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17370)) := by
  have c124614 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.1
  have c124615 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.1
  have c124616 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.1
  have c124617 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.1
  have c124618 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124619 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_14 := arithEq_of_rows h (row := 2043) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_14
  simp only [← c124614, ← c124615, ← c124616] at e_2043_14
  have e_2043_15 := arithEq_of_rows h (row := 2043) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_15
  simp only [← c124617, ← c124618, k2, ← c124619] at e_2043_15
  have hr := e_2043_15
  simp only [e_2043_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17371)) := by
  have c124620 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124621 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124622 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124623 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124624 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124625 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_16 := arithEq_of_rows h (row := 2043) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_16
  simp only [← c124620, ← c124621, ← c124622] at e_2043_16
  have e_2043_17 := arithEq_of_rows h (row := 2043) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_17
  simp only [← c124623, ← c124624, k2, ← c124625] at e_2043_17
  have hr := e_2043_17
  simp only [e_2043_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2043 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17372)) := by
  have c124626 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124627 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124628 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124629 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124630 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124631 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2043_18 := arithEq_of_rows h (row := 2043) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_18
  simp only [← c124626, ← c124627, ← c124628] at e_2043_18
  have e_2043_19 := arithEq_of_rows h (row := 2043) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2043_19
  simp only [← c124629, ← c124630, k2, ← c124631] at e_2043_19
  have hr := e_2043_19
  simp only [e_2043_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17373)) := by
  have c124632 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124633 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124634 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124635 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124636 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124637 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_0 := arithEq_of_rows h (row := 2044) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_0
  simp only [← c124632, ← c124633, ← c124634] at e_2044_0
  have e_2044_1 := arithEq_of_rows h (row := 2044) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_1
  simp only [← c124635, ← c124636, k2, ← c124637] at e_2044_1
  have hr := e_2044_1
  simp only [e_2044_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17374)) := by
  have c124638 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124639 := (publicBatchWrapper89_copies3894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124640 := (publicBatchWrapper89_copies3895 a h).1
  have c124641 := (publicBatchWrapper89_copies3895 a h).2.1
  have c124642 := (publicBatchWrapper89_copies3895 a h).2.2.1
  have c124643 := (publicBatchWrapper89_copies3895 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_2 := arithEq_of_rows h (row := 2044) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_2
  simp only [← c124638, ← c124639, ← c124640] at e_2044_2
  have e_2044_3 := arithEq_of_rows h (row := 2044) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_3
  simp only [← c124641, ← c124642, k2, ← c124643] at e_2044_3
  have hr := e_2044_3
  simp only [e_2044_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17375)) := by
  have c124644 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.1
  have c124645 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.1
  have c124646 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.1
  have c124647 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.1
  have c124648 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.1
  have c124649 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_4 := arithEq_of_rows h (row := 2044) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_4
  simp only [← c124644, ← c124645, ← c124646] at e_2044_4
  have e_2044_5 := arithEq_of_rows h (row := 2044) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_5
  simp only [← c124647, ← c124648, k2, ← c124649] at e_2044_5
  have hr := e_2044_5
  simp only [e_2044_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17376)) := by
  have c124650 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124651 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124652 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124653 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124654 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124655 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_6 := arithEq_of_rows h (row := 2044) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_6
  simp only [← c124650, ← c124651, ← c124652] at e_2044_6
  have e_2044_7 := arithEq_of_rows h (row := 2044) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_7
  simp only [← c124653, ← c124654, k2, ← c124655] at e_2044_7
  have hr := e_2044_7
  simp only [e_2044_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17377)) := by
  have c124656 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124657 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124658 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124659 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124660 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124661 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_8 := arithEq_of_rows h (row := 2044) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_8
  simp only [← c124656, ← c124657, ← c124658] at e_2044_8
  have e_2044_9 := arithEq_of_rows h (row := 2044) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_9
  simp only [← c124659, ← c124660, k2, ← c124661] at e_2044_9
  have hr := e_2044_9
  simp only [e_2044_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17378)) := by
  have c124662 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124663 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124664 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124665 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124666 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124667 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_10 := arithEq_of_rows h (row := 2044) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_10
  simp only [← c124662, ← c124663, ← c124664] at e_2044_10
  have e_2044_11 := arithEq_of_rows h (row := 2044) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_11
  simp only [← c124665, ← c124666, k2, ← c124667] at e_2044_11
  have hr := e_2044_11
  simp only [e_2044_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17379)) := by
  have c124668 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124669 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124670 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124671 := (publicBatchWrapper89_copies3895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124672 := (publicBatchWrapper89_copies3896 a h).1
  have c124673 := (publicBatchWrapper89_copies3896 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_12 := arithEq_of_rows h (row := 2044) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_12
  simp only [← c124668, ← c124669, ← c124670] at e_2044_12
  have e_2044_13 := arithEq_of_rows h (row := 2044) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_13
  simp only [← c124671, ← c124672, k2, ← c124673] at e_2044_13
  have hr := e_2044_13
  simp only [e_2044_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17380)) := by
  have c124674 := (publicBatchWrapper89_copies3896 a h).2.2.1
  have c124675 := (publicBatchWrapper89_copies3896 a h).2.2.2.1
  have c124676 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.1
  have c124677 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.1
  have c124678 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.1
  have c124679 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_14 := arithEq_of_rows h (row := 2044) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_14
  simp only [← c124674, ← c124675, ← c124676] at e_2044_14
  have e_2044_15 := arithEq_of_rows h (row := 2044) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_15
  simp only [← c124677, ← c124678, k2, ← c124679] at e_2044_15
  have hr := e_2044_15
  simp only [e_2044_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17381)) := by
  have c124680 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.1
  have c124681 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.1
  have c124682 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124683 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124684 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124685 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_16 := arithEq_of_rows h (row := 2044) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_16
  simp only [← c124680, ← c124681, ← c124682] at e_2044_16
  have e_2044_17 := arithEq_of_rows h (row := 2044) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_17
  simp only [← c124683, ← c124684, k2, ← c124685] at e_2044_17
  have hr := e_2044_17
  simp only [e_2044_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2044 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17382)) := by
  have c124686 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124687 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124688 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124689 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124690 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124691 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2044_18 := arithEq_of_rows h (row := 2044) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_18
  simp only [← c124686, ← c124687, ← c124688] at e_2044_18
  have e_2044_19 := arithEq_of_rows h (row := 2044) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2044_19
  simp only [← c124689, ← c124690, k2, ← c124691] at e_2044_19
  have hr := e_2044_19
  simp only [e_2044_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17383)) := by
  have c124692 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124693 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124694 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124695 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124696 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124697 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_0 := arithEq_of_rows h (row := 2045) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_0
  simp only [← c124692, ← c124693, ← c124694] at e_2045_0
  have e_2045_1 := arithEq_of_rows h (row := 2045) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_1
  simp only [← c124695, ← c124696, k2, ← c124697] at e_2045_1
  have hr := e_2045_1
  simp only [e_2045_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17384)) := by
  have c124698 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124699 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124700 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124701 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124702 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124703 := (publicBatchWrapper89_copies3896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_2 := arithEq_of_rows h (row := 2045) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_2
  simp only [← c124698, ← c124699, ← c124700] at e_2045_2
  have e_2045_3 := arithEq_of_rows h (row := 2045) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_3
  simp only [← c124701, ← c124702, k2, ← c124703] at e_2045_3
  have hr := e_2045_3
  simp only [e_2045_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17385)) := by
  have c124704 := (publicBatchWrapper89_copies3897 a h).1
  have c124705 := (publicBatchWrapper89_copies3897 a h).2.1
  have c124706 := (publicBatchWrapper89_copies3897 a h).2.2.1
  have c124707 := (publicBatchWrapper89_copies3897 a h).2.2.2.1
  have c124708 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.1
  have c124709 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_4 := arithEq_of_rows h (row := 2045) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_4
  simp only [← c124704, ← c124705, ← c124706] at e_2045_4
  have e_2045_5 := arithEq_of_rows h (row := 2045) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_5
  simp only [← c124707, ← c124708, k2, ← c124709] at e_2045_5
  have hr := e_2045_5
  simp only [e_2045_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17386)) := by
  have c124710 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.1
  have c124711 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.1
  have c124712 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.1
  have c124713 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.1
  have c124714 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124715 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_6 := arithEq_of_rows h (row := 2045) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_6
  simp only [← c124710, ← c124711, ← c124712] at e_2045_6
  have e_2045_7 := arithEq_of_rows h (row := 2045) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_7
  simp only [← c124713, ← c124714, k2, ← c124715] at e_2045_7
  have hr := e_2045_7
  simp only [e_2045_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17387)) := by
  have c124716 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124717 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124718 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124719 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124720 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124721 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_8 := arithEq_of_rows h (row := 2045) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_8
  simp only [← c124716, ← c124717, ← c124718] at e_2045_8
  have e_2045_9 := arithEq_of_rows h (row := 2045) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_9
  simp only [← c124719, ← c124720, k2, ← c124721] at e_2045_9
  have hr := e_2045_9
  simp only [e_2045_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17388)) := by
  have c124722 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124723 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124724 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124725 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124726 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124727 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_10 := arithEq_of_rows h (row := 2045) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_10
  simp only [← c124722, ← c124723, ← c124724] at e_2045_10
  have e_2045_11 := arithEq_of_rows h (row := 2045) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_11
  simp only [← c124725, ← c124726, k2, ← c124727] at e_2045_11
  have hr := e_2045_11
  simp only [e_2045_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17389)) := by
  have c124728 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124729 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124730 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124731 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124732 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124733 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_12 := arithEq_of_rows h (row := 2045) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_12
  simp only [← c124728, ← c124729, ← c124730] at e_2045_12
  have e_2045_13 := arithEq_of_rows h (row := 2045) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_13
  simp only [← c124731, ← c124732, k2, ← c124733] at e_2045_13
  have hr := e_2045_13
  simp only [e_2045_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17390)) := by
  have c124734 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124735 := (publicBatchWrapper89_copies3897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124736 := (publicBatchWrapper89_copies3898 a h).1
  have c124737 := (publicBatchWrapper89_copies3898 a h).2.1
  have c124738 := (publicBatchWrapper89_copies3898 a h).2.2.1
  have c124739 := (publicBatchWrapper89_copies3898 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_14 := arithEq_of_rows h (row := 2045) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_14
  simp only [← c124734, ← c124735, ← c124736] at e_2045_14
  have e_2045_15 := arithEq_of_rows h (row := 2045) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_15
  simp only [← c124737, ← c124738, k2, ← c124739] at e_2045_15
  have hr := e_2045_15
  simp only [e_2045_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17391)) := by
  have c124740 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.1
  have c124741 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.1
  have c124742 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.1
  have c124743 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.1
  have c124744 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.1
  have c124745 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_16 := arithEq_of_rows h (row := 2045) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_16
  simp only [← c124740, ← c124741, ← c124742] at e_2045_16
  have e_2045_17 := arithEq_of_rows h (row := 2045) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_17
  simp only [← c124743, ← c124744, k2, ← c124745] at e_2045_17
  have hr := e_2045_17
  simp only [e_2045_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 79) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17392)) := by
  have c124746 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124747 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124748 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124749 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124750 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124751 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_18 := arithEq_of_rows h (row := 2045) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_18
  simp only [← c124746, ← c124747, ← c124748] at e_2045_18
  have e_2045_19 := arithEq_of_rows h (row := 2045) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_19
  simp only [← c124749, ← c124750, k2, ← c124751] at e_2045_19
  have hr := e_2045_19
  simp only [e_2045_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 7) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17393)) := by
  have c124752 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124753 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124754 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124755 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124756 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124757 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_0 := arithEq_of_rows h (row := 2046) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_0
  simp only [← c124752, ← c124753, ← c124754] at e_2046_0
  have e_2046_1 := arithEq_of_rows h (row := 2046) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_1
  simp only [← c124755, ← c124756, k2, ← c124757] at e_2046_1
  have hr := e_2046_1
  simp only [e_2046_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 15) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17394)) := by
  have c124758 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124759 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124760 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124761 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124762 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124763 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_2 := arithEq_of_rows h (row := 2046) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_2
  simp only [← c124758, ← c124759, ← c124760] at e_2046_2
  have e_2046_3 := arithEq_of_rows h (row := 2046) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_3
  simp only [← c124761, ← c124762, k2, ← c124763] at e_2046_3
  have hr := e_2046_3
  simp only [e_2046_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 23) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17395)) := by
  have c124764 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124765 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124766 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124767 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124768 := (publicBatchWrapper89_copies3899 a h).1
  have c124769 := (publicBatchWrapper89_copies3899 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_4 := arithEq_of_rows h (row := 2046) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_4
  simp only [← c124764, ← c124765, ← c124766] at e_2046_4
  have e_2046_5 := arithEq_of_rows h (row := 2046) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_5
  simp only [← c124767, ← c124768, k2, ← c124769] at e_2046_5
  have hr := e_2046_5
  simp only [e_2046_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 31) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17396)) := by
  have c124770 := (publicBatchWrapper89_copies3899 a h).2.2.1
  have c124771 := (publicBatchWrapper89_copies3899 a h).2.2.2.1
  have c124772 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.1
  have c124773 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.1
  have c124774 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.1
  have c124775 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_6 := arithEq_of_rows h (row := 2046) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_6
  simp only [← c124770, ← c124771, ← c124772] at e_2046_6
  have e_2046_7 := arithEq_of_rows h (row := 2046) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_7
  simp only [← c124773, ← c124774, k2, ← c124775] at e_2046_7
  have hr := e_2046_7
  simp only [e_2046_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 39) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17397)) := by
  have c124776 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.1
  have c124777 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.1
  have c124778 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124779 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124780 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124781 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_8 := arithEq_of_rows h (row := 2046) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_8
  simp only [← c124776, ← c124777, ← c124778] at e_2046_8
  have e_2046_9 := arithEq_of_rows h (row := 2046) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_9
  simp only [← c124779, ← c124780, k2, ← c124781] at e_2046_9
  have hr := e_2046_9
  simp only [e_2046_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 47) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17398)) := by
  have c124782 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124783 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124784 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124785 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124786 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124787 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_10 := arithEq_of_rows h (row := 2046) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_10
  simp only [← c124782, ← c124783, ← c124784] at e_2046_10
  have e_2046_11 := arithEq_of_rows h (row := 2046) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_11
  simp only [← c124785, ← c124786, k2, ← c124787] at e_2046_11
  have hr := e_2046_11
  simp only [e_2046_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 55) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17399)) := by
  have c124788 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124789 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124790 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124791 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124792 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124793 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_12 := arithEq_of_rows h (row := 2046) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_12
  simp only [← c124788, ← c124789, ← c124790] at e_2046_12
  have e_2046_13 := arithEq_of_rows h (row := 2046) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_13
  simp only [← c124791, ← c124792, k2, ← c124793] at e_2046_13
  have hr := e_2046_13
  simp only [e_2046_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 63) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17400)) := by
  have c124794 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124795 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124796 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124797 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124798 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124799 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_14 := arithEq_of_rows h (row := 2046) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_14
  simp only [← c124794, ← c124795, ← c124796] at e_2046_14
  have e_2046_15 := arithEq_of_rows h (row := 2046) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_15
  simp only [← c124797, ← c124798, k2, ← c124799] at e_2046_15
  have hr := e_2046_15
  simp only [e_2046_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 71) = bselect (a (.wire 84 3)) (a (.virt 200)) (a (.virt 17401)) := by
  have c124800 := (publicBatchWrapper89_copies3900 a h).1
  have c124801 := (publicBatchWrapper89_copies3900 a h).2.1
  have c124802 := (publicBatchWrapper89_copies3900 a h).2.2.1
  have c124803 := (publicBatchWrapper89_copies3900 a h).2.2.2.1
  have c124804 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.1
  have c124805 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_16 := arithEq_of_rows h (row := 2046) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_16
  simp only [← c124800, ← c124801, ← c124802] at e_2046_16
  have e_2046_17 := arithEq_of_rows h (row := 2046) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_17
  simp only [← c124803, ← c124804, k2, ← c124805] at e_2046_17
  have hr := e_2046_17
  simp only [e_2046_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17570)) := by
  have c124806 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.1
  have c124807 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.1
  have c124808 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.1
  have c124809 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.1
  have c124810 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124811 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_18 := arithEq_of_rows h (row := 2046) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_18
  simp only [← c124806, ← c124807, ← c124808] at e_2046_18
  have e_2046_19 := arithEq_of_rows h (row := 2046) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_19
  simp only [← c124809, ← c124810, k2, ← c124811] at e_2046_19
  have hr := e_2046_19
  simp only [e_2046_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17571)) := by
  have c124812 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124813 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124814 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124815 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124816 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124817 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_0 := arithEq_of_rows h (row := 2047) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_0
  simp only [← c124812, ← c124813, ← c124814] at e_2047_0
  have e_2047_1 := arithEq_of_rows h (row := 2047) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_1
  simp only [← c124815, ← c124816, k2, ← c124817] at e_2047_1
  have hr := e_2047_1
  simp only [e_2047_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17572)) := by
  have c124818 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124819 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124820 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124821 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124822 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124823 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_2 := arithEq_of_rows h (row := 2047) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_2
  simp only [← c124818, ← c124819, ← c124820] at e_2047_2
  have e_2047_3 := arithEq_of_rows h (row := 2047) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_3
  simp only [← c124821, ← c124822, k2, ← c124823] at e_2047_3
  have hr := e_2047_3
  simp only [e_2047_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17573)) := by
  have c124824 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124825 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124826 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124827 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124828 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124829 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_4 := arithEq_of_rows h (row := 2047) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_4
  simp only [← c124824, ← c124825, ← c124826] at e_2047_4
  have e_2047_5 := arithEq_of_rows h (row := 2047) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_5
  simp only [← c124827, ← c124828, k2, ← c124829] at e_2047_5
  have hr := e_2047_5
  simp only [e_2047_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17574)) := by
  have c124830 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124831 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124832 := (publicBatchWrapper89_copies3901 a h).1
  have c124833 := (publicBatchWrapper89_copies3901 a h).2.1
  have c124834 := (publicBatchWrapper89_copies3901 a h).2.2.1
  have c124835 := (publicBatchWrapper89_copies3901 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_6 := arithEq_of_rows h (row := 2047) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_6
  simp only [← c124830, ← c124831, ← c124832] at e_2047_6
  have e_2047_7 := arithEq_of_rows h (row := 2047) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_7
  simp only [← c124833, ← c124834, k2, ← c124835] at e_2047_7
  have hr := e_2047_7
  simp only [e_2047_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17575)) := by
  have c124836 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.1
  have c124837 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.1
  have c124838 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.1
  have c124839 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.1
  have c124840 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.1
  have c124841 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_8 := arithEq_of_rows h (row := 2047) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_8
  simp only [← c124836, ← c124837, ← c124838] at e_2047_8
  have e_2047_9 := arithEq_of_rows h (row := 2047) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_9
  simp only [← c124839, ← c124840, k2, ← c124841] at e_2047_9
  have hr := e_2047_9
  simp only [e_2047_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17576)) := by
  have c124842 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124843 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124844 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124845 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124846 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124847 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_10 := arithEq_of_rows h (row := 2047) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_10
  simp only [← c124842, ← c124843, ← c124844] at e_2047_10
  have e_2047_11 := arithEq_of_rows h (row := 2047) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_11
  simp only [← c124845, ← c124846, k2, ← c124847] at e_2047_11
  have hr := e_2047_11
  simp only [e_2047_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17577)) := by
  have c124848 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124849 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124850 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124851 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124852 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124853 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_12 := arithEq_of_rows h (row := 2047) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_12
  simp only [← c124848, ← c124849, ← c124850] at e_2047_12
  have e_2047_13 := arithEq_of_rows h (row := 2047) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_13
  simp only [← c124851, ← c124852, k2, ← c124853] at e_2047_13
  have hr := e_2047_13
  simp only [e_2047_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17578)) := by
  have c124854 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124855 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124856 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124857 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124858 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124859 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_14 := arithEq_of_rows h (row := 2047) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_14
  simp only [← c124854, ← c124855, ← c124856] at e_2047_14
  have e_2047_15 := arithEq_of_rows h (row := 2047) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_15
  simp only [← c124857, ← c124858, k2, ← c124859] at e_2047_15
  have hr := e_2047_15
  simp only [e_2047_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17579)) := by
  have c124860 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124861 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124862 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124863 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124864 := (publicBatchWrapper89_copies3902 a h).1
  have c124865 := (publicBatchWrapper89_copies3902 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_16 := arithEq_of_rows h (row := 2047) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_16
  simp only [← c124860, ← c124861, ← c124862] at e_2047_16
  have e_2047_17 := arithEq_of_rows h (row := 2047) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_17
  simp only [← c124863, ← c124864, k2, ← c124865] at e_2047_17
  have hr := e_2047_17
  simp only [e_2047_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17580)) := by
  have c124866 := (publicBatchWrapper89_copies3902 a h).2.2.1
  have c124867 := (publicBatchWrapper89_copies3902 a h).2.2.2.1
  have c124868 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.1
  have c124869 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.1
  have c124870 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.1
  have c124871 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_18 := arithEq_of_rows h (row := 2047) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_18
  simp only [← c124866, ← c124867, ← c124868] at e_2047_18
  have e_2047_19 := arithEq_of_rows h (row := 2047) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_19
  simp only [← c124869, ← c124870, k2, ← c124871] at e_2047_19
  have hr := e_2047_19
  simp only [e_2047_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17581)) := by
  have c124872 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.1
  have c124873 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.1
  have c124874 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124875 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124876 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124877 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_0 := arithEq_of_rows h (row := 2048) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_0
  simp only [← c124872, ← c124873, ← c124874] at e_2048_0
  have e_2048_1 := arithEq_of_rows h (row := 2048) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_1
  simp only [← c124875, ← c124876, k2, ← c124877] at e_2048_1
  have hr := e_2048_1
  simp only [e_2048_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17582)) := by
  have c124878 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124879 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124880 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124881 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124882 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124883 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_2 := arithEq_of_rows h (row := 2048) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_2
  simp only [← c124878, ← c124879, ← c124880] at e_2048_2
  have e_2048_3 := arithEq_of_rows h (row := 2048) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_3
  simp only [← c124881, ← c124882, k2, ← c124883] at e_2048_3
  have hr := e_2048_3
  simp only [e_2048_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17583)) := by
  have c124884 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124885 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124886 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124887 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124888 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124889 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_4 := arithEq_of_rows h (row := 2048) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_4
  simp only [← c124884, ← c124885, ← c124886] at e_2048_4
  have e_2048_5 := arithEq_of_rows h (row := 2048) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_5
  simp only [← c124887, ← c124888, k2, ← c124889] at e_2048_5
  have hr := e_2048_5
  simp only [e_2048_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17584)) := by
  have c124890 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124891 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124892 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124893 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124894 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124895 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_6 := arithEq_of_rows h (row := 2048) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_6
  simp only [← c124890, ← c124891, ← c124892] at e_2048_6
  have e_2048_7 := arithEq_of_rows h (row := 2048) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_7
  simp only [← c124893, ← c124894, k2, ← c124895] at e_2048_7
  have hr := e_2048_7
  simp only [e_2048_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17585)) := by
  have c124896 := (publicBatchWrapper89_copies3903 a h).1
  have c124897 := (publicBatchWrapper89_copies3903 a h).2.1
  have c124898 := (publicBatchWrapper89_copies3903 a h).2.2.1
  have c124899 := (publicBatchWrapper89_copies3903 a h).2.2.2.1
  have c124900 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.1
  have c124901 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_8 := arithEq_of_rows h (row := 2048) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_8
  simp only [← c124896, ← c124897, ← c124898] at e_2048_8
  have e_2048_9 := arithEq_of_rows h (row := 2048) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_9
  simp only [← c124899, ← c124900, k2, ← c124901] at e_2048_9
  have hr := e_2048_9
  simp only [e_2048_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17586)) := by
  have c124902 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.1
  have c124903 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.1
  have c124904 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.1
  have c124905 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.1
  have c124906 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124907 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_10 := arithEq_of_rows h (row := 2048) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_10
  simp only [← c124902, ← c124903, ← c124904] at e_2048_10
  have e_2048_11 := arithEq_of_rows h (row := 2048) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_11
  simp only [← c124905, ← c124906, k2, ← c124907] at e_2048_11
  have hr := e_2048_11
  simp only [e_2048_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17587)) := by
  have c124908 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124909 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124910 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124911 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124912 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124913 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_12 := arithEq_of_rows h (row := 2048) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_12
  simp only [← c124908, ← c124909, ← c124910] at e_2048_12
  have e_2048_13 := arithEq_of_rows h (row := 2048) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_13
  simp only [← c124911, ← c124912, k2, ← c124913] at e_2048_13
  have hr := e_2048_13
  simp only [e_2048_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17588)) := by
  have c124914 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124915 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124916 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124917 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124918 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124919 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_14 := arithEq_of_rows h (row := 2048) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_14
  simp only [← c124914, ← c124915, ← c124916] at e_2048_14
  have e_2048_15 := arithEq_of_rows h (row := 2048) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_15
  simp only [← c124917, ← c124918, k2, ← c124919] at e_2048_15
  have hr := e_2048_15
  simp only [e_2048_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17589)) := by
  have c124920 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124921 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124922 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124923 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124924 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124925 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_16 := arithEq_of_rows h (row := 2048) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_16
  simp only [← c124920, ← c124921, ← c124922] at e_2048_16
  have e_2048_17 := arithEq_of_rows h (row := 2048) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_17
  simp only [← c124923, ← c124924, k2, ← c124925] at e_2048_17
  have hr := e_2048_17
  simp only [e_2048_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17590)) := by
  have c124926 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124927 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124928 := (publicBatchWrapper89_copies3904 a h).1
  have c124929 := (publicBatchWrapper89_copies3904 a h).2.1
  have c124930 := (publicBatchWrapper89_copies3904 a h).2.2.1
  have c124931 := (publicBatchWrapper89_copies3904 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_18 := arithEq_of_rows h (row := 2048) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_18
  simp only [← c124926, ← c124927, ← c124928] at e_2048_18
  have e_2048_19 := arithEq_of_rows h (row := 2048) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_19
  simp only [← c124929, ← c124930, k2, ← c124931] at e_2048_19
  have hr := e_2048_19
  simp only [e_2048_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17591)) := by
  have c124932 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.1
  have c124933 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.1
  have c124934 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.1
  have c124935 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.1
  have c124936 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.1
  have c124937 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_0 := arithEq_of_rows h (row := 2049) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_0
  simp only [← c124932, ← c124933, ← c124934] at e_2049_0
  have e_2049_1 := arithEq_of_rows h (row := 2049) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_1
  simp only [← c124935, ← c124936, k2, ← c124937] at e_2049_1
  have hr := e_2049_1
  simp only [e_2049_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17592)) := by
  have c124938 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124939 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124940 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124941 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124942 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124943 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_2 := arithEq_of_rows h (row := 2049) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_2
  simp only [← c124938, ← c124939, ← c124940] at e_2049_2
  have e_2049_3 := arithEq_of_rows h (row := 2049) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_3
  simp only [← c124941, ← c124942, k2, ← c124943] at e_2049_3
  have hr := e_2049_3
  simp only [e_2049_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17593)) := by
  have c124944 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124945 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124946 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124947 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124948 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124949 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_4 := arithEq_of_rows h (row := 2049) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_4
  simp only [← c124944, ← c124945, ← c124946] at e_2049_4
  have e_2049_5 := arithEq_of_rows h (row := 2049) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_5
  simp only [← c124947, ← c124948, k2, ← c124949] at e_2049_5
  have hr := e_2049_5
  simp only [e_2049_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17594)) := by
  have c124950 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124951 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124952 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124953 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124954 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124955 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_6 := arithEq_of_rows h (row := 2049) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_6
  simp only [← c124950, ← c124951, ← c124952] at e_2049_6
  have e_2049_7 := arithEq_of_rows h (row := 2049) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_7
  simp only [← c124953, ← c124954, k2, ← c124955] at e_2049_7
  have hr := e_2049_7
  simp only [e_2049_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17595)) := by
  have c124956 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124957 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124958 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124959 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124960 := (publicBatchWrapper89_copies3905 a h).1
  have c124961 := (publicBatchWrapper89_copies3905 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_8 := arithEq_of_rows h (row := 2049) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_8
  simp only [← c124956, ← c124957, ← c124958] at e_2049_8
  have e_2049_9 := arithEq_of_rows h (row := 2049) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_9
  simp only [← c124959, ← c124960, k2, ← c124961] at e_2049_9
  have hr := e_2049_9
  simp only [e_2049_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17596)) := by
  have c124962 := (publicBatchWrapper89_copies3905 a h).2.2.1
  have c124963 := (publicBatchWrapper89_copies3905 a h).2.2.2.1
  have c124964 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.1
  have c124965 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.1
  have c124966 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.1
  have c124967 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_10 := arithEq_of_rows h (row := 2049) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_10
  simp only [← c124962, ← c124963, ← c124964] at e_2049_10
  have e_2049_11 := arithEq_of_rows h (row := 2049) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_11
  simp only [← c124965, ← c124966, k2, ← c124967] at e_2049_11
  have hr := e_2049_11
  simp only [e_2049_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17597)) := by
  have c124968 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.1
  have c124969 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.1
  have c124970 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124971 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124972 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124973 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_12 := arithEq_of_rows h (row := 2049) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_12
  simp only [← c124968, ← c124969, ← c124970] at e_2049_12
  have e_2049_13 := arithEq_of_rows h (row := 2049) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_13
  simp only [← c124971, ← c124972, k2, ← c124973] at e_2049_13
  have hr := e_2049_13
  simp only [e_2049_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17598)) := by
  have c124974 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124975 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124976 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124977 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124978 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124979 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_14 := arithEq_of_rows h (row := 2049) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_14
  simp only [← c124974, ← c124975, ← c124976] at e_2049_14
  have e_2049_15 := arithEq_of_rows h (row := 2049) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_15
  simp only [← c124977, ← c124978, k2, ← c124979] at e_2049_15
  have hr := e_2049_15
  simp only [e_2049_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17599)) := by
  have c124980 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124981 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124982 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124983 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124984 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124985 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_16 := arithEq_of_rows h (row := 2049) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_16
  simp only [← c124980, ← c124981, ← c124982] at e_2049_16
  have e_2049_17 := arithEq_of_rows h (row := 2049) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_17
  simp only [← c124983, ← c124984, k2, ← c124985] at e_2049_17
  have hr := e_2049_17
  simp only [e_2049_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17600)) := by
  have c124986 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124987 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124988 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124989 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124990 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124991 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_18 := arithEq_of_rows h (row := 2049) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_18
  simp only [← c124986, ← c124987, ← c124988] at e_2049_18
  have e_2049_19 := arithEq_of_rows h (row := 2049) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_19
  simp only [← c124989, ← c124990, k2, ← c124991] at e_2049_19
  have hr := e_2049_19
  simp only [e_2049_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17601)) := by
  have c124992 := (publicBatchWrapper89_copies3906 a h).1
  have c124993 := (publicBatchWrapper89_copies3906 a h).2.1
  have c124994 := (publicBatchWrapper89_copies3906 a h).2.2.1
  have c124995 := (publicBatchWrapper89_copies3906 a h).2.2.2.1
  have c124996 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.1
  have c124997 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_0 := arithEq_of_rows h (row := 2050) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_0
  simp only [← c124992, ← c124993, ← c124994] at e_2050_0
  have e_2050_1 := arithEq_of_rows h (row := 2050) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_1
  simp only [← c124995, ← c124996, k2, ← c124997] at e_2050_1
  have hr := e_2050_1
  simp only [e_2050_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17770)) := by
  have c124998 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.1
  have c124999 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.1
  have c125000 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.1
  have c125001 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.1
  have c125002 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125003 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_2 := arithEq_of_rows h (row := 2050) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_2
  simp only [← c124998, ← c124999, ← c125000] at e_2050_2
  have e_2050_3 := arithEq_of_rows h (row := 2050) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_3
  simp only [← c125001, ← c125002, k2, ← c125003] at e_2050_3
  have hr := e_2050_3
  simp only [e_2050_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17771)) := by
  have c125004 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125005 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125006 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125007 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125008 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125009 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_4 := arithEq_of_rows h (row := 2050) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_4
  simp only [← c125004, ← c125005, ← c125006] at e_2050_4
  have e_2050_5 := arithEq_of_rows h (row := 2050) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_5
  simp only [← c125007, ← c125008, k2, ← c125009] at e_2050_5
  have hr := e_2050_5
  simp only [e_2050_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17772)) := by
  have c125010 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125011 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125012 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125013 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125014 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125015 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_6 := arithEq_of_rows h (row := 2050) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_6
  simp only [← c125010, ← c125011, ← c125012] at e_2050_6
  have e_2050_7 := arithEq_of_rows h (row := 2050) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_7
  simp only [← c125013, ← c125014, k2, ← c125015] at e_2050_7
  have hr := e_2050_7
  simp only [e_2050_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17773)) := by
  have c125016 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125017 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125018 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125019 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125020 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125021 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_8 := arithEq_of_rows h (row := 2050) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_8
  simp only [← c125016, ← c125017, ← c125018] at e_2050_8
  have e_2050_9 := arithEq_of_rows h (row := 2050) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_9
  simp only [← c125019, ← c125020, k2, ← c125021] at e_2050_9
  have hr := e_2050_9
  simp only [e_2050_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17774)) := by
  have c125022 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125023 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125024 := (publicBatchWrapper89_copies3907 a h).1
  have c125025 := (publicBatchWrapper89_copies3907 a h).2.1
  have c125026 := (publicBatchWrapper89_copies3907 a h).2.2.1
  have c125027 := (publicBatchWrapper89_copies3907 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_10 := arithEq_of_rows h (row := 2050) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_10
  simp only [← c125022, ← c125023, ← c125024] at e_2050_10
  have e_2050_11 := arithEq_of_rows h (row := 2050) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_11
  simp only [← c125025, ← c125026, k2, ← c125027] at e_2050_11
  have hr := e_2050_11
  simp only [e_2050_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17775)) := by
  have c125028 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.1
  have c125029 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.1
  have c125030 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.1
  have c125031 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.1
  have c125032 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.1
  have c125033 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_12 := arithEq_of_rows h (row := 2050) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_12
  simp only [← c125028, ← c125029, ← c125030] at e_2050_12
  have e_2050_13 := arithEq_of_rows h (row := 2050) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_13
  simp only [← c125031, ← c125032, k2, ← c125033] at e_2050_13
  have hr := e_2050_13
  simp only [e_2050_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17776)) := by
  have c125034 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125035 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125036 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125037 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125038 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125039 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_14 := arithEq_of_rows h (row := 2050) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_14
  simp only [← c125034, ← c125035, ← c125036] at e_2050_14
  have e_2050_15 := arithEq_of_rows h (row := 2050) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_15
  simp only [← c125037, ← c125038, k2, ← c125039] at e_2050_15
  have hr := e_2050_15
  simp only [e_2050_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17777)) := by
  have c125040 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125041 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125042 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125043 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125044 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125045 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_16 := arithEq_of_rows h (row := 2050) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_16
  simp only [← c125040, ← c125041, ← c125042] at e_2050_16
  have e_2050_17 := arithEq_of_rows h (row := 2050) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_17
  simp only [← c125043, ← c125044, k2, ← c125045] at e_2050_17
  have hr := e_2050_17
  simp only [e_2050_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17778)) := by
  have c125046 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125047 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125048 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125049 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125050 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125051 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_18 := arithEq_of_rows h (row := 2050) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_18
  simp only [← c125046, ← c125047, ← c125048] at e_2050_18
  have e_2050_19 := arithEq_of_rows h (row := 2050) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_19
  simp only [← c125049, ← c125050, k2, ← c125051] at e_2050_19
  have hr := e_2050_19
  simp only [e_2050_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17779)) := by
  have c125052 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125053 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125054 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125055 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125056 := (publicBatchWrapper89_copies3908 a h).1
  have c125057 := (publicBatchWrapper89_copies3908 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_0 := arithEq_of_rows h (row := 2051) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_0
  simp only [← c125052, ← c125053, ← c125054] at e_2051_0
  have e_2051_1 := arithEq_of_rows h (row := 2051) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_1
  simp only [← c125055, ← c125056, k2, ← c125057] at e_2051_1
  have hr := e_2051_1
  simp only [e_2051_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f20092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17780)) := by
  have c125058 := (publicBatchWrapper89_copies3908 a h).2.2.1
  have c125059 := (publicBatchWrapper89_copies3908 a h).2.2.2.1
  have c125060 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.1
  have c125061 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.1
  have c125062 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.1
  have c125063 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_2 := arithEq_of_rows h (row := 2051) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_2
  simp only [← c125058, ← c125059, ← c125060] at e_2051_2
  have e_2051_3 := arithEq_of_rows h (row := 2051) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_3
  simp only [← c125061, ← c125062, k2, ← c125063] at e_2051_3
  have hr := e_2051_3
  simp only [e_2051_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
